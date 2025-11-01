#Compile a process instance written in standard format.


import re
from scheduler import *
from process import process
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_coupling_map
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



def compile_quantum_script(input_script: str) -> process:
    lines = [line.strip() for line in input_script.strip().split('\n') if line.strip()]

    spaces = {}  # var -> (is_syn, size)
    var_to_space = {}  # var -> virtualSpace object
    instructions = []
    deallocates = []
    shots = None

    for line in lines:
        if line.startswith('set_shot('):
            shots = int(re.search(r'\((\d+)\)', line).group(1))
            # Note: shots is parsed but not used in the process construction
        elif line.startswith('deallocate_data('):
            var = re.search(r'\((.*?)\)', line).group(1).strip()
            deallocates.append((False, var))
        elif line.startswith('deallocate_helper('):
            var = re.search(r'\((.*?)\)', line).group(1).strip()
            deallocates.append((True, var))
        elif '=' in line:
            left, right = [part.strip() for part in line.split('=', 1)]
            if right.startswith('alloc_data('):
                size = int(re.search(r'\((\d+)\)', right).group(1))
                spaces[left] = (False, size)
            elif right.startswith('alloc_helper('):
                size = int(re.search(r'\((\d+)\)', right).group(1))
                spaces[left] = (True, size)
            elif right.startswith('MEASURE '):
                meas_arg = right.split(' ')[1].strip()
                instructions.append(('MEASURE', [meas_arg], left))  # assignment ignored
        else:
            # gate instruction
            parts = re.split(r'\s+', line)
            gate = parts[0]
            args_str = ' '.join(parts[1:])
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
            instructions.append((gate, args, None))

    # Create virtual spaces
    for var, (is_syn, size) in spaces.items():
        vspace = virtualSpace(size=size, label=var, is_syndrome=is_syn)
        vspace.allocate_range(0, size - 1)
        var_to_space[var] = vspace

    # Determine data and syndrome spaces (assuming one of each)
    data_var = next(var for var, (is_syn, _) in spaces.items() if not is_syn)
    syn_var = next(var for var, (is_syn, _) in spaces.items() if is_syn)
    vdataspace = var_to_space[data_var]
    vsyndromespace = var_to_space[syn_var]

    # Create process
    proc = process(processID=1, start_time=0, vdataspace=vdataspace, vsyndromespace=vsyndromespace)

    # Add allocation syscalls
    for var, (is_syn, size) in spaces.items():
        vspace = var_to_space[var]
        addresses = [vspace.get_address(i) for i in range(size)]
        if not is_syn:
            syscall = syscall_allocate_data_qubits(address=addresses, size=size, processID=1)
        else:
            syscall = syscall_allocate_syndrome_qubits(address=addresses, size=size, processID=1)
        proc.add_syscall(syscallinst=syscall)

    # Add instructions
    for gate, args, assign in instructions:
        addrs = []
        for arg in args:
            match = re.match(r'(\w+)(\d+)', arg)
            if match:
                space_var, index = match.groups()
                vspace = var_to_space[space_var]
                addr = vspace.get_address(int(index))
                addrs.append(addr)
        inst_type = getattr(Instype, gate)
        proc.add_instruction(inst_type, addrs)

    # Add deallocation syscalls
    for is_syn, var in deallocates:
        size = spaces[var][1]
        vspace = var_to_space[var]
        addresses = [vspace.get_address(i) for i in range(size)]
        if not is_syn:
            syscall = syscall_deallocate_data_qubits(address=addresses, size=size, processID=1)
        else:
            syscall = syscall_deallocate_syndrome_qubits(address=addresses, size=size, processID=1)
        proc.add_syscall(syscallinst=syscall)

    return proc






def compile_qasm3_no_helperqubit(input_script: str) -> process:
    pass






#Test code

if __name__ == "__main__":
    path="C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\example1"
    with open(path, 'r') as file:
        test_script = file.read()

    compiled_process = compile_quantum_script(test_script)
    print(compiled_process)