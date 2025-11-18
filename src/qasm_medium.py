from qiskit.providers.fake_provider import GenericBackendV2  # lives here
from qiskit import QuantumCircuit, transpile
# visualize_layout.py
from qiskit.visualization import plot_coupling_map
import matplotlib
matplotlib.use("Agg")              # call BEFORE importing pyplot
import matplotlib.pyplot as plt
from qiskit.transpiler import CouplingMap
from scheduler import *
from process import process
import numpy as np
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error,pauli_error
from fakeHardware import construct_10_qubit_hardware, construct_20_qubit_hardware, plot_process_schedule_on_20_qubit_hardware, save_circuit_png, plot_process_schedule_on_10_qubit_hardware, build_noise_model,distribution_fidelity
from instruction import *


label_name_map = {
    1: "bigadder_n18",
    2: "bv_n14",
    3: "bv_n19",
    4: "cat_state_n22",
    5: "cc_n12",
    6: "dnn_n16",
    7: "gcm_h6",
    8: "ghz_state_n23",
    9: "ising_n26",
    10: "knn_n25",
    11: "multiplier_n15",
    12: "multiply_n13",
    13: "qec9xz_n17",
    14: "qf21_n15",
    15: "qft_n18",
    16: "qram_n20",
    17: "sat_n11",
    18: "seca_n11",
    19: "square_root_n18",
    20: "swap_test_n25",
    21: "wstate_n27"
}



def parse_qasm_instruction(shots: int,process_ID: int,instruction_str: str) -> process:
    """
    Parse a QASM instruction string and return an Instruction object.
    
    Args:
        instruction_str (str): The QASM instruction string.
    """

    circuit = qiskit.qasm2.loads(instruction_str)
    qubit_number = circuit.num_qubits

    vdata = virtualSpace(size=qubit_number, label="vdata")    
    vdata.allocate_range(0, qubit_number - 1)
    vsyn = virtualSpace(size=1, label="vsyn", is_syndrome=True)
    vsyn.allocate_range(0, 0)
    proc = process(processID=process_ID, start_time=0, vdataspace=vdata, vsyndromespace=vsyn, shots=shots)
    proc.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata.get_address(0)],size=qubit_number,processID=process_ID))  # Allocate 2 data qubits
    proc.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn.get_address(0)],size=1,processID=process_ID))  # Allocate 1 syndrome qubit


    for instr, qargs, cargs in circuit.data:
        name = instr.name.lower()
        # Map QASM instruction names to Instruction types
        #print(instr)
        if name == "h":
            proc.add_instruction(Instype.H, [vdata.get_address(qargs[0]._index)])
        elif name == "x":
            proc.add_instruction(Instype.X, [vdata.get_address(qargs[0]._index)])
        elif name == "y":
            proc.add_instruction(Instype.Y, [vdata.get_address(qargs[0]._index)])
        elif name == "z":
            proc.add_instruction(Instype.Z, [vdata.get_address(qargs[0]._index)])
        elif name == "t":
            proc.add_instruction(Instype.T, [vdata.get_address(qargs[0]._index)])
        elif name == "tdg":
            proc.add_instruction(Instype.Tdg, [vdata.get_address(qargs[0]._index)])
        elif name == "s":
            proc.add_instruction(Instype.S, [vdata.get_address(qargs[0]._index)])
        elif name == "sdg":
            proc.add_instruction(Instype.Sdg, [vdata.get_address(qargs[0]._index)])
        elif name == "sx":
            proc.add_instruction(Instype.SX, [vdata.get_address(qargs[0]._index)])
        elif name == "rz":
            proc.add_instruction(Instype.RZ, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "rx":
            #print(instr)
            proc.add_instruction(Instype.RX, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "ry":
            proc.add_instruction(Instype.RY, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "u":
            proc.add_instruction(Instype.U, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "u3":
            proc.add_instruction(Instype.U3, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "ccx":
            proc.add_instruction(Instype.Toffoli, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index), vdata.get_address(qargs[2]._index)])
        elif name == "cx":
            proc.add_instruction(Instype.CNOT, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)])
        elif name == "ch":
            proc.add_instruction(Instype.CH, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)])
        elif name == "swap":
            proc.add_instruction(Instype.SWAP, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)])
        elif name == "cswap":
            proc.add_instruction(Instype.CSWAP, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index), vdata.get_address(qargs[2]._index)])
        elif name == "cu1":
            proc.add_instruction(Instype.CU1, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)], params=instr.params)
        elif name == "reset":
            proc.add_instruction(Instype.RESET, [vdata.get_address(qargs[0]._index)])
        elif name == "measure":
            proc.add_instruction(Instype.MEASURE, [vdata.get_address(qargs[0]._index)], classical_address=cargs[0]._index)
        elif name == "barrier":
            continue
        else:
            raise ValueError(f"Unsupported instruction: {name}")

    proc.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata.get_address(0)],size=qubit_number ,processID=process_ID))  # Allocate 2 data qubits
    proc.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn.get_address(0)],size=1,processID=process_ID))  # Allocate 2 syndrome qubits

    return proc





if __name__ == "__main__":


    for id in range(2, 22):
        label_id =  id
        label_name = label_name_map[label_id]
        print(f"Testing QASM file: {label_name}.qasm")
        file_path = f"C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\mediumqasm\\{label_name}.qasm"
        with open(file_path, "r") as file:
            qasm_code = file.read()   
        shots =  1000
        proc_instance = parse_qasm_instruction(shots=shots, process_ID=1, instruction_str=qasm_code)

