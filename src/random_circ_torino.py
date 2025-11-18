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
from fakeHardware import construct_fake_ibm_torino, save_circuit_png, plot_process_schedule_on_torino, build_noise_model,distribution_fidelity
from fakeHardware import construct_10_qubit_hardware
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime


import re





APIKEY ="zkIgM0xZIJfR0CgMMvD7A6N-76pgelZ10cAp9gt1fywy"


label_name_map = {
    1: "test1",
    2: "test2",
    3: "test3",
    4: "test4",
    5: "test5",
    6: "test6",
    7: "test7"
}




def compile_quantum_script(processID:int,input_script: str) -> process:
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
    proc = process(processID=processID, start_time=0, vdataspace=vdataspace, vsyndromespace=vsyndromespace)

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




def torino_coupling_map():
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], # The first long row
        [0,15], [15,19], [4,16], [16,23], [8,17], [17,27], [12,18], [18,31], # Short row 1
        [19,20], [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], # The second long row
        [21,34], [34,40], [25,35], [35,44], [29,36], [36,48], [33,37], [37,52], # Short row 2
        [38,39], [39,40], [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], # The third long row
        [38,53], [53,57], [42,54], [54,61], [46,55], [55,65], [50,56], [56,69], # Short row 3
        [57,58], [58,59], [59,60], [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], # The forth long row
        [59,72], [72,78], [63,73], [73,82], [67,74], [74,86], [71,75], [75,90], # Short row 4
        [76,77], [77,78], [78,79], [79,80], [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], # The fifth long row
        [76,91], [91,95], [80,92], [92,99], [84,93], [93,103], [88,94], [94,107], # Short row 5
        [95,96], [96,97], [97,98], [98,99], [99,100], [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], # The sixth long row
        [97,110], [110,116], [101,111], [111,120], [105,112], [112,124],[109,113], [113,128], # Short row 6
        [114,115], [115,116], [116,117], [117,118], [118,119], [119,120], [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], # The seventh long row
        [114,129], [118, 130], [122,131], [126,132]  # Short row 7
    ]
    return COUPLING



def _as_dt(x):
    """Parse Qiskit metrics timestamps to timezone-aware datetime or None."""
    if x is None:
        return None
    if hasattr(x, "tzinfo"):  # already a datetime
        return x
    if isinstance(x, str):
        # Handle trailing 'Z' and general ISO-8601
        s = x.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            try:
                # Optional fallback if python-dateutil is available
                from dateutil import parser as _parser
                return _parser.isoparse(x)
            except Exception:
                return None
    return None




def generate_random_kernel(num_proc,max_shot):
    """
    Randomly generate a kernel with specified number of processes.
    All those processes are selected randomly from the small QASM benchmark set.
    Args:
        num_proc (int): Number of processes in the kernel.
        max_shot (int): Maximum number of shots for each process.
    """

    COUPLING = torino_coupling_map()


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})

    for pid in range(1, num_proc + 1):
        label_id = np.random.randint(1, 8)  # Assuming there are 7 benchmark QASM files
        label_name = label_name_map[label_id]
        file_path = f"C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\randomShare\\{label_name}.qasm"
        with open(file_path, "r") as file:
            qasm_code = file.read()   
        shots = np.random.randint(800, max_shot + 1)
        proc_instance = parse_qasm_instruction(shots=shots, process_ID=pid, instruction_str=qasm_code)
        kernel_instance.add_process(proc_instance)

    virtual_hardware = virtualHardware(qubit_number=133, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware










def test_scheduling(kernel_instance, virtual_hardware, baseline=False, consider_connectivity=True, share_syndrome_qubits=True):


    #kernel_instance, virtual_hardware = generate_example_ppt10_on_10_qubit_device()

    schedule_instance = Scheduler(kernel_instance=kernel_instance, hardware_instance=virtual_hardware)


    dis=schedule_instance.calculate_all_pair_distance()
    total_time=0
    total_wait_time=0
    total_wall_time=0
    while not kernel_instance.processes_all_finished():
        if baseline:
            time1, inst_list1, shots=schedule_instance.baseline_scheduling()
        else:
            if consider_connectivity:
                time1, inst_list1, shots=schedule_instance.dynamic_scheduling()
            else:
                if share_syndrome_qubits:
                    time1, inst_list1, shots=schedule_instance.dynamic_scheduling_no_consider_connectivity()
                else:
                    time1, inst_list1, shots=schedule_instance.scheduling_with_out_sharing_syndrome_qubit()


        schedule_instance.print_dynamic_instruction_list(inst_list1)
        qc=schedule_instance.construct_qiskit_circuit_for_backend(inst_list1)



        # fig_t = qc.draw(output="mpl", fold=-1)
        # fig_t.savefig("before_transpiled.png", dpi=200, bbox_inches="tight")
        # plt.close(fig_t)

        # qc.draw("mpl", fold=-1).show()
        # print(qc.num_qubits)

        # 0) Fake 156-qubit backend (your Pittsburgh layout)
        fake_hard_ware = construct_fake_ibm_torino()


        # 1) Build the abstract (logical) circuit and save as PNG
        # qc = build_dynamic_circuit_15()
        # save_circuit_png(qc, "abstract_circuit.png")  # uses Matplotlib

        # 2) Transpile to hardware; map 15 logical qubits onto a single long row
        #    (contiguous physical qubits minimize SWAPs on your lattice)
        initial_layout = [i for i in range(133)]  # logical i -> physical i



        transpiled = transpile(
            qc,
            backend= fake_hard_ware,
            initial_layout=initial_layout,
            optimization_level=3,
        )
        print("\n=== Transpiled circuit ===")
        print(transpiled)

        # Save the transpiled circuit PNG too
        # import matplotlib.pyplot as plt
        # fig_t = transpiled.draw(output="mpl", fold=-1)
        # fig_t.savefig("transpiled_circuit.png", dpi=200, bbox_inches="tight")
        # plt.close(fig_t)



        process_list = schedule_instance.get_all_processes()
        syndrome_history = schedule_instance.get_syndrome_map_history()
        plot_process_schedule_on_torino(
            coupling_edges= fake_hard_ware.coupling_map,
            syndrome_qubit_history=syndrome_history,
            process_list=process_list,
            out_png="hardware_processes.png",
        )
        

        service = QiskitRuntimeService(channel="ibm_cloud",token=APIKEY)
        
        #backend = service.least_busy(simulator=False, operational=True)

        backend = service.backend("ibm_torino")

        # Convert to an ISA circuit and layout-mapped observables.
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        isa_circuit = pm.run(transpiled)
        
        # run SamplerV2 on the chosen backend
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots

        job = sampler.run([isa_circuit])
        # Ensure it’s done so metrics/timestamps are populated
        job.wait_for_final_state()


        # --- timings ---
        # 1) Quantum execution time (QPU time)
        quantum_sec = float(job.usage() or 0.0)

        # 2) Metrics/timestamps for waiting + wall time
        mets = job.metrics() or {}
        ts = mets.get("timestamps", {}) or {}

        created  = _as_dt(ts.get("created"))
        queued   = _as_dt(ts.get("queued"))
        running  = _as_dt(ts.get("running"))
        finished = _as_dt(ts.get("finished"))


        # Waiting time: queued->running if available, else created->running as fallback
        if running:
            if queued:
                wait_sec = (running - queued).total_seconds()
            elif created:
                wait_sec = (running - created).total_seconds()
            else:
                wait_sec = 0.0
        else:
            wait_sec = 0.0  # couldn't determine

        # Wall time: created->finished if both present; fallback to running->finished
        if created and finished:
            wall_sec = (finished - created).total_seconds()
        elif running and finished:
            wall_sec = (finished - running).total_seconds()
        else:
            wall_sec = 0.0


        # Accumulate
        total_time += quantum_sec
        total_wait_time += max(0.0, wait_sec)
        total_wall_time += max(0.0, wall_sec)


        # --- results ---
        pub = job.result()[0]  # first (and only) PUB result
        counts = pub.join_data().get_counts()
        final_result = schedule_instance.return_measure_states(counts)
        print(counts)

        kernel_instance.update_process_results(final_result)
        kernel_instance.reset_all_processes()
        schedule_instance.reset_all_states()



    # print(schedule_instance._measure_index_to_process)
    # print(schedule_instance._process_measure_index)



    final_result = kernel_instance._process_result_count
    ideal_result=schedule_instance.return_process_ideal_output()
    #print(ideal_result)



    average_fidelity=0
    for pid in final_result.keys():
        print("Ideal result for process ", pid)
        print(ideal_result[pid])
        print("Final result for process ", pid)
        print(final_result[pid])
        fidelity=distribution_fidelity(final_result[pid], ideal_result[pid])
        average_fidelity+=fidelity
        print(f"Fidelity for process {pid}: {fidelity:.4f}")




    print("The TRANSPILED circuit depth is:", transpiled.depth())
    print("\n=== Time taken:===")
    print(total_time)

    average_fidelity/=len(final_result.keys())
    print(f"Average fidelity: {average_fidelity:.4f}")


    return average_fidelity, transpiled.depth(), total_time



if __name__ == "__main__":
    pass