from qiskit import QuantumCircuit, transpile
# visualize_layout.py
from qiskit.visualization import plot_coupling_map
import matplotlib
matplotlib.use("Agg")              # call BEFORE importing pyplot
import matplotlib.pyplot as plt
from qiskit.transpiler import CouplingMap
from scheduler import *
from process import process, parse_qasm_instruction
import numpy as np
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error,pauli_error
from fakeHardware import construct_10_qubit_hardware, construct_20_qubit_hardware, plot_process_schedule_on_20_qubit_hardware, save_circuit_png, plot_process_schedule_on_10_qubit_hardware, build_noise_model,distribution_fidelity
from instruction import *
from qiskit.qasm2 import dumps



label_name_map = {
    1: "adder_n4",
    2: "basis_trotter_n4",
    3: "bb84_n8",
    4: "bell_n4",
    5: "cat_state_n4",
    6: "deutsch_n2",
    7: "dnn_n2",
    8: "dnn_n8",
    9: "error_correctiond3_n5",
    10: "fredkin_n3",
    11: "grover_n2",
    12: "hs4_n4",
    13: "ising_n10",
    14: "iswap_n2",
    15: "lpn_n5",
    16: "qaoa_n3",
    17: "qaoa_n6",
    18: "qec_en_n5",
    19: "qft_n4",
    20: "qrng_n4",
    21: "simon_n6",
    22: "teleportation_n3",
    23: "toffoli_n3",
    24: "vqe_n4",
    25: "wstate_n3"
}



def generate_random_kernel(num_proc,max_shot):
    """
    Randomly generate a kernel with specified number of processes.
    All those processes are selected randomly from the small QASM benchmark set.
    Args:
        num_proc (int): Number of processes in the kernel.
        max_shot (int): Maximum number of shots for each process.
    """

    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], 
                [0,5], [1,6], [2,7], [3,8], [4,9],
                [5,6], [6,7],[7,8],[8,9],
                [5,10], [6, 11], [7, 12], [8, 13], [9, 14],
                [10,11],[11,12], [12,13], [13,14], 
                [10,15], [11,16], [12,17], [13,18], [14,19],
                [15,16], [16,17], [17,18], [18,19]]  


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    name_list = []
    for pid in range(1, num_proc + 1):
        label_id = np.random.randint(1, 26)  # Assuming there are 25 benchmark QASM files
        label_name = label_name_map[label_id]
        name_list.append(label_name)
        file_path = f"C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\smallqasm\\{label_name}.qasm"
        with open(file_path, "r") as file:
            qasm_code = file.read()   
        shots = np.random.randint(100, max_shot + 1)
        proc_instance = parse_qasm_instruction(shots=shots, process_ID=pid, instruction_str=qasm_code)
        kernel_instance.add_process(proc_instance)

    virtual_hardware = virtualHardware(qubit_number=20, error_rate=0.001,edge_list=COUPLING)
    print("Generated processes:", name_list)
    return kernel_instance, virtual_hardware




def test_scheduling(kernel_instance, virtual_hardware, baseline=False, consider_connectivity=True, share_syndrome_qubits=True):


    #kernel_instance, virtual_hardware = generate_example_ppt10_on_10_qubit_device()

    schedule_instance = Scheduler(kernel_instance=kernel_instance, hardware_instance=virtual_hardware)


    dis=schedule_instance.calculate_all_pair_distance()

    while not kernel_instance.processes_all_finished():
        if baseline:
            time1, inst_list1, shots=schedule_instance.baseline_scheduling()
        else:
            if consider_connectivity:
                time1, inst_list1,shots=schedule_instance.dynamic_scheduling()
            else:
                if share_syndrome_qubits:
                    time1, inst_list1=schedule_instance.dynamic_scheduling_no_consider_connectivity()
                else:
                    time1, inst_list1=schedule_instance.scheduling_with_out_sharing_syndrome_qubit()


        schedule_instance.print_dynamic_instruction_list(inst_list1)
        qc=schedule_instance.construct_qiskit_circuit_for_backend(inst_list1)



        # fig_t = qc.draw(output="mpl", fold=-1)
        # fig_t.savefig("before_transpiled.png", dpi=200, bbox_inches="tight")
        # plt.close(fig_t)

        # qc.draw("mpl", fold=-1).show()
        # print(qc.num_qubits)

        # 0) Fake 156-qubit backend (your Pittsburgh layout)
        fake_hard_ware = construct_20_qubit_hardware()


        # 1) Build the abstract (logical) circuit and save as PNG
        # qc = build_dynamic_circuit_15()
        # save_circuit_png(qc, "abstract_circuit.png")  # uses Matplotlib

        # 2) Transpile to hardware; map 15 logical qubits onto a single long row
        #    (contiguous physical qubits minimize SWAPs on your lattice)
        initial_layout = [i for i in range(20)]  # logical i -> physical i



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
        plot_process_schedule_on_20_qubit_hardware(
            coupling_edges= fake_hard_ware.coupling_map,
            syndrome_qubit_history=syndrome_history,
            process_list=process_list,
            out_png="hardware_processes.png",
        )
        



        # 4) Run on the fake backend (Aer noise if installed; otherwise ideal) and print counts

        # job = fake_hard_ware.run(transpiled, shots=shots)
        # result = job.result()
        # running_time=result.time_taken

    

        sim = AerSimulator(noise_model=build_noise_model(error_rate_1q=0.00, error_rate_2q=0.00, p_reset=0.00, p_meas=0.00))
        tqc = transpile(transpiled, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts(tqc)
        # print("\n=== Counts(Fake hardware) ===")
        print(counts)   



        '''
        Get the ideal result
        '''
        # sim = AerSimulator()
        # tqc = transpile(qc, sim)

        # Run with 1000 shots
        # result = sim.run(tqc, shots=2000).result()
        # idcounts = result.get_counts(tqc)
        # print("\n=== Counts(Ideal) ===")
        # print(idcounts)



        # print(schedule_instance._measure_index_to_process)
        # print(schedule_instance._process_measure_index)


        final_result=schedule_instance.return_measure_states(counts)
        #print(final_result)

        #print(ideal_result)

        kernel_instance.update_process_results(final_result)
        kernel_instance.reset_all_processes()
        schedule_instance.reset_all_states()



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




    print("The TRANSPILED circuit depth is:", transpiled .depth())

    # print("\n=== Time taken:===")
    # print(running_time)

    average_fidelity/=len(final_result.keys())
    print(f"Average fidelity: {average_fidelity:.4f}")


    return average_fidelity, transpiled.depth()






if __name__ == "__main__":


    # for id in range(1, 26):
    #     label_id =  id
    #     label_name = label_name_map[label_id]
    #     print(f"Testing QASM file: {label_name}.qasm")
    #     file_path = f"C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\smallqasm\\{label_name}.qasm"
    #     with open(file_path, "r") as file:
    #         qasm_code = file.read()   
    #     shots =  1000
    #     proc_instance = parse_qasm_instruction(shots=shots, process_ID=1, instruction_str=qasm_code)


    kernel_instance, virtual_hardware = generate_random_kernel(num_proc=5, max_shot=1024)

    print(kernel_instance)

    test_scheduling(kernel_instance=kernel_instance, virtual_hardware=virtual_hardware, baseline=True, consider_connectivity=True, share_syndrome_qubits=True)


    # label_name = label_name_map[4]
    # file_path = f"C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\smallqasm\\{label_name}.qasm"
    # with open(file_path, "r") as file:
    #     qasm_code = file.read()  
    #     proc=parse_qasm_instruction(shots=1000, process_ID=1, instruction_str=qasm_code)

    #     print(proc.process_str())