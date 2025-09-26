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
from fakeHardware import construct_10_qubit_hardware, save_circuit_png, plot_process_schedule_on_10_qubit_hardware, build_noise_model,distribution_fidelity



def generate_simples_example_for_test_1():
    vdata1 = virtualSpace(size=1, label="vdata1")    
    vdata1.allocate_range(0,0)
    vsyn1 = virtualSpace(size=1, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,0)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)],size=1,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=1,processID=1))  # Allocate 1 syndrome qubit
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation     
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation    
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)],size=1 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=1,processID=1))  # Allocate 2 syndrome qubits



    vdata2 = virtualSpace(size=1, label="vdata2")    
    vdata2.allocate_range(0,0)
    vsyn2 = virtualSpace(size=1, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,0)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)],size=1,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=1,processID=2))  # Allocate 1 syndrome qubit
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation        
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation  
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)],size=1 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=1,processID=2))  # Allocate 2 syndrome qubits


    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware


def generate_simples_example_for_test_2():
    vdata1 = virtualSpace(size=3, label="vdata1")    
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=3, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,2)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 1 syndrome qubit
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(1)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(2)])
    proc1.add_instruction(Instype.X, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation    
    proc1.add_instruction(Instype.X, [vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])  # Measure operation    
    proc1.add_instruction(Instype.X, [vsyn1.get_address(2)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(2)])  # Measure operation    
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation        
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 2 syndrome qubits

    vdata2 = virtualSpace(size=3, label="vdata2")    
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=3, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,2)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 1 syndrome qubit
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.X, [vdata2.get_address(1)])
    proc2.add_instruction(Instype.X, [vdata2.get_address(2)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])  # Measure operation    
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation  
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])  # Measure operation      
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2),vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])  # Measure operation           
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 2 syndrome qubits




    vdata3 = virtualSpace(size=1, label="vdata3")    
    vdata3.allocate_range(0,0)
    vsyn3 = virtualSpace(size=1, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0,0)
    proc3 = process(processID=3, start_time=0, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0)],size=1,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=1,processID=3))  # Allocate 1 syndrome qubit
    proc3.add_instruction(Instype.X, [vdata3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0),vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vdata3.get_address(0)])  # Measure operation    
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])  # Measure operation        
    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0)],size=1 ,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=1,processID=3))  # Allocate 2 syndrome qubits


    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)

    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware








def generate_simples_example_for_test_3():
    vdata1 = virtualSpace(size=3, label="vdata1")    
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=3, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,2)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 1 syndrome qubit
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(1)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(2)])
    proc1.add_instruction(Instype.X, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation    
    proc1.add_instruction(Instype.X, [vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])  # Measure operation    
    proc1.add_instruction(Instype.X, [vsyn1.get_address(2)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(2)])  # Measure operation    
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation        
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 2 syndrome qubits

    vdata2 = virtualSpace(size=3, label="vdata2")    
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=3, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,2)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 1 syndrome qubit
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.X, [vdata2.get_address(1)])
    proc2.add_instruction(Instype.X, [vdata2.get_address(2)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])  # Measure operation    
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation  
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])  # Measure operation      
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2),vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])  # Measure operation           
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 2 syndrome qubits


    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware


def test_scheduling():


    kernel_instance, virtual_hardware = generate_simples_example_for_test_1()
    #kernel_instance, virtual_hardware = generate_example_ppt10_on_10_qubit_device()

    schedule_instance = Scheduler(kernel_instance=kernel_instance, hardware_instance=virtual_hardware)



    dis=schedule_instance.calculate_all_pair_distance()


    time1, inst_list1=schedule_instance.dynamic_scheduling()
    #time1, inst_list1=schedule_instance.baseline_scheduling()
    schedule_instance.print_dynamic_instruction_list(inst_list1)
    qc=schedule_instance.construct_qiskit_circuit_for_backend(inst_list1)



    # fig_t = qc.draw(output="mpl", fold=-1)
    # fig_t.savefig("before_transpiled.png", dpi=200, bbox_inches="tight")
    # plt.close(fig_t)

    # qc.draw("mpl", fold=-1).show()
    # print(qc.num_qubits)

    # 0) Fake 156-qubit backend (your Pittsburgh layout)
    fake_hard_ware = construct_10_qubit_hardware()


    # 1) Build the abstract (logical) circuit and save as PNG
    # qc = build_dynamic_circuit_15()
    # save_circuit_png(qc, "abstract_circuit.png")  # uses Matplotlib

    # 2) Transpile to hardware; map 15 logical qubits onto a single long row
    #    (contiguous physical qubits minimize SWAPs on your lattice)
    initial_layout = [i for i in range(10)]  # logical i -> physical i



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
    plot_process_schedule_on_10_qubit_hardware(
        coupling_edges= fake_hard_ware.coupling_map,
        syndrome_qubit_history=syndrome_history,
        process_list=process_list,
        out_png="hardware_processes.png",
    )
    



    # 4) Run on the fake backend (Aer noise if installed; otherwise ideal) and print counts

    job = fake_hard_ware.run(transpiled, shots=2000)
    result = job.result()
    running_time=result.time_taken

   

    sim = AerSimulator(noise_model=build_noise_model(error_rate_1q=0.005, error_rate_2q=0.05, p_reset=0.001, p_meas=0.02))
    tqc = transpile(transpiled, sim)
    result = sim.run(tqc, shots=2000).result()
    counts = result.get_counts(tqc)
    # print("\n=== Counts(Fake hardware) ===")
    print(counts)   



    '''
    Get the ideal result
    '''
    sim = AerSimulator()
    tqc = transpile(qc, sim)

    # Run with 1000 shots
    result = sim.run(tqc, shots=2000).result()
    idcounts = result.get_counts(tqc)
    print("\n=== Counts(Ideal) ===")
    print(idcounts)



    # print(schedule_instance._measure_index_to_process)
    # print(schedule_instance._process_measure_index)


    final_result=schedule_instance.return_measure_states(counts)
    #print(final_result)


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

    print("\n=== Time taken:===")
    print(running_time)

    average_fidelity/=len(final_result.keys())
    print(f"Average fidelity: {average_fidelity:.4f}")



if __name__ == "__main__":
    test_scheduling()