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



def generate_simple_example():
    vdata1 = virtualSpace(size=2, label="vdata1")    
    vdata1.allocate_range(0,1)
    vsyn1 = virtualSpace(size=3, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,2)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1, shots =1000)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)],size=2,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 1 syndrome qubit
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1),vsyn1.get_address(1)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1),vsyn1.get_address(2)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(2)])  # Measure operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)],size=2 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 2 syndrome qubits



    vdata2 = virtualSpace(size=3, label="vdata2")    
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=3, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,2)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2, shots=1000)

    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 1 syndrome qubit
    proc2.add_instruction(Instype.H, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.H, [vdata2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2),vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])  # Measure operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])  # Measure operation
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 2 syndrome qubits



    vdata3 = virtualSpace(size=3, label="vdata3")    
    vdata3.allocate_range(0,2)
    vsyn3 = virtualSpace(size=2, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0,1)
    proc3 = process(processID=1, start_time=0, vdataspace=vdata3, vsyndromespace=vsyn3, shots =1000)
    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0)],size=3,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=2,processID=3))  # Allocate 1 syndrome qubit
    proc3.add_instruction(Instype.X, [vdata3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0),vsyn3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(1),vsyn3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(1),vsyn3.get_address(1)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(2),vsyn3.get_address(1)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])  # Measure operation
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(1)])  # Measure operation
    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0)],size=3 ,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=2,processID=3))  # Allocate 2 syndrome qubits



    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware




def generate_simple_example0():
    vdata1 = virtualSpace(size=3, label="vdata1")    
    vdata1.allocate_range(0,1)
    vsyn1 = virtualSpace(size=3, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,2)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1, shots =1000)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 1 syndrome qubit
    
    proc1.add_instruction(Instype.H, [vdata1.get_address(2)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1),vsyn1.get_address(2)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1),vsyn1.get_address(1)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1),vsyn1.get_address(2)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(2)])  # Measure operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)],size=2 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 2 syndrome qubits



    vdata2 = virtualSpace(size=3, label="vdata2")    
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=3, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,2)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2, shots=1000)

    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 1 syndrome qubit
    proc2.add_instruction(Instype.H, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.H, [vdata2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2),vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])  # Measure operation
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


if __name__ == "__main__":
    kernel_instance, virtual_hardware = generate_simple_example0()

    schedule_instance = Scheduler(kernel_instance=kernel_instance, hardware_instance=virtual_hardware)
    dis=schedule_instance.calculate_all_pair_distance()

    time1, inst_list1, shots=schedule_instance.dynamic_scheduling()



    schedule_instance.print_dynamic_instruction_list(inst_list1)
    qc=schedule_instance.construct_qiskit_circuit_for_backend(inst_list1)



    fig_t = qc.draw(output="mpl", fold=-1)
    fig_t.savefig("before_transpiled.pdf", bbox_inches="tight")
    plt.close(fig_t)

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
        out_png="hardware_processes.pdf",
    )
    



