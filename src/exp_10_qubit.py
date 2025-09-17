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



def generate_simples_example_for_test_4():
    # P1: 2 data, 1 syn  (total 3)
    vdata1 = virtualSpace(size=2, label="vdata1")
    vdata1.allocate_range(0, 1)
    vsyn1 = virtualSpace(size=1, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 0)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)], size=2, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=1, processID=1))
    # Entangle data[0] and data[1] sequentially with the same syndrome
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(1)])
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)], size=2, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=1, processID=1))

    # P2: 1 data, 1 syn (total 2), offset start_time to exercise scheduler
    vdata2 = virtualSpace(size=1, label="vdata2")
    vdata2.allocate_range(0, 0)
    vsyn2 = virtualSpace(size=1, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 0)
    proc2 = process(processID=2, start_time=5, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)], size=1, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)], size=1, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))

    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
                [5, 6], [6, 7], [7, 8], [8, 9]]

    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001, edge_list=COUPLING)
    return kernel_instance, virtual_hardware


def generate_simples_example_for_test_5():
    # P1: 3 data, 1 syn (total 4)
    vdata1 = virtualSpace(size=3, label="vdata1")
    vdata1.allocate_range(0, 2)
    vsyn1 = virtualSpace(size=1, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 0)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)], size=3, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=1, processID=1))
    # Parity of three data qubits onto one syndrome, with some single-qubit dressing
    proc1.add_instruction(Instype.H, [vdata1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(2), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(2)])
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)], size=3, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=1, processID=1))

    # P2: 2 data, 2 syn (total 4) — two separate checks
    vdata2 = virtualSpace(size=2, label="vdata2")
    vdata2.allocate_range(0, 1)
    vsyn2 = virtualSpace(size=2, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 1)
    proc2 = process(processID=2, start_time=2, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=2, processID=2))
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(1)])
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=2, processID=2))

    # P3: 1 data, 1 syn (total 2)
    vdata3 = virtualSpace(size=1, label="vdata3")
    vdata3.allocate_range(0, 0)
    vsyn3 = virtualSpace(size=1, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0, 0)
    proc3 = process(processID=3, start_time=4, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0)], size=1, processID=3))
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0)], size=1, processID=3))
    proc3.add_instruction(Instype.H, [vdata3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0), vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vdata3.get_address(0)])
    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0)], size=1, processID=3))
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0)], size=1, processID=3))

    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
                [5, 6], [6, 7], [7, 8], [8, 9]]

    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001, edge_list=COUPLING)
    return kernel_instance, virtual_hardware


def generate_simples_example_for_test_6():
    # P1: 1 data, 2 syn (total 3) — same data checked twice (serial checks)
    vdata1 = virtualSpace(size=1, label="vdata1")
    vdata1.allocate_range(0, 0)
    vsyn1 = virtualSpace(size=2, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 1)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)], size=1, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=2, processID=1))
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)], size=1, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=2, processID=1))

    # P2: 2 data, 1 syn (total 3) — interleaved ops and measurements
    vdata2 = virtualSpace(size=2, label="vdata2")
    vdata2.allocate_range(0, 1)
    vsyn2 = virtualSpace(size=1, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 0)
    proc2 = process(processID=2, start_time=1, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))
    proc2.add_instruction(Instype.H, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])
    proc2.add_instruction(Instype.X, [vdata2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(1)])
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))

    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
                [5, 6], [6, 7], [7, 8], [8, 9]]

    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001, edge_list=COUPLING)
    return kernel_instance, virtual_hardware




def generate_simples_example_for_test_7():
    """
    Deep parity-sweep: P1 has 3 data + 1 syndrome, P2 has 2 data + 1 syndrome.
    Runs many CNOT rounds to stress depth; varies start times to tickle scheduler.
    """
    # ---------- Process 1: 3 data, 1 syndrome (total 4 qubits) ----------
    vdata1 = virtualSpace(size=3, label="vdata1")
    vdata1.allocate_range(0, 2)
    vsyn1 = virtualSpace(size=1, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 0)
    p1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    p1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)], size=3, processID=1))
    p1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=1, processID=1))

    # Prepare a little variety
    p1.add_instruction(Instype.H, [vdata1.get_address(0)])
    p1.add_instruction(Instype.X, [vdata1.get_address(1)])
    p1.add_instruction(Instype.Z, [vdata1.get_address(2)])

    # 10 rounds of parity-sweep onto the same syndrome (30 CNOTs total)
    R1 = 10
    for r in range(R1):
        # Optional toggling to create hook/echo effects
        if r % 2 == 0:
            p1.add_instruction(Instype.H, [vdata1.get_address(1)])
        # Sweep all data -> syndrome
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(0)])
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(2), vsyn1.get_address(0)])
        # Measure syndrome each round (like repeated stabilizer readout)
        p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])

    # Read out data at the end
    p1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])
    p1.add_instruction(Instype.MEASURE, [vdata1.get_address(1)])
    p1.add_instruction(Instype.MEASURE, [vdata1.get_address(2)])

    p1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)], size=3, processID=1))
    p1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=1, processID=1))

    # ---------- Process 2: 2 data, 1 syndrome (total 3 qubits) ----------
    vdata2 = virtualSpace(size=2, label="vdata2")
    vdata2.allocate_range(0, 1)
    vsyn2 = virtualSpace(size=1, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 0)
    p2 = process(processID=2, start_time=3, vdataspace=vdata2, vsyndromespace=vsyn2)
    p2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    p2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))

    # 20 CNOTs total: alternate which data controls the syndrome each step
    R2 = 20
    for r in range(R2):
        src = vdata2.get_address(r % 2)
        p2.add_instruction(Instype.CNOT, [src, vsyn2.get_address(0)])
        if r % 4 == 0:
            p2.add_instruction(Instype.X, [src])  # inject toggles to diversify Pauli frames
        if (r + 1) % 5 == 0:
            p2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])

    p2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])
    p2.add_instruction(Instype.MEASURE, [vdata2.get_address(1)])
    p2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])

    p2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    p2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))

    # ---------- Hardware & kernel ----------
    COUPLING = [[0,1],[1,2],[2,3],[3,4],[0,5],[1,6],[2,7],[3,8],[4,9],[5,6],[6,7],[7,8],[8,9]]
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000,'max_physical_qubits': 10000,'max_syndrome_qubits': 1000})
    kernel_instance.add_process(p1)
    kernel_instance.add_process(p2)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001, edge_list=COUPLING)
    return kernel_instance, virtual_hardware


def generate_simples_example_for_test_8():
    """
    Two-syndrome interleaving and long chains:
    - P1: 2 data + 2 syndrome, alternating checks to both syndromes over many rounds.
    - P2: 3 data + 1 syndrome with long repeated sweeps.
    """
    # ---------- Process 1: 2 data, 2 syndrome (total 4) ----------
    vdata1 = virtualSpace(size=2, label="vdata1")
    vdata1.allocate_range(0, 1)
    vsyn1 = virtualSpace(size=2, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 1)
    p1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    p1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)], size=2, processID=1))
    p1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=2, processID=1))

    # Interleave checks onto syn0 and syn1 for depth; ~40 CNOTs
    R = 10
    for r in range(R):
        # Round r: check parity onto syn0
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(0)])
        # Round r: then onto syn1 (like X- and Z-type checks on separate ancillas)
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(1)])
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])
        if r % 3 == 2:
            p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
            p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])
        if r % 2 == 1:
            p1.add_instruction(Instype.H, [vdata1.get_address(0)])

    p1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])
    p1.add_instruction(Instype.MEASURE, [vdata1.get_address(1)])
    p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])

    p1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)], size=2, processID=1))
    p1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=2, processID=1))

    # ---------- Process 2: 3 data, 1 syndrome (total 4) ----------
    vdata2 = virtualSpace(size=3, label="vdata2")
    vdata2.allocate_range(0, 2)
    vsyn2 = virtualSpace(size=1, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 0)
    p2 = process(processID=2, start_time=4, vdataspace=vdata2, vsyndromespace=vsyn2)
    p2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)], size=3, processID=2))
    p2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))

    # Long repeated sweep: 15 rounds (45 CNOTs) with occasional mid-circuit syndrome reads
    R2 = 15
    for r in range(R2):
        p2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])
        p2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(0)])
        p2.add_instruction(Instype.CNOT, [vdata2.get_address(2), vsyn2.get_address(0)])
        if r % 5 == 4:
            p2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])
        if r % 3 == 1:
            p2.add_instruction(Instype.X, [vdata2.get_address((r // 3) % 3)])

    p2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])
    p2.add_instruction(Instype.MEASURE, [vdata2.get_address(1)])
    p2.add_instruction(Instype.MEASURE, [vdata2.get_address(2)])
    p2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])

    p2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)], size=3, processID=2))
    p2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=1, processID=2))

    # ---------- Hardware & kernel ----------
    COUPLING = [[0,1],[1,2],[2,3],[3,4],[0,5],[1,6],[2,7],[3,8],[4,9],[5,6],[6,7],[7,8],[8,9]]
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000,'max_physical_qubits': 10000,'max_syndrome_qubits': 1000})
    kernel_instance.add_process(p1)
    kernel_instance.add_process(p2)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001, edge_list=COUPLING)
    return kernel_instance, virtual_hardware


def generate_simples_example_for_test_9():
    """
    Three processes, each <5 qubits, lots of CNOTs with different patterns:
    - P1: 1 data + 3 syndromes (fanout across different ancillas, many repeats).
    - P2: 2 data + 2 syndromes (checkerboard toggling).
    - P3: 3 data + 1 syndrome (dense long chain).
    """
    # ---------- Process 1: 1 data, 3 syndromes (total 4) ----------
    vdata1 = virtualSpace(size=1, label="vdata1")
    vdata1.allocate_range(0, 0)
    vsyn1 = virtualSpace(size=3, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 2)
    p1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    p1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)], size=1, processID=1))
    p1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=3, processID=1))

    # Fanout the same data onto 3 different syndromes repeatedly
    R = 12  # 36 CNOTs total
    for r in range(R):
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(1)])
        p1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(2)])
        if r % 4 == 3:
            p1.add_instruction(Instype.MEASURE, [vsyn1.get_address((r // 4) % 3)])
        if r % 3 == 0:
            p1.add_instruction(Instype.H, [vdata1.get_address(0)])

    p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])
    p1.add_instruction(Instype.MEASURE, [vsyn1.get_address(2)])
    p1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])

    p1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)], size=1, processID=1))
    p1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)], size=3, processID=1))

    # ---------- Process 2: 2 data, 2 syndromes (total 4) ----------
    vdata2 = virtualSpace(size=2, label="vdata2")
    vdata2.allocate_range(0, 1)
    vsyn2 = virtualSpace(size=2, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 1)
    p2 = process(processID=2, start_time=2, vdataspace=vdata2, vsyndromespace=vsyn2)
    p2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    p2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=2, processID=2))

    # Checkerboard toggling across two ancillas: ~48 CNOTs
    R2 = 12
    for r in range(R2):
        # onto syn0
        p2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])
        p2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(0)])
        # onto syn1
        p2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(1)])
        p2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])
        if r % 2 == 0:
            p2.add_instruction(Instype.X, [vdata2.get_address((r // 2) % 2)])
        if r % 3 == 2:
            p2.add_instruction(Instype.MEASURE, [vsyn2.get_address((r // 3) % 2)])

    p2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])
    p2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])
    p2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])
    p2.add_instruction(Instype.MEASURE, [vdata2.get_address(1)])

    p2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)], size=2, processID=2))
    p2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)], size=2, processID=2))

    # ---------- Process 3: 3 data, 1 syndrome (total 4) ----------
    vdata3 = virtualSpace(size=3, label="vdata3")
    vdata3.allocate_range(0, 2)
    vsyn3 = virtualSpace(size=1, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0, 0)
    p3 = process(processID=3, start_time=6, vdataspace=vdata3, vsyndromespace=vsyn3)
    p3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0)], size=3, processID=3))
    p3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0)], size=1, processID=3))

    # Dense long chain with periodic echo (H) and readout: ~60 CNOTs
    R3 = 20
    for r in range(R3):
        p3.add_instruction(Instype.CNOT, [vdata3.get_address(0), vsyn3.get_address(0)])
        p3.add_instruction(Instype.CNOT, [vdata3.get_address(1), vsyn3.get_address(0)])
        p3.add_instruction(Instype.CNOT, [vdata3.get_address(2), vsyn3.get_address(0)])
        if r % 4 == 1:
            p3.add_instruction(Instype.H, [vdata3.get_address((r // 4) % 3)])
        if r % 5 == 4:
            p3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])

    p3.add_instruction(Instype.MEASURE, [vdata3.get_address(0)])
    p3.add_instruction(Instype.MEASURE, [vdata3.get_address(1)])
    p3.add_instruction(Instype.MEASURE, [vdata3.get_address(2)])
    p3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])

    p3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0)], size=3, processID=3))
    p3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0)], size=1, processID=3))

    # ---------- Hardware & kernel ----------
    COUPLING = [[0,1],[1,2],[2,3],[3,4],[0,5],[1,6],[2,7],[3,8],[4,9],[5,6],[6,7],[7,8],[8,9]]
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000,'max_physical_qubits': 10000,'max_syndrome_qubits': 1000})
    kernel_instance.add_process(p1)
    kernel_instance.add_process(p2)
    kernel_instance.add_process(p3)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001, edge_list=COUPLING)
    return kernel_instance, virtual_hardware



def test_scheduling(test_func, baseline=False, consider_connectivity=True):


    kernel_instance, virtual_hardware = test_func()
    #kernel_instance, virtual_hardware = generate_example_ppt10_on_10_qubit_device()

    schedule_instance = Scheduler(kernel_instance=kernel_instance, hardware_instance=virtual_hardware)


    dis=schedule_instance.calculate_all_pair_distance()


    if baseline:
        time1, inst_list1=schedule_instance.baseline_scheduling()
    else:
        if consider_connectivity:
            time1, inst_list1=schedule_instance.dynamic_scheduling()
        else:
            time1, inst_list1=schedule_instance.dynamic_scheduling_no_consider_connectivity()


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

   

    sim = AerSimulator(noise_model=build_noise_model(error_rate_1q=0.01, error_rate_2q=0.05, p_reset=0.02, p_meas=0.02))
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


    ideal_result=schedule_instance.return_process_ideal_output(shots=2000)
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


    return average_fidelity, transpiled.depth(), running_time



if __name__ == "__main__":

    test_list=[]
    test_list.append(generate_simples_example_for_test_1)
    test_list.append(generate_simples_example_for_test_2)
    test_list.append(generate_simples_example_for_test_3)
    test_list.append(generate_simples_example_for_test_4)
    test_list.append(generate_simples_example_for_test_5)
    test_list.append(generate_simples_example_for_test_6)
    test_list.append(generate_simples_example_for_test_7)
    test_list.append(generate_simples_example_for_test_8)
    test_list.append(generate_simples_example_for_test_9)

    baselineFidelity=[]
    baselineDepth=[]
    baselineTime=[]


    no_connectivity_Fidelity=[]
    no_connectivity_Depth=[]
    no_connectivity_Time=[]   


    ourFidelity=[]
    ourDepth=[]
    ourTime=[]

    for test in test_list:
        print("======== Baseline scheduling ========")
        fidelity, depth, time=test_scheduling(test, baseline=True)
        baselineFidelity.append(fidelity)
        baselineDepth.append(depth)
        baselineTime.append(time)

        print("======== Our scheduling (consider connectivity) ========")
        fidelity, depth, time=test_scheduling(test, baseline=False, consider_connectivity=True)
        ourFidelity.append(fidelity)
        ourDepth.append(depth)
        ourTime.append(time)


        print("======== Our scheduling (not consider connectivity) ========")
        fidelity, depth, time=test_scheduling(test, baseline=False, consider_connectivity=False)
        no_connectivity_Fidelity.append(fidelity)
        no_connectivity_Depth.append(depth)
        no_connectivity_Time.append(time)



    """
    Print the result of fidelity in bar plot
    Compare three algorithms: baseline, our algorithm considering connectivity, our algorithm not considering connectivity
    Store the plot in resultFidelity.png
    """
    labels = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Test 6','Test 7', 'Test 8', 'Test 9']
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, baselineFidelity, width, label='Baseline', color='red')
    rects2 = ax.bar(x, ourFidelity, width, label='Our Algorithm (consider connectivity)', color='blue')
    rects3 = ax.bar(x + width, no_connectivity_Fidelity, width, label='Our Algorithm (not consider connectivity)', color='green')
    ax.set_ylabel('Fidelity')
    ax.set_title('Fidelity by different scheduling algorithms')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    fig.tight_layout()
    plt.ylim(0, 1.1)
    plt.savefig("result_fidelity.png")
    plt.close(fig)


    """
    Print the result of width in bar plot
    Compare three algorithms: baseline, our algorithm considering connectivity, our algorithm not considering connectivity
    Store the plot in resultWidth.png
    """
    labels = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Test 6','Test 7', 'Test 8', 'Test 9']
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, baselineDepth, width, label='Baseline', color='red')
    rects2 = ax.bar(x, ourDepth, width, label='Our Algorithm (consider connectivity)', color='blue')
    rects3 = ax.bar(x + width, no_connectivity_Depth, width, label='Our Algorithm (not consider connectivity)', color='green')
    ax.set_ylabel('Circuit depth')
    ax.set_title('Circuit depth by different scheduling algorithms')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    fig.tight_layout()
    plt.ylim(0, max(baselineDepth + ourDepth + no_connectivity_Depth)*1.1)
    plt.savefig("result_depth.png")
    plt.close(fig)



    """
    Print the result of time in bar plot
    Compare three algorithms: baseline, our algorithm considering connectivity, our algorithm not considering connectivity
    Store the plot in resultTime.png
    """
    labels = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Test 6','Test 7', 'Test 8', 'Test 9']
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, baselineTime, width, label='Baseline', color='red')
    rects2 = ax.bar(x, ourTime, width, label='Our Algorithm (consider connectivity)', color='blue')
    rects3 = ax.bar(x + width, no_connectivity_Time, width, label='Our Algorithm (not consider connectivity)', color='green')
    ax.set_ylabel('Circuit depth')
    ax.set_title('Circuit depth by different scheduling algorithms')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    fig.tight_layout()
    plt.ylim(0, max(baselineTime + ourTime + no_connectivity_Time)*1.1)
    plt.savefig("result_time.png")
    plt.close(fig)
