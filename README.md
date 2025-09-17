# FTQos

A tiny operating system for fault-tolerant quantum computer based on lattice surgery. 
We create some new concepts such as virtual logical qubit, virtual T factory, etc., for furture design of the OS.
Also, we support a tiny kernel for resource managagement of fault-tolerant quantum computation.


The first implementation will be based on TQEC software developed by Google https://tqec.github.io/tqec/
All virtual quantum processes can be compiled to STIM program of qiskit program. 




# Initialize a FT quantum process

In FTQOS, the quantum circuit is programmed on a virtual machine. 
There are two different virtual space, virtual data space and virtual syndrome space. 
The process use syscall to initialize/deallocate virtual space.
Following is an example of initialize a process object with a quantum circuit.


```python

vdata1 = virtualSpace(size=3, label="vdata1")
vdata1.allocate_range(0,2)
vsyn1 = virtualSpace(size=2, label="vsyn1", is_syndrome=True)
vsyn1.allocate_range(0,1)

proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1) # Initialize quantum process

# System call to create virtual space
proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))
proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))

# Define circuit on virtual machine
proc1.add_instruction(Instype.H, [vsyn1.get_address(0)])
proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0),vsyn1.get_address(1)])
proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])
proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0), vsyn1.get_address(1)])
proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])

# System call to deallocate virtual space
proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))
proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))

```

You can see the ideal(Noiseless) result of running the process by:


```python
proc1.construct_qiskit_circuit()
shots=2000
#Return a dictionary of bitstring. 
counts = process_instance.simulate_circuit(shots=shots)
print(counts)
```


# Initialize a QKernel with multiple quantum processes

QKernel handle the task of scheduling different quantum processes, map virtual qubits to real hardware qubit, collect and distribution the measurement results.
For example, you can initialize a quantum kernel with several processes by:


```python
#Define proc1,proc2,proc3 in the previous code
kernel_instance = Kernel(config={
    'max_virtual_logical_qubits': 10_000,
    'max_physical_qubits': 100_000,
    'max_syndrome_qubits': 10_000
})
kernel_instance.add_process(proc1)
kernel_instance.add_process(proc2)
kernel_instance.add_process(proc3)
```



# Construct quantum hardware with noise model

We use GenericBackendV2 from qiskit to construct fake hardware with layout constraint. 


```python

def construct_10_qubit_hardware():
    NUM_QUBITS = 10
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain
    BASIS = ["cx", "id", "rz", "sx", "x"]  # add more *only* if truly native
    SINGLE_QUBIT_GATE_LENGTH_NS = 32       # example: 0.222 ns timestep
    SINGLE_QUBIT_GATE_LENGTH_NS = 88       # example: 0.222 ns timestep
    READOUT_LENGTH_NS = 2584     # example measurement timestep

    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,         # optional
        coupling_map=COUPLING,     # strongly recommended
        control_flow=True,        # set True if you want dynamic circuits            
        seed=1234,                 # reproducible auto-generated props
        noise_info=True            # attach plausible noise/durations
    )

    return backend    
```

You can also specify the noise model by:


```python
def build_noise_model(error_rate_1q=0.001, error_rate_2q=0.01, p_reset=0.001, p_meas=0.01):
    custom_noise_model = NoiseModel()
    
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])

    custom_noise_model.add_all_qubit_quantum_error(error_reset,"reset")

    custom_noise_model.add_all_qubit_quantum_error(error_meas,"measure")
    custom_noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate_1q, 1), ['id','rx','rz','sx','x'])

    # Add a depolarizing error to two-qubit gates on specific qubits
    custom_noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate_2q, 2), ['cz','rzz'])


    return custom_noise_model
```



# How to use

To initialize and run the scheduling of many quantum processes, see the following example:


```python

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

# Define a function that return a kernel_instance and a virtual_hardware
def generate_example():
    return kernel_instance,virtual_hardware

# ========================  MAIN  ========================
if __name__ == "__main__":




    kernel_instance, virtual_hardware =generate_example()
    schedule_instance=Scheduler(kernel_instance,virtual_hardware)
    time1, inst_list1=schedule_instance.dynamic_scheduling()
    schedule_instance.print_dynamic_instruction_list(inst_list1)
    qc=schedule_instance.construct_qiskit_circuit_for_backend(inst_list1)


    # fig_t = qc.draw(output="mpl", fold=-1)
    # fig_t.savefig("before_transpiled.png", dpi=200, bbox_inches="tight")
    # plt.close(fig_t)

    qc.draw("mpl", fold=-1).show()
    print(qc.num_qubits)

    # 0) Fake 156-qubit backend (your Pittsburgh layout)
    fake_ibm_pittsburgh = construct_fake_ibm_pittsburgh()
    print(f"[backend] num_qubits = {fake_ibm_pittsburgh.num_qubits}")

    # 1) Build the abstract (logical) circuit and save as PNG
    # qc = build_dynamic_circuit_15()
    # save_circuit_png(qc, "abstract_circuit.png")  # uses Matplotlib

    # 2) Transpile to hardware; map 15 logical qubits onto a single long row
    #    (contiguous physical qubits minimize SWAPs on your lattice)
    initial_layout = [i for i in range(156)]  # logical i -> physical i




    transpiled = transpile(
        qc,
        backend=fake_ibm_pittsburgh,
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
    plot_process_schedule_on_pittsburgh(
        coupling_edges=fake_ibm_pittsburgh.coupling_map,
        syndrome_qubit_history=syndrome_history,
        process_list=process_list,
        out_png="hardware_processes.png",
    )

    # 4) Run on the fake backend (Aer noise if installed; otherwise ideal) and print counts
    job = fake_ibm_pittsburgh.run(transpiled, shots=2000)
    result = job.result()
    counts = result.get_counts()
    print("\n=== Counts ===")
    print(counts)

```








# High-level design

On the highlest level, our kenel deal with logical process that users want to execute on a fault-tolerant quantum hardware.
The kernel manage all these process, optimize the resource usage.


- [x] Logical quantum process
- [x] Kernel for logical quantum computer



# TODO List




- [x] Add a function to print the scheduling log.
- [x] Change Process Status while scheduling.
- [x] For T-factory syscall, add space requirement.
- [x] Consider T-factory scheduling in the baseline algorithm.
- [x] Consider T-factory scheduling in the round robin algorithm. 
- [x] Construct a smaller fake backend with 10 qubits
- [x] Visualize the scheduling on the backend with 10 qubits
- [x] Separate readout information of multiple processes
- [x] For each process, get the ideal output distribution
- [x] Get the ideal output of the transpiled circuit, to test the correctness of scheduling
- [x] Test and verify the correctness of process output
- [x] Complete and test baseline scheduling(Alg0)
- [x] Complete and test scheduling that doesn't share ansilla qubit(Alg1)
- [ ] Get more accurate estimation on hardware running time
- [ ] Figure out if parallel control is possible in IBM quantum
- [x] Analyze and add a cost function for routing
- [x] Complete and test Routing aware scheduling algorithm(Alg3)
- [x] Create small baseline(With 10 qubits)
- [ ] Understand the input/output format of IBM cloud by a free small backend
- [ ] Scheduling algorithm with fixed syndrome qubit area
- [ ] Now a reset is automatically inserted after measurement, optimize it in the future development
- [ ] Construct process from STIM file
- [ ] Generate STIM circuit for evalution