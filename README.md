# FTQos

A tiny operating system for fault-tolerant quantum computer based on lattice surgery. 
We create some new concepts such as virtual logical qubit, virtual T factory, etc., for furture design of the OS.
Also, we support a tiny kernel for resource managagement of fault-tolerant quantum computation.


The first implementation will be based on TQEC software developed by Google https://tqec.github.io/tqec/
All virtual quantum processes can be compiled to STIM program of qiskit program. 



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
- [X] Construct a smaller fake backend with 10 qubits
- [ ] Separate readout information of multiple processes
- [X] For each process, get the ideal output distribution
- [ ] Test and verify the correctness of process output
- [ ] Analyze and add a cost function for routing
- [ ] Routing aware scheduling algorithm
- [ ] Understand the input/output format of IBM cloud by a free small backend
