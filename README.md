# FTQos
A tiny operating system for fault-tolerant quantum computer based on lattice surgery. 
We create some new concepts such as virtual logical qubit, virtual T factory, etc., for furture design of the OS.
Also, we support a tiny kernel for resource managagement of fault-tolerant quantum computation.

The first implementation is based on TQEC software developed by Google https://tqec.github.io/tqec/



# High-level design

On the highlest level, our kenel deal with logical process that users want to execute on a fault-tolerant quantum hardware.
The kernel manage all these process, optimize the resource usage.


- [ ] Logical quantum process
- [ ] Kernel for logical quantum computer



# TODO List



In the first step, we implement the demo for a small fault-tolerant quantum computer. 
All our implementation must be evalulated for further optimization.


- [ ] Logical CNOT gate  
- [ ] Logical Hadamard gate  
- [ ] A simple T factory  
- [ ] A small circuit with T gate