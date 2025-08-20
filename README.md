# FTQos
A tiny operating system for fault-tolerant quantum computer based on lattice surgery. 
We create some new concepts such as virtual logical qubit, virtual T factory, etc., for furture design of the OS.
Also, we support a tiny kernel for resource managagement of fault-tolerant quantum computation.

The first implementation is based on TQEC software developed by Google https://tqec.github.io/tqec/



# High-level design

On the highlest level, our kenel deal with logical process that users want to execute on a fault-tolerant quantum hardware.
The kernel manage all these process, optimize the resource usage.


- [*] Logical quantum process
- [*] Kernel for logical quantum computer



# TODO List




- [ ] Add a function to print the scheduling log.
- [ ] Change Process Status while scheduling.
- [ ] For T-factory syscall, add space requirement.
- [ ] Consider T-factory scheduling in the baseline algorithm.
- [ ] Consider T-factory scheduling in the round robin algorithm. 
