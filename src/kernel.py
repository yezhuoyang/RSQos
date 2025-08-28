#A tiny kernel to execute fault-tolerant quantum programs
#from tqec import BlockGraph, compile_block_graph, NoiseModel
from process import process
from instruction import *
from syscall import *
from virtualSpace import virtualSpace, virtualAddress


class Kernel:


    def __init__(self, config):
        self.config = config
        self._max_virtual_logical_qubits = config.get('max_virtual_logical_qubits', 1000)
        self._max_physical_qubits = config.get('max_physical_qubits', 10000)
        self._max_syndrome_qubits = config.get('max_syndrome_qubits', 1000)
        self._processes = []

    def add_process(self, process_instance: process):
        """
        Add a process to the kernel.
        """
        if isinstance(process_instance, process):
            self._processes.append(process_instance)
        else:
            raise TypeError("Expected a process instance.")

    def get_processes(self):
        return self._processes

    def execute(self, circuit):
        # Execute the given quantum circuit
        pass


    def get_time_space_volume(self):
        # Calculate and return the time-space volume of the executed circuit
        pass


    def __str__(self):
        outputstr = f"Kernel\n"
        outputstr += "Processes:\n"
        for proc in self._processes:
            outputstr += str(proc) + "\n"
        return outputstr


if __name__ == "__main__":
    process_instance1 = process(processID=1)
    vdata = virtualSpace(size=10, label="vdata")
    vsyn = virtualSpace(size=5, label="vsyn")

    process_instance1.add_syscall(syscallinst=syscall_allocate_data_qubits(size=2,processID=1))  # Allocate 2 data qubits
    process_instance1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=2,processID=1))  # Allocate 2 syndrome qubits


    process_instance1.add_instruction(Instype.CNOT, [vdata.get_address(0), vsyn.get_address(0)])  # CNOT operation
    process_instance1.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(1)])  # CNOT operation
    process_instance1.add_instruction(Instype.MEASURE, [vsyn.get_address(0)])  # Measure operation
    process_instance1.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(2)])  # CNOT operation

    print(process_instance1)


    process_instance2 = process(processID=2)
    vdata = virtualSpace(size=10, label="vdata")
    vsyn = virtualSpace(size=5, label="vsyn")

    process_instance2.add_syscall(syscallinst=syscall_allocate_data_qubits(size=2,processID=2))  # Allocate 2 data qubits
    process_instance2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=2,processID=2))  # Allocate 2 syndrome qubits


    process_instance2.add_instruction(Instype.CNOT, [vdata.get_address(0), vsyn.get_address(0)])  # CNOT operation
    process_instance2.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(1)])  # CNOT operation
    process_instance2.add_instruction(Instype.MEASURE, [vsyn.get_address(0)])  # Measure operation
    process_instance2.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(2)])  # CNOT operation

    print(process_instance2)


    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(process_instance1)
    kernel_instance.add_process(process_instance2)


    print(kernel_instance)
