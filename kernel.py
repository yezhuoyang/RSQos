#A tiny kernel to execute fault-tolerant quantum programs
from tqec import BlockGraph, compile_block_graph, NoiseModel
from .process import process



class Kernel:


    def __init__(self, config):
        self.config = config
        self._max_virtual_logical_qubits = config.get('max_virtual_logical_qubits', 1000)
        self._max_physical_qubits = config.get('max_physical_qubits', 10000)
        self._max_syndrome_qubits = config.get('max_syndrome_qubits', 1000)
        self._processes = []

    def add_process(self, process):
        """
        Add a process to the kernel.
        """
        if isinstance(process, process):
            self._processes.append(process)
        else:
            raise TypeError("Expected a process instance.")


    def execute(self, circuit):
        # Execute the given quantum circuit
        pass


    def get_time_space_volume(self):
        # Calculate and return the time-space volume of the executed circuit
        pass