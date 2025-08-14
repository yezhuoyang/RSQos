from kernel import *
from process import *
from instruction import *
from hardware import *
from instruction import *
from syscall import *



class Scheduler:
    def __init__(self, kernel_instance: Kernel, hardware_instance: virtualHardware):
        self._kernel = kernel_instance
        self._hardware = hardware_instance

        self._qubit_num = hardware_instance.get_qubit_num()
        self._current_avalible = { i: True for i in range(1, self._qubit_num + 1) }
        self._mapped_to_syndrome = { i: False for i in range(1, self._qubit_num + 1) }
        self._used_counter = { i: 0 for i in range(1, self._qubit_num + 1) }   # Count of how many process used each qubit. Syndrome qubit might be reused
        self._num_availble = self._qubit_num
        self._virtual_hardware_mapping = virtualHardwareMapping(hardware_instance)
        self._current_process_in_execution = []


    def get_virtual_hardware_mapping(self):
        """
        Get the virtual hardware mapping.
        """
        for i in range(start, self._qubit_num + 1):
            if self._current_avalible[i]:
                return i
        return None

    def free_resources(self, process_instance: process):
        """
        Free resources allocated to a process.
        Free data qubits and syndrome qubits separately
        """
        for addr in process_instance.get_virtual_data_addresses():
            hardware_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
            self._used_counter[hardware_qid] = 0
            self._current_avalible[hardware_qid] = True
            self._num_availble += 1

        for addr in process_instance.get_virtual_syndrome_addresses():
            hardware_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
            self._used_counter[hardware_qid] -= 1
            if self._used_counter[hardware_qid] == 0:
                self._current_avalible[hardware_qid] = True
                self._num_availble += 1
                self._mapped_to_syndrome[hardware_qid] = False

    def have_enough_resources(self, process_instance: process) -> bool:
        """
        Check if there are enough resources available for a process.
        Return True if num_available >= num_required
        """
        if self._num_availble < process_instance.get_num_data_qubits() + 1:
            return False
        return True


    def next_free_index(self, start: int) -> int:
        """
        Find the next free index starting from 'start'.
        """
        for i in range(start, self._qubit_num + 1):
            if self._current_avalible[i]:
                return i
        return None


    def least_busy_syndrome_qubit(self) -> int:
        """
        Find the least busy syndrome qubit, which has the minimum usage count.
        """
        min_usage = float('inf')
        min_index = None
        for i in range(1, self._qubit_num + 1):
            if self._mapped_to_syndrome[i] and self._used_counter[i] < min_usage:
                min_usage = self._used_counter[i]
                min_index = i
        return min_index if min_index is not None else -1

    def allocate_resources(self, process_instance: process):
        """
        Allocate resources for a process.
        Now assume that the number of data qubits required is less than or equal to the number of available physical qubits.
        Idea: Each data qubit must be mapped to different physical qubits.
              However, syndrome qubits can be reused.
        """
        next_free_index_=self.next_free_index(0)
        for addr in process_instance.get_virtual_data_addresses():
            """
            Find the next available index
            Case 1: The address is a data qubit, find the first availble qubit
            """
            self._current_avalible[next_free_index_] = False
            self._used_counter[next_free_index_] = 1
            self._num_availble -= 1
            process_instance.map_virtual_to_physical(addr, next_free_index_)
            next_free_index_=self.next_free_index(next_free_index_+1)

        next_free_index_=self.next_free_index(0)
        for addr in process_instance.get_virtual_syndrome_addresses():
            """
            Greedy algorithm: Use available qubit until none are available
            Otherwise, use the syndrome qubit with the lowest usage count.
            """
            if num_available > 0:
                self._current_avalible[next_free_index_] = False
                self._used_counter[next_free_index_] = 1
                self._num_availble -= 1
                self._mapped_to_syndrome[next_free_index_] = True
                process_instance.map_virtual_to_physical(addr, next_free_index_)
                next_free_index_ = self.next_free_index(next_free_index_ + 1)
            else:
                least_busy_qubit = self.least_busy_syndrome_qubit()
                if least_busy_qubit != -1:
                    self._used_counter[least_busy_qubit] += 1
                    process_instance.map_virtual_to_physical(addr, least_busy_qubit)
                else:
                    assert False, "No available qubit found for syndrome mapping."


    def schedule(self):
        """
        Schedule processes for execution.
        The input is a kernel with a process queue.
        The output is a single process instance with virtual hardware mapping.
        """
        processes_stack = self._kernel.get_processes().copy()

        num_process = len(processes_stack)
        num_finish_process = 0

        while num_finish_process < num_process:
            """
            Greedy algorithm to allocate hardware resources to processes.
            Three steps:
            1. Check resource availability.
            2. Allocate resources.
            3. Update process status.
            """

