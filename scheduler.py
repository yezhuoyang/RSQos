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
        self._current_avalible = { i: True for i in range(0, self._qubit_num + 1) }
        self._mapped_to_syndrome = { i: False for i in range(0, self._qubit_num + 1) }
        self._used_counter = { i: 0 for i in range(0, self._qubit_num + 1) }   # Count of how many process used each qubit. Syndrome qubit might be reused
        self._num_availble = self._qubit_num
        self._virtual_hardware_mapping = virtualHardwareMapping(hardware_instance)
        self._current_process_in_execution = []


    def get_virtual_hardware_mapping(self) -> virtualHardwareMapping:
        """
        Get the virtual hardware mapping.
        """
        return self._virtual_hardware_mapping

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
            self._virtual_hardware_mapping.add_mapping(addr, next_free_index_)
            next_free_index_=self.next_free_index(next_free_index_+1)

        next_free_index_=self.next_free_index(0)
        for addr in process_instance.get_virtual_syndrome_addresses():
            """
            Greedy algorithm: Use available qubit until none are available
            Otherwise, use the syndrome qubit with the lowest usage count.
            """
            if self._num_availble > 0:
                self._current_avalible[next_free_index_] = False
                self._used_counter[next_free_index_] = 1
                self._num_availble -= 1
                self._mapped_to_syndrome[next_free_index_] = True
                self._virtual_hardware_mapping.add_mapping(addr, next_free_index_)
                next_free_index_ = self.next_free_index(next_free_index_ + 1)
            else:
                least_busy_qubit = self.least_busy_syndrome_qubit()
                if least_busy_qubit != -1:
                    self._used_counter[least_busy_qubit] += 1
                    self._virtual_hardware_mapping.add_mapping(addr, least_busy_qubit)
                else:
                    assert False, "No available qubit found for syndrome mapping."


    def baseline_scheduling(self):
        """
        Base line scheduling algorithm.
        Naively allocate resources to processes in the order they arrive.
        """
        processes_stack = self._kernel.get_processes().copy()

        num_process = len(processes_stack)
        num_finish_process = 0
        total_qpu_time = 0
        for process_instance in processes_stack:
            if self.have_enough_resources(process_instance):
                self.allocate_resources(process_instance)
            else:
                assert False, "Process ask too many qubits!"

            while not process_instance.is_done():
                process_instance.execute_instruction()
            self.free_resources(process_instance)
            total_qpu_time += process_instance.get_consumed_qpu_time()

        return total_qpu_time


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



if __name__ == "__main__":

    vdata1 = virtualSpace(size=10, label="vdata1")
    vsyn1 = virtualSpace(size=5, label="vsyn1", is_syndrome=True)

    proc1 = process(processID=1,start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=3,processID=1))  # Allocate 2 syndrome qubits
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])  # CNOT operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(2)])  # CNOT operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(vdata1 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(vsyn1,processID=1))  # Allocate 2 syndrome qubits


    vdata2 = virtualSpace(size=10, label="vdata2")
    vsyn2 = virtualSpace(size=5, label="vsyn2", is_syndrome=True)

    proc2 = process(processID=2,start_time=3, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(size=3,processID=1))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=3,processID=1))  # Allocate 2 syndrome qubits
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])  # CNOT operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])  # CNOT operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(2)])  # CNOT operation
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(vdata2 ,processID=1))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(vsyn2,processID=1))  # Allocate 2 syndrome qubits

    #print(proc2)


    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=6, error_rate=0.001)


    schedule_instance=Scheduler(kernel_instance,virtual_hardware)
    time=schedule_instance.baseline_scheduling()

    print(f"Total time {time}")
    #print(kernel_instance)
