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
        The result of scheduling is a list of instruction which contain the instructions
        of all processes
        """
        processes_stack = self._kernel.get_processes().copy()

        num_process = len(processes_stack)
        num_finish_process = 0
        total_qpu_time = 0
        final_inst_list = []
        for process_instance in processes_stack:
            if self.have_enough_resources(process_instance):
                self.allocate_resources(process_instance)
            else:
                assert False, "Process ask too many qubits!"

            while not process_instance.is_done():
                inst=process_instance.execute_instruction(total_qpu_time)
                if isinstance(inst, instruction):
                    total_qpu_time += get_clocktime(inst.get_type())
                else:
                    total_qpu_time += get_syscall_time(inst)
                final_inst_list.append(inst)
            self.free_resources(process_instance)

        return total_qpu_time,final_inst_list


    def schedule(self):
        """
        Schedule processes for execution.
        The input is a kernel with a process queue.
        The output is a single process instance with virtual hardware mapping.
        """
        processes_stack = self._kernel.get_processes().copy()
        processes_stack.sort(key=lambda x: x.get_start_time())  # Sort processes by start time
        num_process = len(processes_stack)
        num_finish_process = 0
        total_qpu_time = 0
        final_inst_list = []
        process_finish_map = {i: False for i in processes_stack}

        active_processes = []
        while num_finish_process < num_process:
            """
            Greedy algorithm to allocate hardware resources to processes.
            Three steps:
            1. Check resource availability.
            2. Allocate resources.
            3. Update process status.
            """
            for process_instance in processes_stack:
                if not process_finish_map[process_instance]:
                    if process_instance not in active_processes:
                        if self.have_enough_resources(process_instance):
                            self.allocate_resources(process_instance)
                            active_processes.append(process_instance)

            """
            Execute instructions for active processes.
            Update the final_inst_list.
            These process run in parallel
            """
            cost_time = 0
            for process_instance in active_processes:
                inst = process_instance.execute_instruction()
                if isinstance(inst, instruction):
                    final_inst_list.append(inst)
                    tmp_time= get_clocktime(inst.get_type())
                    cost_time = max(cost_time, tmp_time)
            total_qpu_time += cost_time


            """
            Free resources for finished processes.
            """
            for process_instance in active_processes:
                if process_instance.is_done():
                    self.free_resources(process_instance)
                    process_finish_map[process_instance] = True
                    num_finish_process += 1
                    active_processes.remove(process_instance)


        return total_qpu_time,final_inst_list




    def print_instruction_list(self, inst_list):
        """
        Print the instruction list in an organized and clean format.
        For example:
            P1(t=5):  CNOT qubit 0 (->vspace[3]),  1 (->vspace[4])
            P1(t=6):  Syscall MAGIC_STATE_DISTILLATION qubit 0 (->vTspace[0]), 1 (->vTspace[1]), ...
        """
        for inst in inst_list:
            if isinstance(inst, instruction):
                process_id = inst.get_processID()
                inst_name = get_gate_type_name(inst.get_type())
                inst_time = inst.get_scheduled_time()
                addresses = inst.get_qubitaddress()

                mapped_addresses = [
                    f"{self._virtual_hardware_mapping.get_physical_qubit(addr)} (->{addr})"
                    for addr in addresses
                ]
                addr_str = ", ".join(mapped_addresses)
                print(f"P{process_id}(t={inst_time}): {inst_name} qubit {addr_str}")

            elif isinstance(inst, syscall):
                process_id = inst.get_processID()
                inst_name = get_syscall_type_name(inst)
                inst_time = inst.get_scheduled_time()
                addresses = inst.get_address()

                mapped_addresses = [
                    f"{self._virtual_hardware_mapping.get_physical_qubit(addr)} (->{addr})"
                    for addr in addresses
                ]
                addr_str = ", ".join(mapped_addresses)
                print(f"P{process_id}(t={inst_time}): Syscall {inst_name} qubit {addr_str}")

            else:
                print("Unknown instruction type.")


def generate_example():
    vdata1 = virtualSpace(size=10, label="vdata1")
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=5, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,2)
    proc1 = process(processID=1,start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1),vsyn1.get_address(2)],size=3,processID=1))  # Allocate 2 syndrome qubits
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])  # CNOT operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(2)])  # CNOT operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1),vsyn1.get_address(2)],size=3,processID=1))  # Allocate 2 syndrome qubits


    vdata2 = virtualSpace(size=10, label="vdata2")
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=5, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,2)
    proc2 = process(processID=2,start_time=3, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1),vsyn2.get_address(2)],size=3,processID=2))  # Allocate 2 syndrome qubits
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])  # CNOT operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])  # CNOT operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(2)])  # CNOT operation
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1),vsyn2.get_address(2)],size=3,processID=2))  # Allocate 2 syndrome qubits

    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001)

    return kernel_instance, virtual_hardware


def generate_example1():
    vdata1 = virtualSpace(size=10, label="vdata1")
    vsyn1 = virtualSpace(size=5, label="vsyn1", is_syndrome=True)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscall_allocate_data_qubits(size=3, processID=1))
    proc1.add_syscall(syscall_allocate_syndrome_qubits(size=3, processID=1))
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_syscall(syscall_deallocate_data_qubits(vdata1, processID=1))
    proc1.add_syscall(syscall_deallocate_syndrome_qubits(vsyn1, processID=1))

    vdata2 = virtualSpace(size=12, label="vdata2")
    vsyn2 = virtualSpace(size=6, label="vsyn2", is_syndrome=True)

    proc2 = process(processID=2, start_time=1, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscall_allocate_data_qubits(size=4, processID=2))
    proc2.add_syscall(syscall_allocate_syndrome_qubits(size=2, processID=2))
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2), vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])
    proc2.add_syscall(syscall_deallocate_data_qubits(vdata2, processID=2))
    proc2.add_syscall(syscall_deallocate_syndrome_qubits(vsyn2, processID=2))

    vdata3 = virtualSpace(size=8, label="vdata3")
    vsyn3 = virtualSpace(size=4, label="vsyn3", is_syndrome=True)

    proc3 = process(processID=3, start_time=2, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscall_allocate_data_qubits(size=2, processID=3))
    proc3.add_syscall(syscall_allocate_syndrome_qubits(size=2, processID=3))
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0), vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])
    proc3.add_syscall(syscall_deallocate_data_qubits(vdata3, processID=3))
    proc3.add_syscall(syscall_deallocate_syndrome_qubits(vsyn3, processID=3))

    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 
                                     'max_physical_qubits': 10000, 
                                     'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(qubit_number=12, error_rate=0.001)
    return kernel_instance, virtual_hardware


def generate_example2():
    vdata1 = virtualSpace(size=14, label="vdata1")
    vsyn1 = virtualSpace(size=7, label="vsyn1", is_syndrome=True)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscall_allocate_data_qubits(size=5, processID=1))
    proc1.add_syscall(syscall_allocate_syndrome_qubits(size=3, processID=1))
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_syscall(syscall_deallocate_data_qubits(vdata1, processID=1))
    proc1.add_syscall(syscall_deallocate_syndrome_qubits(vsyn1, processID=1))

    vdata2 = virtualSpace(size=10, label="vdata2")
    vsyn2 = virtualSpace(size=5, label="vsyn2", is_syndrome=True)

    proc2 = process(processID=2, start_time=1, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscall_allocate_data_qubits(size=3, processID=2))
    proc2.add_syscall(syscall_allocate_syndrome_qubits(size=3, processID=2))
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2), vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])
    proc2.add_syscall(syscall_deallocate_data_qubits(vdata2, processID=2))
    proc2.add_syscall(syscall_deallocate_syndrome_qubits(vsyn2, processID=2))

    vdata3 = virtualSpace(size=9, label="vdata3")
    vsyn3 = virtualSpace(size=4, label="vsyn3", is_syndrome=True)

    proc3 = process(processID=3, start_time=3, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscall_allocate_data_qubits(size=2, processID=3))
    proc3.add_syscall(syscall_allocate_syndrome_qubits(size=2, processID=3))
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0), vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])
    proc3.add_syscall(syscall_deallocate_data_qubits(vdata3, processID=3))
    proc3.add_syscall(syscall_deallocate_syndrome_qubits(vsyn3, processID=3))

    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 
                                     'max_physical_qubits': 10000, 
                                     'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(qubit_number=14, error_rate=0.0015)
    return kernel_instance, virtual_hardware







def generate_example3():
    vdata1 = virtualSpace(size=8, label="vdata1")
    vsyn1 = virtualSpace(size=4, label="vsyn1", is_syndrome=True)

    proc1 = process(1, 0, vdata1, vsyn1)
    proc1.add_syscall(syscall_allocate_data_qubits(2, 1))
    proc1.add_syscall(syscall_allocate_syndrome_qubits(2, 1))
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_syscall(syscall_deallocate_data_qubits(vdata1, 1))
    proc1.add_syscall(syscall_deallocate_syndrome_qubits(vsyn1, 1))

    vdata2 = virtualSpace(size=10, label="vdata2")
    vsyn2 = virtualSpace(size=5, label="vsyn2", is_syndrome=True)

    proc2 = process(2, 1, vdata2, vsyn2)
    proc2.add_syscall(syscall_allocate_data_qubits(3, 2))
    proc2.add_syscall(syscall_allocate_syndrome_qubits(3, 2))
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])
    proc2.add_syscall(syscall_deallocate_data_qubits(vdata2, 2))
    proc2.add_syscall(syscall_deallocate_syndrome_qubits(vsyn2, 2))

    vdata3 = virtualSpace(size=12, label="vdata3")
    vsyn3 = virtualSpace(size=6, label="vsyn3", is_syndrome=True)

    proc3 = process(3, 3, vdata3, vsyn3)
    proc3.add_syscall(syscall_allocate_data_qubits(4, 3))
    proc3.add_syscall(syscall_allocate_syndrome_qubits(2, 3))
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(2), vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])
    proc3.add_syscall(syscall_deallocate_data_qubits(vdata3, 3))
    proc3.add_syscall(syscall_deallocate_syndrome_qubits(vsyn3, 3))

    kernel_instance = Kernel({'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(12, 0.002)
    return kernel_instance, virtual_hardware


def generate_example4():
    vdata1 = virtualSpace(size=15, label="vdata1")
    vsyn1 = virtualSpace(size=7, label="vsyn1", is_syndrome=True)

    proc1 = process(1, 0, vdata1, vsyn1)
    proc1.add_syscall(syscall_allocate_data_qubits(5, 1))
    proc1.add_syscall(syscall_allocate_syndrome_qubits(3, 1))
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(4), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])
    proc1.add_syscall(syscall_deallocate_data_qubits(vdata1, 1))
    proc1.add_syscall(syscall_deallocate_syndrome_qubits(vsyn1, 1))

    vdata2 = virtualSpace(size=9, label="vdata2")
    vsyn2 = virtualSpace(size=5, label="vsyn2", is_syndrome=True)

    proc2 = process(2, 2, vdata2, vsyn2)
    proc2.add_syscall(syscall_allocate_data_qubits(3, 2))
    proc2.add_syscall(syscall_allocate_syndrome_qubits(2, 2))
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])
    proc2.add_syscall(syscall_deallocate_data_qubits(vdata2, 2))
    proc2.add_syscall(syscall_deallocate_syndrome_qubits(vsyn2, 2))

    vdata3 = virtualSpace(size=10, label="vdata3")
    vsyn3 = virtualSpace(size=6, label="vsyn3", is_syndrome=True)

    proc3 = process(3, 4, vdata3, vsyn3)
    proc3.add_syscall(syscall_allocate_data_qubits(4, 3))
    proc3.add_syscall(syscall_allocate_syndrome_qubits(2, 3))
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(2), vsyn3.get_address(3)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(3)])
    proc3.add_syscall(syscall_deallocate_data_qubits(vdata3, 3))
    proc3.add_syscall(syscall_deallocate_syndrome_qubits(vsyn3, 3))

    kernel_instance = Kernel({'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(15, 0.001)
    return kernel_instance, virtual_hardware


def generate_example5():
    vdata1 = virtualSpace(size=11, label="vdata1")
    vsyn1 = virtualSpace(size=5, label="vsyn1", is_syndrome=True)

    proc1 = process(1, 1, vdata1, vsyn1)
    proc1.add_syscall(syscall_allocate_data_qubits(3, 1))
    proc1.add_syscall(syscall_allocate_syndrome_qubits(2, 1))
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])
    proc1.add_syscall(syscall_deallocate_data_qubits(vdata1, 1))
    proc1.add_syscall(syscall_deallocate_syndrome_qubits(vsyn1, 1))

    vdata2 = virtualSpace(size=12, label="vdata2")
    vsyn2 = virtualSpace(size=6, label="vsyn2", is_syndrome=True)

    proc2 = process(2, 2, vdata2, vsyn2)
    proc2.add_syscall(syscall_allocate_data_qubits(4, 2))
    proc2.add_syscall(syscall_allocate_syndrome_qubits(3, 2))
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2), vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])
    proc2.add_syscall(syscall_deallocate_data_qubits(vdata2, 2))
    proc2.add_syscall(syscall_deallocate_syndrome_qubits(vsyn2, 2))

    vdata3 = virtualSpace(size=14, label="vdata3")
    vsyn3 = virtualSpace(size=7, label="vsyn3", is_syndrome=True)

    proc3 = process(3, 4, vdata3, vsyn3)
    proc3.add_syscall(syscall_allocate_data_qubits(5, 3))
    proc3.add_syscall(syscall_allocate_syndrome_qubits(3, 3))
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(4), vsyn3.get_address(1)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(1)])
    proc3.add_syscall(syscall_deallocate_data_qubits(vdata3, 3))
    proc3.add_syscall(syscall_deallocate_syndrome_qubits(vsyn3, 3))

    kernel_instance = Kernel({'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(14, 0.0012)
    return kernel_instance, virtual_hardware


def generate_example6():
    vdata1 = virtualSpace(size=9, label="vdata1")
    vsyn1 = virtualSpace(size=4, label="vsyn1", is_syndrome=True)

    proc1 = process(1, 0, vdata1, vsyn1)
    proc1.add_syscall(syscall_allocate_data_qubits(3, 1))
    proc1.add_syscall(syscall_allocate_syndrome_qubits(2, 1))
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_syscall(syscall_deallocate_data_qubits(vdata1, 1))
    proc1.add_syscall(syscall_deallocate_syndrome_qubits(vsyn1, 1))

    vdata2 = virtualSpace(size=10, label="vdata2")
    vsyn2 = virtualSpace(size=5, label="vsyn2", is_syndrome=True)

    proc2 = process(2, 1, vdata2, vsyn2)
    proc2.add_syscall(syscall_allocate_data_qubits(4, 2))
    proc2.add_syscall(syscall_allocate_syndrome_qubits(3, 2))
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(3), vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])
    proc2.add_syscall(syscall_deallocate_data_qubits(vdata2, 2))
    proc2.add_syscall(syscall_deallocate_syndrome_qubits(vsyn2, 2))

    vdata3 = virtualSpace(size=11, label="vdata3")
    vsyn3 = virtualSpace(size=5, label="vsyn3", is_syndrome=True)

    proc3 = process(3, 3, vdata3, vsyn3)
    proc3.add_syscall(syscall_allocate_data_qubits(3, 3))
    proc3.add_syscall(syscall_allocate_syndrome_qubits(2, 3))
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0), vsyn3.get_address(1)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(1)])
    proc3.add_syscall(syscall_deallocate_data_qubits(vdata3, 3))
    proc3.add_syscall(syscall_deallocate_syndrome_qubits(vsyn3, 3))

    kernel_instance = Kernel({'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    virtual_hardware = virtualHardware(11, 0.0011)
    return kernel_instance, virtual_hardware



if __name__ == "__main__":

    kernel_instance, virtual_hardware = generate_example()
    schedule_instance=Scheduler(kernel_instance,virtual_hardware)
    time1, inst_list1=schedule_instance.baseline_scheduling()




    mapping = schedule_instance.get_virtual_hardware_mapping()


   #print("Mapping after scheduling:")

    #print(mapping)


    kernel_instance, virtual_hardware = generate_example()
    schedule_instance=Scheduler(kernel_instance,virtual_hardware)
    time2, inst_list2=schedule_instance.schedule()


    print("Baseline: {}".format(time1))


    print("Our: {}".format(time2))
    #print(kernel_instance)


    schedule_instance.print_instruction_list(inst_list2)




    # stim_circuit = mapping.transpile(inst_list)

    # print(stim_circuit)