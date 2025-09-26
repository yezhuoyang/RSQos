from typing import Union
from kernel import *
from process import *
from instruction import *
from hardware import *
from instruction import *
from syscall import *
import qiskit
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer
from qiskit.circuit.library import HGate
from collections import deque


def all_pairs_distances(n, edges):
    """
    n: number of vertices labeled 0..n-1
    edges: list of [u, v] pairs (undirected)
    returns: n x n list of ints, distances; -1 means unreachable
    """
    # build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u},{v}) out of range for n={n}")
        adj[u].append(v)
        adj[v].append(u)

    # BFS from every source
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dq = deque([s])
        dist[s][s] = 0
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if dist[s][w] == -1:
                    dist[s][w] = dist[s][u] + 1
                    dq.append(w)
    return dist


"""
The real hardware job that is sent to hardware.
The output of the scheduler is a set of HardwareJobs.

For example, if a kernel has 3 processes. 
P1:1000 shots, P2:2000 shots, P3:3000 shots

Then, the scheduler will generate 2 HardwareJobs:

Job1: [P1, P2, P3], 1000 shots
Job2: [P2, P3, P3], 1000 shots

Note that in one HardwareJob, all processes must have the same number of shots, but one
processes can appear many times
"""
class HardwareJob:
    def __init__(self, instruction_list: List[Union[instruction, syscall]], total_time: int, total_shot: int):
        self._instruction_list = instruction_list
        self._total_time = total_time
        self._total_shot = total_shot


    def get_instruction_list(self) -> List[Union[instruction, syscall]]:
        return self._instruction_list


    def calc_result(self):
        pass





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
        self._dynamic_virtual_hardware_mapping = dynamicvirtualHardwareMapping(hardware_instance) #Keep track of the dynamic mapping of syndrome qubit
        self._current_process_in_execution = []
        self._num_measurement = 0  # Keep track of the number of measurement instructions executed

        self._syndrome_map_history = {}  # Keep track of all syndrome qubits that are used
        """
        Store the mapping between the index of measurement instruction to the process ID that generates it.
        """
        self._measure_index_to_process = {}  # Map the index of measurement instruction to the process ID that generates it
        self._process_measure_index = {}  # Map the process ID to the list of measurement instruction indices it generates
        """
        Store the distance between every pair of qubits on the hardware.
        """
        self._all_pair_distance = None



    def calculate_all_pair_distance(self):
        """
        Calculate the distance between every pair of qubits on the hardware.
        """
        edges = self._hardware.get_edge_list()
        n = self._hardware.get_qubit_num()
        self._all_pair_distance = all_pairs_distances(n, edges)
        return self._all_pair_distance



    def return_process_ideal_output(self):
        """
        For each process, return the ideal output distribution.
        """
        all_process_ideal_counts = {}
        for process_instance in self._kernel.get_processes():
            process_instance.construct_qiskit_circuit()
            counts = process_instance.simulate_circuit(shots=process_instance.get_total_shots())
            process_id = process_instance.get_processID()
            all_process_ideal_counts[process_id] = counts
        return all_process_ideal_counts


    def return_measure_states(self, backend_result_counts):
        """
        After simulation/execution on hardware, return the measurement results for each process.
        This is the key interface, which convert the real hardware result
        back to the result of each process.
        """
        all_process_counts = {}
        
        for process_id, measure_indices in self._process_measure_index.items():
            process_counts = {}
            for bitstring, count in backend_result_counts.items():
                # Extract the bits corresponding to the current process's measurements
                extracted_bits = ''.join(bitstring[self._num_measurement-1-idx] for idx in measure_indices)
                # Reverse the bitstring to match Qiskit's output format
                extracted_bits = extracted_bits[::-1]
                if extracted_bits in process_counts:
                    process_counts[extracted_bits] += count
                else:
                    process_counts[extracted_bits] = count
            print(f"\n=== Counts for Process {process_id} ===")
            print(process_counts)
            all_process_counts[process_id] = process_counts
        return all_process_counts


    def get_syndrome_map_history(self):
        return self._syndrome_map_history


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
    

    def least_cost_data_qubit_for_initial_mapping(self, dataaddr:virtualAddress, process_instance: process) -> int:
        """
        Based on connectivity, current mapping of data qubits, the virtual data qubit,
        return the physical qubit that is unused, but has the least cost to map to the virtual data qubit.
        This is used for greedy initial mapping of data qubits.
        """
        sorted_list= process_instance.ranked_data_mapping_cost(dataaddr, self._current_avalible)
        assert len(sorted_list)>0, "No available qubit found for data qubit mapping."
        return sorted_list[0][0]


    def least_cost_unused_qubit_for_syn(self, synaddr:virtualAddress, process_instance: process) -> int:
        """
        Based on connectivity, current mapping of data qubits, the virtual syndrome qubit,
        return the physical qubit that is unused, but has the least cost to map to the virtual syndrome qubit.
        """
        sorted_list= process_instance.ranked_syn_mapping_cost(synaddr, self._current_avalible)
        assert len(sorted_list)>0, "No available qubit found for syndrome mapping."
        return sorted_list[0][0]




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


    def greedy_allocate_data_qubit(self, process_instance: process):
        """
        Allocate data qubits for a process.
        This function is used for dynamic scheduling algorithm.
        Each data qubit must be mapped to different physical qubits.

        We use greedy algorithm to set the initial mapping of data qubits.
        TODO: Optimize the algorithm, reduce redundant sorting operations.
        """

        for addr in process_instance.get_virtual_data_addresses():
            """
            Find the next available index
            Case 1: The address is a data qubit, find the first availble qubit

            Notice that the mapping cost must be updated freshly for each data qubit!
            """
            process_instance.calc_data_mapping_cost(self._all_pair_distance, self._hardware.get_qubit_num())
            free_index_ = self.least_cost_data_qubit_for_initial_mapping(addr, process_instance)
            self._current_avalible[free_index_] = False
            self._used_counter[free_index_] = 1
            self._num_availble -= 1
            self._virtual_hardware_mapping.add_mapping(addr, free_index_)
            process_instance.set_data_qubit_virtual_hardware_mapping(addr, free_index_)
 




    def greedy_allocate_syndrome_qubit(self, process_instance: process):
        """
        Greedy allocate syndrome qubits for a process.
        """
        for addr in process_instance.get_virtual_syndrome_addresses():
            """
            Greedy algorithm: Use available qubit until none are available
            Otherwise, use the syndrome qubit with the lowest usage count.

            There are two cases:
              1. Free qubits are available, use the one with the least cost
              2. No free qubits, use the least busy syndrome qubit
            """
            if self._num_availble > 0:
                free_index_ = self.least_cost_unused_qubit_for_syn(addr, process_instance)
                self._current_avalible[free_index_] = False
                self._used_counter[free_index_] = 1
                self._num_availble -= 1
                self._virtual_hardware_mapping.add_mapping(addr, free_index_)
                process_instance.set_syndrome_qubit_virtual_hardware_mapping(addr, free_index_) 
            else:
                least_busy_qubit = self.least_busy_syndrome_qubit()
                if least_busy_qubit != -1:
                    self._used_counter[least_busy_qubit] += 1
                    self._virtual_hardware_mapping.add_mapping(addr, least_busy_qubit)
                    self._virtual_hardware_mapping.add_mapping(addr, free_index_)
                    process_instance.set_syndrome_qubit_virtual_hardware_mapping(addr, least_busy_qubit)
                else:
                    assert False, "No available qubit found for syndrome mapping."



    def free_syndrome_qubit(self, process_instance: process):
        """
        Free syndrome qubits allocated to a process.
        This function is used for dynamic scheduling algorithm.
        Syndrome qubits can be reused.
        """
        for addr in process_instance.get_virtual_syndrome_addresses():
            if process_instance.syndrome_qubit_is_allocated(addr):
                hardware_qid = process_instance.get_syndrome_qubit_virtual_hardware_mapping(addr)
                self._used_counter[hardware_qid] -= 1
                self._current_avalible[hardware_qid] = True
                self._num_availble += 1
                process_instance.empty_syndrome_qubit_mappings(addr)



    def free_data_qubit(self, process_instance: process):
        """
        Free data qubits allocated to a process.
        """
        for addr in process_instance.get_virtual_data_addresses():
            hardware_qid = process_instance.get_data_qubit_virtual_hardware_mapping(addr)
            self._used_counter[hardware_qid] -= 1
            self._current_avalible[hardware_qid] = True
            self._num_availble += 1




    def allocate_resources(self, process_instance: process):
        """
        Allocate resources for a process.(This is used for static mapping)
        Now assume that the number of data qubits required is less than or equal to the number of available physical qubits.
        Idea: Each data qubit must be mapped to different physical qubits.
              However, syndrome qubits can be reused.
        """
        process_instance.analyze_data_qubit_connectivity()
        self.greedy_allocate_data_qubit(process_instance)
        process_instance.analyze_syndrome_connectivity()
        process_instance.calc_syn_mapping_cost(self._all_pair_distance, self._hardware.get_qubit_num())
        self.greedy_allocate_syndrome_qubit(process_instance)


    def advanced_scheduling(self):
        """
        Dynamic scheduling algorithm with sharing syndrome qubits between processes in the same batch.
        Also, consider the connectivity of hardware when allocating syndrome qubits.
        """
        pass


    def scheduling_with_out_sharing_syndrome_qubit(self):
        """
        Put processes into a batch. But not sharying resources between processes in the same batch.
        """
        processes_stack = self._kernel.get_processes().copy()

        num_finish_process = 0
        total_qpu_time = 0
        final_inst_list = []
        current_measurement_index = 0 

        """
        Analysis and calculate the connectivity and mapping cost for each process.
        """
        self.calculate_all_pair_distance()

        while num_finish_process < len(processes_stack):
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.FINISHED:
                    continue

                if process_instance.get_status() == ProcessStatus.WAIT_TO_START:
                    process_instance.analyze_data_qubit_connectivity()
                    if self.have_enough_resources(process_instance):
                        self.allocate_resources(process_instance)
                        process_instance.set_status(ProcessStatus.RUNNING)
                    continue

                if process_instance.get_status() == ProcessStatus.RUNNING:

                    inst=process_instance.execute_instruction(total_qpu_time)
                    if isinstance(inst, instruction):
                        total_qpu_time += get_clocktime(inst.get_type())
                    else:
                        total_qpu_time += get_syscall_time(inst)
                    addresses = inst.get_qubitaddress() if isinstance(inst, instruction) else inst.get_address()
                    for addr in addresses:
                        physical_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
                        inst.set_scheduled_mapped_address(addr, physical_qid)
                    final_inst_list.append(inst)

                    if isinstance(inst, instruction):
                        if inst.is_measurement():
                            self._num_measurement += 1
                            self._measure_index_to_process[ current_measurement_index ] = process_instance.get_processID()
                            if not process_instance.get_processID() in self._process_measure_index.keys():
                                self._process_measure_index[ process_instance.get_processID() ] = [current_measurement_index]
                            else:
                                self._process_measure_index[ process_instance.get_processID() ].append(current_measurement_index)
                            current_measurement_index += 1


                            for addr in inst.get_qubitaddress():
                                physical_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
                                newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                                newReset.set_scheduled_mapped_address(addr, physical_qid)
                                newReset.set_scheduled_time(total_qpu_time)
                                final_inst_list.append(newReset)

                if process_instance.is_done():
                    process_instance.set_status(ProcessStatus.FINISHED)
                    num_finish_process += 1
                    #Reset all data qubits to |0> after the process is done
                    for addr in process_instance.get_virtual_data_addresses():
                        physical_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
                        newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                        newReset.set_scheduled_mapped_address(addr, physical_qid)
                        newReset.set_scheduled_time(total_qpu_time)
                        final_inst_list.append(newReset)
                    self.free_resources(process_instance)
                    newReset=instruction(type=Instype.BARRIER,qubitaddress=None, processID=process_instance.get_processID(), time=total_qpu_time)
                    final_inst_list.append(newReset)            

        return total_qpu_time,final_inst_list



    def baseline_scheduling(self):
        """
        Base line scheduling algorithm.
        Naively allocate resources to processes in the order they arrive.
        The result of scheduling is a list of instruction which contain the instructions
        of all processes
        """
        processes_stack = self._kernel.get_processes().copy()
        total_qpu_time = 0
        final_inst_list = []
        current_measurement_index = 0 

        """
        Analysis and calcualte the connectivity and mapping cost for each process.
        """
        self.calculate_all_pair_distance()

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
                addresses = inst.get_qubitaddress() if isinstance(inst, instruction) else inst.get_address()
                for addr in addresses:
                    physical_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
                    inst.set_scheduled_mapped_address(addr, physical_qid)
                final_inst_list.append(inst)

                if isinstance(inst, instruction):
                    if inst.is_measurement():
                        self._num_measurement += 1
                        self._measure_index_to_process[ current_measurement_index ] = process_instance.get_processID()
                        if not process_instance.get_processID() in self._process_measure_index.keys():
                            self._process_measure_index[ process_instance.get_processID() ] = [current_measurement_index]
                        else:
                            self._process_measure_index[ process_instance.get_processID() ].append(current_measurement_index)
                        current_measurement_index += 1


                        for addr in inst.get_qubitaddress():
                            physical_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
                            newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                            newReset.set_scheduled_mapped_address(addr, physical_qid)
                            newReset.set_scheduled_time(total_qpu_time)
                            final_inst_list.append(newReset)

            #Reset all data qubits to |0> after the process is done
            for addr in process_instance.get_virtual_data_addresses():
                physical_qid = self._virtual_hardware_mapping.get_physical_qubit(addr)
                newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                newReset.set_scheduled_mapped_address(addr, physical_qid)
                newReset.set_scheduled_time(total_qpu_time)
                final_inst_list.append(newReset)

            self.free_resources(process_instance)
            newReset=instruction(type=Instype.BARRIER,qubitaddress=None, processID=process_instance.get_processID(), time=total_qpu_time)
            final_inst_list.append(newReset)            
        return total_qpu_time,final_inst_list



    def get_all_processes(self):
        """
        Get all processes in the kernel.
        """
        return self._kernel.get_processes()




    def dynamic_scheduling_no_consider_connectivity(self):
        """
        Baseline dynamic scheduling algorithm without considering the connectivity of hardware.
        """
        processes_stack = self._kernel.get_processes().copy()
        processes_stack.sort(key=lambda x: x.get_start_time())  # Sort processes by start time
        num_process = len(processes_stack)
        num_finish_process = 0
        process_finish_map = {i: False for i in processes_stack}
        total_qpu_time = 0
        final_inst_list = []
        current_measurement_index = 0


        while num_finish_process < num_process:
            """
            Three steps:
            1. For process still wait to be started, check data qubit availability. Allocate data qubits if possible.
            2. For process that wait for syndrome qubit, allocate syndrome qubit if possible.
            3. For process in execution, execute the next instruction.
            4. Update process status, free resources if process is done.
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.WAIT_TO_START:
                    process_instance.analyze_data_qubit_connectivity()
                    if self.have_enough_resources(process_instance):
                        self.greedy_allocate_data_qubit(process_instance)
                        process_instance.set_status(ProcessStatus.WAIT_FOR_ANSILLA)
            """
            Collect all the instructions that are waiting for syndrome qubit
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.WAIT_FOR_ANSILLA:
                    num_ansilla_qubit_needed = 0
                    next_inst = process_instance.get_next_instruction()
                    if isinstance(next_inst, syscall):
                        """
                        When the next instruction is a syscall for T factory or magic state distillation,
                        we change the status of the process to WAIT_FOR_T_GATE
                        """
                        if isinstance(next_inst, syscall_magic_state_distillation):
                            process_instance.set_status(ProcessStatus.WAIT_FOR_T_GATE)
                        else:
                            process_instance.set_status(ProcessStatus.RUNNING)
                        continue
                    addresses = next_inst.get_qubitaddress()
                    
                    for addr in addresses:
                        if process_instance.is_syndrome_qubit(addr):
                            if not process_instance.syndrome_qubit_is_allocated(addr):
                                num_ansilla_qubit_needed += 1
                            else:
                                physical_qid = process_instance.get_syndrome_qubit_virtual_hardware_mapping(addr)
                                next_inst.set_scheduled_mapped_address(addr, physical_qid)
                        elif process_instance.is_data_qubit(addr):
                            """
                            Update the mapping of data qubit in the instruction
                            """
                            physical_qid = process_instance.get_data_qubit_virtual_hardware_mapping(addr)
                            next_inst.set_scheduled_mapped_address(addr, physical_qid)
                        
                    if self._num_availble >= num_ansilla_qubit_needed:
                        next_free_index_ = self.next_free_index(0)
                        for addr in addresses:
                            if process_instance.is_syndrome_qubit(addr) and not process_instance.syndrome_qubit_is_allocated(addr):
                                self._current_avalible[next_free_index_] = False
                                if not next_free_index_ in self._syndrome_map_history.keys():
                                    self._syndrome_map_history[next_free_index_] = [addr]
                                else:
                                    self._syndrome_map_history[next_free_index_].append(addr)
                                self._used_counter[next_free_index_] = 1
                                self._num_availble -= 1
                                next_inst.set_scheduled_mapped_address(addr, next_free_index_)
                                process_instance.set_syndrome_qubit_virtual_hardware_mapping(addr, next_free_index_)
                                next_free_index_ = self.next_free_index(next_free_index_ + 1)
                        process_instance.set_status(ProcessStatus.RUNNING)


            """
            If there is any process waiting for T gate,
            we use greedy algorithm to allocate T gate resources.
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.WAIT_FOR_T_GATE:
                    t_next_inst = process_instance.get_next_instruction()
                    addresses = t_next_inst.get_address()    
                    if self._num_availble >= len(addresses):
                        next_free_index_ = self.next_free_index(0)
                        for addr in addresses:
                            self._current_avalible[next_free_index_] = False
                            self._used_counter[next_free_index_] = 1
                            self._num_availble -= 1
                            t_next_inst.set_scheduled_mapped_address(addr, next_free_index_)
                            process_instance.set_syndrome_qubit_virtual_hardware_mapping(addr, next_free_index_)
                            next_free_index_ = self.next_free_index(next_free_index_)
                        process_instance.set_status(ProcessStatus.RUNNING)       



            """
            Execute the next instruction for all processes in RUNNING status.
            These processes run in parallel.
            Update the total_qpu_time by the maximum time cost among all processes.
            Free ansilla qubit after running the instruction.
            """
            move_time = 0
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.RUNNING:
                    next_inst = process_instance.get_next_instruction()
                    process_instance.execute_instruction(total_qpu_time)
                    if not next_inst == None:
                        if isinstance(next_inst, instruction):
                            tmp_time = get_clocktime(next_inst.get_type())
                        else:
                            tmp_time = get_syscall_time(next_inst)
                        move_time = max(move_time, tmp_time)
                        final_inst_list.append(next_inst)

                        if isinstance(next_inst, syscall_magic_state_distillation):
                            """
                            If the instruction is a syscall for magic state distillation,
                            We also have to freed the ansilla qubits used in this syscall
                            """
                            addresses = next_inst.get_address()
                            for addr in addresses:
                                physical_qid = next_inst.get_scheduled_mapped_address(addr)
                                self._used_counter[physical_qid] -= 1
                                self._current_avalible[physical_qid] = True
                                self._num_availble += 1
                                process_instance.empty_syndrome_qubit_mappings(addr)


                        if isinstance(next_inst, instruction):
                            if next_inst.is_measurement():
                                addresses = next_inst.get_qubitaddress()
                                self._num_measurement += 1
                                self._measure_index_to_process[ current_measurement_index ] = process_instance.get_processID()
                                if not process_instance.get_processID() in self._process_measure_index.keys():
                                    self._process_measure_index[ process_instance.get_processID() ] = [current_measurement_index]
                                else:
                                    self._process_measure_index[ process_instance.get_processID() ].append(current_measurement_index)
                                current_measurement_index += 1
                                """
                                If the instruction is a measurement instruction
                                Free the ansilla qubit used in this instruction
                                The state of this process will be changed to WAIT_FOR_ANSILLA
                                """
                                for addr in addresses:
                                    physical_qid = next_inst.get_scheduled_mapped_address(addr)
                                    if process_instance.is_syndrome_qubit(addr):
                                        self._used_counter[physical_qid] -= 1
                                        self._current_avalible[physical_qid] = True
                                        self._num_availble += 1
                                        process_instance.empty_syndrome_qubit_mappings(addr)

                                    """
                                    After measurement, we need to reset the measured qubit to |0>!
                                    """
                                    newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                                    newReset.set_scheduled_mapped_address(addr, physical_qid)
                                    newReset.set_scheduled_time(total_qpu_time)
                                    final_inst_list.append(newReset)

                    if not process_instance.get_status() == ProcessStatus.FINISHED:
                        process_instance.set_status(ProcessStatus.WAIT_FOR_ANSILLA)
            total_qpu_time += move_time

            """
            Free all data qubit resources for finished processes.
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.FINISHED and (not process_finish_map[process_instance]):
                    #Add reset for all data qubits to |0> after the process is done
                    #TODO: Optimize it in the future development
                    for addr in process_instance.get_virtual_data_addresses():
                        physical_qid = process_instance.get_data_qubit_virtual_hardware_mapping(addr)
                        newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                        newReset.set_scheduled_mapped_address(addr, physical_qid)
                        newReset.set_scheduled_time(total_qpu_time)
                        final_inst_list.append(newReset)
                    self.free_data_qubit(process_instance)
                    self.free_syndrome_qubit(process_instance)
                    num_finish_process += 1
                    process_finish_map[process_instance] = True
        return total_qpu_time, final_inst_list        





    def scheduling_not_share_syndrome_qubit(self):
        """
        The scheduling algorithm that does not share syndrome qubits between processes.
        """
        pass





    def dynamic_scheduling(self):
        """
        Our main algorithm
        Schedule processes for execution where syndrome qubits are dynamically allocated.
        (Each syndrome qubit in the virtual space is allocated on-the-fly)
        TODO: Optimize the allocation of syndrome qubits, should consider the connectivity of hardware.
        """
        processes_stack = self._kernel.get_processes().copy()
        processes_stack.sort(key=lambda x: x.get_start_time())  # Sort processes by start time
        num_process = len(processes_stack)
        num_finish_process = 0
        process_finish_map = {i: False for i in processes_stack}
        total_qpu_time = 0
        final_inst_list = []
        current_measurement_index = 0


        """
        Analysis and calcualte the connectivity and mapping cost for each process.
        """
        self.calculate_all_pair_distance()


        while num_finish_process < num_process:
            """
            Three steps:
            1. For process still wait to be started, check data qubit availability. Allocate data qubits if possible.
            2. For process that wait for syndrome qubit, allocate syndrome qubit if possible.
            3. For process in execution, execute the next instruction.
            4. Update process status, free resources if process is done.
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.WAIT_TO_START:
                    if self.have_enough_resources(process_instance):
                        """
                        First, analyze the connectivity within data qubits for greedy initial mapping.
                        """
                        process_instance.analyze_data_qubit_connectivity()
                        process_instance.calc_data_mapping_cost(self._all_pair_distance, self._hardware.get_qubit_num())
                        self.greedy_allocate_data_qubit(process_instance)
                        process_instance.set_status(ProcessStatus.WAIT_FOR_ANSILLA)
                        """
                        Once the data qubits are allocated, we can analyze the connectivity and mapping cost for this process.
                        """
                        process_instance.analyze_syndrome_connectivity()
                        process_instance.calc_syn_mapping_cost(self._all_pair_distance, self._hardware.get_qubit_num())

            """
            Collect all the instructions that are waiting for syndrome qubit
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.WAIT_FOR_ANSILLA:
                    num_ansilla_qubit_needed = 0
                    next_inst = process_instance.get_next_instruction()
                    if isinstance(next_inst, syscall):
                        """
                        When the next instruction is a syscall for T factory or magic state distillation,
                        we change the status of the process to WAIT_FOR_T_GATE
                        """
                        if isinstance(next_inst, syscall_magic_state_distillation):
                            process_instance.set_status(ProcessStatus.WAIT_FOR_T_GATE)
                        else:
                            process_instance.set_status(ProcessStatus.RUNNING)
                        continue
                    addresses = next_inst.get_qubitaddress()
                    
                    for addr in addresses:
                        if process_instance.is_syndrome_qubit(addr):
                            if not process_instance.syndrome_qubit_is_allocated(addr):
                                num_ansilla_qubit_needed += 1
                            else:
                                physical_qid = process_instance.get_syndrome_qubit_virtual_hardware_mapping(addr)
                                next_inst.set_scheduled_mapped_address(addr, physical_qid)
                        elif process_instance.is_data_qubit(addr):
                            """
                            Update the mapping of data qubit in the instruction
                            """
                            physical_qid = process_instance.get_data_qubit_virtual_hardware_mapping(addr)
                            next_inst.set_scheduled_mapped_address(addr, physical_qid)
                        
                    if self._num_availble >= num_ansilla_qubit_needed:
                        #next_free_index_ = self.next_free_index(0)
                        for addr in addresses:
                            if process_instance.is_syndrome_qubit(addr) and not process_instance.syndrome_qubit_is_allocated(addr):
                                best_free_index = self.least_cost_unused_qubit_for_syn(addr, process_instance)
                                self._current_avalible[best_free_index] = False
                                if not best_free_index in self._syndrome_map_history.keys():
                                    self._syndrome_map_history[best_free_index] = [addr]
                                else:
                                    self._syndrome_map_history[best_free_index].append(addr)
                                self._used_counter[best_free_index] = 1
                                self._num_availble -= 1
                                next_inst.set_scheduled_mapped_address(addr, best_free_index)
                                process_instance.set_syndrome_qubit_virtual_hardware_mapping(addr, best_free_index)

                        process_instance.set_status(ProcessStatus.RUNNING)


            """
            If there is any process waiting for T gate,
            we use greedy algorithm to allocate T gate resources.
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.WAIT_FOR_T_GATE:
                    t_next_inst = process_instance.get_next_instruction()
                    addresses = t_next_inst.get_address()    
                    if self._num_availble >= len(addresses):
                        next_free_index_ = self.next_free_index(0)
                        for addr in addresses:
                            self._current_avalible[next_free_index_] = False
                            self._used_counter[next_free_index_] = 1
                            self._num_availble -= 1
                            t_next_inst.set_scheduled_mapped_address(addr, next_free_index_)
                            process_instance.set_syndrome_qubit_virtual_hardware_mapping(addr, next_free_index_)
                            next_free_index_ = self.next_free_index(next_free_index_)
                        process_instance.set_status(ProcessStatus.RUNNING)       



            """
            Execute the next instruction for all processes in RUNNING status.
            These processes run in parallel.
            Update the total_qpu_time by the maximum time cost among all processes.
            Free ansilla qubit after running the instruction.
            """
            move_time = 0
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.RUNNING:
                    next_inst = process_instance.get_next_instruction()
                    process_instance.execute_instruction(total_qpu_time)
                    if not next_inst == None:
                        if isinstance(next_inst, instruction):
                            tmp_time = get_clocktime(next_inst.get_type())
                        else:
                            tmp_time = get_syscall_time(next_inst)
                        move_time = max(move_time, tmp_time)
                        final_inst_list.append(next_inst)

                        if isinstance(next_inst, syscall_magic_state_distillation):
                            """
                            If the instruction is a syscall for magic state distillation,
                            We also have to freed the ansilla qubits used in this syscall
                            """
                            addresses = next_inst.get_address()
                            for addr in addresses:
                                physical_qid = next_inst.get_scheduled_mapped_address(addr)
                                self._used_counter[physical_qid] -= 1
                                self._current_avalible[physical_qid] = True
                                self._num_availble += 1
                                process_instance.empty_syndrome_qubit_mappings(addr)


                        if isinstance(next_inst, instruction):
                            if next_inst.is_measurement():
                                addresses = next_inst.get_qubitaddress()
                                self._num_measurement += 1
                                self._measure_index_to_process[ current_measurement_index ] = process_instance.get_processID()
                                if not process_instance.get_processID() in self._process_measure_index.keys():
                                    self._process_measure_index[ process_instance.get_processID() ] = [current_measurement_index]
                                else:
                                    self._process_measure_index[ process_instance.get_processID() ].append(current_measurement_index)
                                current_measurement_index += 1
                                """
                                If the instruction is a measurement instruction
                                Free the ansilla qubit used in this instruction
                                The state of this process will be changed to WAIT_FOR_ANSILLA
                                """
                                for addr in addresses:
                                    physical_qid = next_inst.get_scheduled_mapped_address(addr)
                                    if process_instance.is_syndrome_qubit(addr):
                                        self._used_counter[physical_qid] -= 1
                                        self._current_avalible[physical_qid] = True
                                        self._num_availble += 1
                                        process_instance.empty_syndrome_qubit_mappings(addr)

                                    """
                                    After measurement, we need to reset the measured qubit to |0>!
                                    """
                                    newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                                    newReset.set_scheduled_mapped_address(addr, physical_qid)
                                    newReset.set_scheduled_time(total_qpu_time)
                                    final_inst_list.append(newReset)

                    if not process_instance.get_status() == ProcessStatus.FINISHED:
                        process_instance.set_status(ProcessStatus.WAIT_FOR_ANSILLA)
            total_qpu_time += move_time

            """
            Free all data qubit resources for finished processes.
            """
            for process_instance in processes_stack:
                if process_instance.get_status() == ProcessStatus.FINISHED and (not process_finish_map[process_instance]):
                    #Add reset for all data qubits to |0> after the process is done
                    #TODO: Optimize it in the future development
                    for addr in process_instance.get_virtual_data_addresses():
                        physical_qid = process_instance.get_data_qubit_virtual_hardware_mapping(addr)
                        newReset=instruction(type=Instype.RESET, qubitaddress=[addr], processID=process_instance.get_processID(), time=total_qpu_time)
                        newReset.set_scheduled_mapped_address(addr, physical_qid)
                        newReset.set_scheduled_time(total_qpu_time)
                        final_inst_list.append(newReset)
                    self.free_data_qubit(process_instance)
                    self.free_syndrome_qubit(process_instance)
                    num_finish_process += 1
                    process_finish_map[process_instance] = True
        return total_qpu_time, final_inst_list

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
                inst = process_instance.execute_instruction(total_qpu_time)
                if not inst==None:
                    if isinstance(inst, instruction):
                        tmp_time= get_clocktime(inst.get_type())
                    else:
                        tmp_time= get_syscall_time(inst)
                    final_inst_list.append(inst)
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



    def print_dynamic_instruction_list(self, inst_list):
        """
        Print the result of instruction list of dynamic scheduling in an organized and clean format.
        For example:
            P1(t=5):  CNOT qubit 0 (->vspace[3]),  1 (->vspace[4])
            P1(t=6):  Syscall MAGIC_STATE_DISTILLATION qubit 0 (->vTspace[0]), 1 (->vTspace[1]), ...
        """
        for inst in inst_list:
            if isinstance(inst, instruction):
                if inst.get_type() == Instype.BARRIER:
                    process_id = inst.get_processID()
                    print(f"P{process_id}(t={inst_time}): BARRIER")
                    continue
                process_id = inst.get_processID()
                inst_name = get_gate_type_name(inst.get_type())
                inst_time = inst.get_scheduled_time()
                addresses = inst.get_qubitaddress()

                mapped_addresses = [
                    f"{inst.get_scheduled_mapped_address(addr)} (->{addr})"
                    for addr in addresses
                ]
                addr_str = ", ".join(mapped_addresses)
                print(f"P{process_id}(t={inst_time}): {inst_name} qubit {addr_str}")

            elif isinstance(inst, syscall):
                process_id = inst.get_processID()
                inst_name = get_syscall_type_name(inst)
                inst_time = inst.get_scheduled_time()
                addresses = inst.get_address()
                if isinstance(inst, syscall_magic_state_distillation):
                    mapped_addresses = [
                        f"{inst.get_scheduled_mapped_address(addr)} (->{addr})"
                        for addr in addresses
                    ]
                    addr_str = ", ".join(mapped_addresses)
                    print(f"P{process_id}(t={inst_time}): Syscall {inst_name} qubit {addr_str}")
                else:
                    print(f"P{process_id}(t={inst_time}): Syscall {inst_name}")

            else:
                print("Unknown instruction type.")



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



    def construct_qiskit_circuit_for_backend(self, inst_list):
        """
        Construct a qiskit circuit from the instruction list.
        Also help to visualize the circuit.
        """
        qiskit_circuit = qiskit.QuantumCircuit(self._hardware.get_qubit_num(), self._num_measurement)
        current_measurement = 0
        for inst in inst_list:
            if isinstance(inst, instruction):
                process_id = inst.get_processID()
                if inst.get_type() == Instype.BARRIER:
                    qiskit_circuit.barrier(label=f"P{process_id} barrier")
                    continue

                addresses = inst.get_qubitaddress()
                mapped_addresses = [
                    f"{inst.get_scheduled_mapped_address(addr)} ({addr})"
                    for addr in addresses
                ]
                addr_str = ", ".join(mapped_addresses)
                addr_str= f"P{process_id}:" + addr_str
                match inst.get_type():
                    case Instype.CNOT:
                        qiskit_circuit.cx(inst.get_scheduled_mapped_address(addresses[0]), inst.get_scheduled_mapped_address(addresses[1]), label=addr_str)
                    case Instype.X:
                        qiskit_circuit.x(inst.get_scheduled_mapped_address(addresses[0]), label=f"P{process_id}")                 
                    case Instype.Y:
                        qiskit_circuit.y(inst.get_scheduled_mapped_address(addresses[0])) 
                    case Instype.Z:
                        qiskit_circuit.z(inst.get_scheduled_mapped_address(addresses[0]), label=f"P{process_id}")   
                    case Instype.H: 
                        qiskit_circuit.append(HGate(label=f"H(P{process_id})"),[inst.get_scheduled_mapped_address(addresses[0])])
                        #qiskit_circuit.h(inst.get_scheduled_mapped_address(addresses[0]), label=f"P{process_id}")
                    case Instype.RESET:
                        qiskit_circuit.reset(inst.get_scheduled_mapped_address(addresses[0])) 
                    case Instype.MEASURE:
                        qiskit_circuit.measure(inst.get_scheduled_mapped_address(addresses[0]), current_measurement)
                        current_measurement += 1
                        #Add reset after measurement
                        qiskit_circuit.reset(inst.get_scheduled_mapped_address(addresses[0])) 
                        
                 
        # fig = circuit_drawer(qiskit_circuit, output="mpl", fold=-1) 
        # fig.savefig("my_circuit.png", dpi=300, bbox_inches="tight")
        return qiskit_circuit



    def construct_qiskit_from_instruction_list(self,inst_list):
        """
        Construct a qiskit circuit from the instruction list.
        Also help to visualize the circuit.
        """
        qiskit_circuit = qiskit.QuantumCircuit(self._hardware.get_qubit_num(), self._num_measurement)
        current_measurement = 0
        current_time = 0 
        for inst in inst_list:
            if isinstance(inst, instruction):
                process_id = inst.get_processID()
                inst_name = get_gate_type_name(inst.get_type())
                inst_time = inst.get_scheduled_time()
                addresses = inst.get_qubitaddress()
                mapped_addresses = [
                    f"{inst.get_scheduled_mapped_address(addr)} ({addr})"
                    for addr in addresses
                ]
                addr_str = ", ".join(mapped_addresses)
                addr_str= f"P{process_id}:" + addr_str
                match inst.get_type():
                    case Instype.CNOT:
                        qiskit_circuit.cx(inst.get_scheduled_mapped_address(addresses[0]), inst.get_scheduled_mapped_address(addresses[1]), label=addr_str)
                    case Instype.X:
                        qiskit_circuit.x(inst.get_scheduled_mapped_address(addresses[0]), label=f"P{process_id}")                 
                    case Instype.Y:
                        qiskit_circuit.y(inst.get_scheduled_mapped_address(addresses[0]), label=f"P{process_id}") 
                    case Instype.Z:
                        qiskit_circuit.z(inst.get_scheduled_mapped_address(addresses[0]), label=f"P{process_id}")   
                    case Instype.H: 
                        qiskit_circuit.append(HGate(label=f"H(P{process_id})"),[inst.get_scheduled_mapped_address(addresses[0])])
                        #qiskit_circuit.h(inst.get_scheduled_mapped_address(addresses[0]), label=f"P{process_id}")
                    case Instype.RESET:
                        qiskit_circuit.reset(inst.get_scheduled_mapped_address(addresses[0])) 
                    case Instype.MEASURE:
                        qiskit_circuit.measure(inst.get_scheduled_mapped_address(addresses[0]), current_measurement)
                        qiskit_circuit.barrier(label=f"P{process_id} measure, t={inst_time}")
                        current_measurement += 1
                

            elif isinstance(inst, syscall):
                process_id = inst.get_processID()
                inst_name = get_syscall_type_name(inst)
                inst_time = inst.get_scheduled_time()
                addresses = inst.get_address()
                if isinstance(inst, syscall_magic_state_distillation):
                    mapped_addresses = [
                        f"{inst.get_scheduled_mapped_address(addr)} (->{addr})"
                        for addr in addresses
                    ]
                    qubit_num = len(mapped_addresses)
                    my_gate = Gate(name=f"MAGIC, P{process_id}", num_qubits=qubit_num, params=[])
                    qiskit_circuit.append(my_gate, [inst.get_scheduled_mapped_address(addr) for addr in addresses])
                else:
                    my_gate = Gate(name=f"{inst.get_simple_name()}, P{process_id}", num_qubits=self._qubit_num, params=[])
                    qiskit_circuit.append(my_gate, list(range(self._qubit_num)))

            if inst_time > current_time:
                qiskit_circuit.barrier(label=f"t={inst_time}")     
            current_time = inst_time

        style = {
            "fontsize": 15  # increase/decrease as needed
        }

        fig = circuit_drawer(qiskit_circuit, output="mpl", fold=-1, style=style) 
        fig.savefig("my_circuit.png", dpi=300, bbox_inches="tight")






def simple_example_with_T_gate():
    vdata1 = virtualSpace(size=10, label="vdata1")
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=5, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,4)
    proc1 = process(processID=1,start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1),vsyn1.get_address(2)],size=3,processID=1))  # Allocate 2 syndrome qubits
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])  # CNOT operation

    proc1.add_syscall(syscallinst=syscall_magic_state_distillation(address=[vsyn1.get_address(2),vsyn1.get_address(3)],processID=1))  # Magic state distillation

    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(2)])  # CNOT operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1),vsyn1.get_address(2)],size=3,processID=1))  # Allocate 2 syndrome qubits

    #proc1.construct_qiskit_diagram()

    vdata2 = virtualSpace(size=10, label="vdata2")
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=5, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,4)
    proc2 = process(processID=2,start_time=3, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1),vsyn2.get_address(2)],size=3,processID=2))  # Allocate 2 syndrome qubits
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])  # CNOT operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])  # CNOT operation

    proc2.add_syscall(syscallinst=syscall_magic_state_distillation(address=[vsyn2.get_address(2),vsyn2.get_address(3)],processID=2))  # Magic state distillation

    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(2)])  # CNOT operation
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1),vsyn2.get_address(2)],size=3,processID=2))  # Allocate 2 syndrome qubits

    #proc2.construct_qiskit_diagram()

    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001)

    return kernel_instance, virtual_hardware





def generate_example_ppt():
    vdata1 = virtualSpace(size=3, label="vdata1")
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=2, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,1)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))  # Allocate 2 syndrome qubits
    proc1.add_instruction(Instype.H, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0),vsyn1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0), vsyn1.get_address(1)])  # CNOT operation    
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])  # Measure operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))  # Allocate 2 syndrome qubits
    #proc1.construct_qiskit_diagram()


    vdata2 = virtualSpace(size=3, label="vdata2")
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=2, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,1)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))  # Allocate 2 syndrome qubits
    proc2.add_instruction(Instype.H, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])  # CNOT operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])  # CNOT operation
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0), vsyn2.get_address(1)])  # CNOT operation    
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])  # Measure operation
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))  # Allocate 2 syndrome qubits


    #proc2.construct_qiskit_diagram()

    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain
    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001, edge_list=COUPLING)

    return kernel_instance, virtual_hardware


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
    virtual_hardware = virtualHardware(qubit_number=7, error_rate=0.001)

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

    kernel_instance, virtual_hardware = generate_example_ppt()
    schedule_instance=Scheduler(kernel_instance,virtual_hardware)
    time1, inst_list1=schedule_instance.scheduling_with_out_sharing_syndrome_qubit()
    #time1, inst_list1=schedule_instance.baseline_scheduling()
    schedule_instance.print_dynamic_instruction_list(inst_list1)


    #schedule_instance.construct_qiskit_from_instruction_list(inst_list1)

    print("-------------------------------------------------------------")

    # kernel_instance, virtual_hardware = generate_example()
    # schedule_instance=Scheduler(kernel_instance,virtual_hardware)
    # time2, inst_list2=schedule_instance.schedule()
    # schedule_instance.print_instruction_list(inst_list2)




#    #print("Mapping after scheduling:")

#     #print(mapping)


#     kernel_instance, virtual_hardware = generate_example()
#     schedule_instance=Scheduler(kernel_instance,virtual_hardware)
#     time2, inst_list2=schedule_instance.schedule()


#     print("Baseline: {}".format(time1))


#     print("Our: {}".format(time2))
#     #print(kernel_instance)


#     schedule_instance.print__instruction_list(inst_list2)




    # stim_circuit = mapping.transpile(inst_list)

    # print(stim_circuit)