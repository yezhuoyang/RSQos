#Fault-tolerant process on a fault-tolerant quantum computer

from unicodedata import name
from instruction import *
from virtualSpace import virtualSpace, virtualAddress
from syscall import *
import qiskit
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator  # <- use AerSimulator (Qiskit 2.x)
from qiskit.qasm2 import dumps



class ProcessStatus(Enum):
    WAIT_TO_START = 0
    RUNNING = 1
    WAIT_FOR_ANSILLA = 2
    WAIT_FOR_T_GATE = 3
    FINISHED = 4


def relabel_clbits(instruction_str: str)-> QuantumCircuit:
    """
    Input is a quantum circuit with arbitrary measurement order and 
    arbitrary set of classical register

    We will relabel all qubit register and classical register to 
    a single one dimensional virtual space. 


    Output is a new quantum circuit with only one classical register with increasing order
    """
    circuit = qiskit.qasm2.loads(instruction_str)
    num_clbits=len(circuit.clbits)
    num_qubit=len(circuit.qubits)
    qubit_index_map={circuit.qubits[i]:i for i in range(len(circuit.qubits))}
    clbit_index_map={circuit.clbits[i]:i for i in range(len(circuit.clbits))}
    

    new_circ=QuantumCircuit(num_qubit,num_clbits)
    for inst in circuit:
        name=inst.operation.name
        if name == "h":
            reg=inst.qubits[0]
            new_circ.h(qubit_index_map[reg])
        elif name=='x':
            reg=inst.qubits[0]
            new_circ.x(qubit_index_map[reg])
        elif name == "y":
            reg=inst.qubits[0]
            new_circ.y(qubit_index_map[reg])
        elif name == "z":
            reg=inst.qubits[0]
            new_circ.z(qubit_index_map[reg])
        elif name == "t":
            reg=inst.qubits[0]
            new_circ.t(qubit_index_map[reg])
        elif name == "tdg":
            reg=inst.qubits[0]
            new_circ.tdg(qubit_index_map[reg])
        elif name == "s":
            reg=inst.qubits[0]
            new_circ.s(qubit_index_map[reg])
        elif name == "sdg":
            reg=inst.qubits[0]
            new_circ.sdg(qubit_index_map[reg])
        elif name == "sx":
            reg=inst.qubits[0]
            new_circ.sx(qubit_index_map[reg])
        elif name == "rx":
            reg=inst.qubits[0]
            new_circ.rx(inst.operation.params[0],qubit_index_map[reg])
        elif name == "rz":
            reg=inst.qubits[0]
            new_circ.rz(inst.operation.params[0],qubit_index_map[reg])
        elif name == "ry":
            reg=inst.qubits[0]
            new_circ.ry(inst.operation.params[0],qubit_index_map[reg])
        elif name == "u":
            reg=inst.qubits[0]
            new_circ.u(inst.operation.params[0], inst.operation.params[1], inst.operation.params[2],qubit_index_map[reg])
        elif name == "u3":
            reg=inst.qubits[0]
            new_circ.u(inst.operation.params[0], inst.operation.params[1], inst.operation.params[2],qubit_index_map[reg])
        elif name == "ccx":
            reg0, reg1, reg2 = inst.qubits[0], inst.qubits[1], inst.qubits[2]
            new_circ.ccx(qubit_index_map[reg0], qubit_index_map[reg1], qubit_index_map[reg2])   
        elif name == "cx":
            reg0, reg1 = inst.qubits[0], inst.qubits[1]
            new_circ.cx(qubit_index_map[reg0], qubit_index_map[reg1])
        elif name == "cH":
            reg0, reg1 = inst.qubits[0], inst.qubits[1]
            new_circ.ch(qubit_index_map[reg0], qubit_index_map[reg1])
        elif name == "swap":
            reg0, reg1 = inst.qubits[0], inst.qubits[1]
            new_circ.swap(qubit_index_map[reg0], qubit_index_map[reg1])
        elif name == "cswap":
            reg0, reg1, reg2 = inst.qubits[0], inst.qubits[1], inst.qubits[2]
            new_circ.cswap(qubit_index_map[reg0], qubit_index_map[reg1], qubit_index_map[reg2])
        elif name == "cu1":
            params=inst.operation.params
            reg0, reg1 = inst.qubits[0], inst.qubits[1]
            new_circ.cp(params[0], qubit_index_map[reg0], qubit_index_map[reg1])
        elif name == "reset":
            reg=inst.qubits[0]
            new_circ.reset(qubit_index_map[reg])
        elif name == "measure":
            creg=inst.clbits[0]
            reg=inst.qubits[0]
            new_circ.measure(qubit_index_map[reg], clbit_index_map[creg])
        elif name == "barrier":
            continue
        else:
            raise ValueError(f"Unsupported instruction: {name}")            
        
    return new_circ




class process:

    def __init__(self, processID: int , start_time: int, vdataspace: virtualSpace, vsyndromespace: virtualSpace, shots=1000) -> None:
        self._start_time=start_time
        self._processID = processID
        self._instruction_list = []
        self._syscall_list = []
        self._current_time = start_time
        self._num_measurement = 0
        self._status = ProcessStatus.WAIT_TO_START
        self._shots = shots
        """
        Keep track of the remaining shots to be executed.
        """
        self._remaining_shots = shots  
        """
        The virtual data space, and virtual syndrome qubit space allocated to this process by the OS
        """
        self._vdataspace = vdataspace
        self._vsyndromespace = vsyndromespace
        """
        The number of data qubits, syndrome qubits 
        used in this process
        """
        self._num_data_qubits = 0
        self._num_syndrome_qubits = 0
        """
        Keep tract of the execution of all instruction
        Help the kernel and schedule to estimate the resources
        consumed_qpu_time keep track of how long this process has been executed on the hardware
        """
        self._executed = {}
        self._is_done = False
        self._next_instruction_label = 0
        self._consumed_qpu_time = 0
        self._virtual_data_addresses = []  # List to hold virtual data qubit addresses allocated to this process
        self._virtual_syndrome_addresses = []  # List to hold virtual syndrome qubit addresses allocated to this process        
        """
        The mapping from virtual address to physical address of all data qubits
        This should be fixed throughout the life cycle of the process
        """
        self._data_qubit_virtual_hardware_mapping = {}  # type: dict[virtualAddress, int]
        """
        The mapping from virtual address to physical address of all syndrome qubits
        This mapping can be changed during the life cycle of the process
        """
        self._syndrome_qubit_virtual_hardware_mapping = {}  # type: dict[virtualAddress, int]
        self._syndrome_qubit_is_allocated = {}  # type: dict[virtualAddress, bool]
        """
        Qiskit circuit representation of the process
        """
        self._qiskit_circuit = None
        """
        Connectivity analysis and mapping cost between syndrome qubit and data qubits
        """
        self._connectivity=None
        self._syn_mapping_cost=None
        """
        The connectivity analysis and mapping cost between data qubits
        e.g.  From data qubit A to data qubit B, how many CNOT gates are there in the process
        """
        self._data_qubit_inner_connectivity=None
        self._data_mapping_cost=None
        """
        The mapping from the measurement index to measured 

        For example, the real circuit might look like:

        c3 = MEASURE q5
        c2 = MEASURE q1
        c1 = MEASURE q0

        However, the measurement index should be 0,1,2,... in the order of measurement.
        After we get the measurement result according to the measurement index,
        we should map it back to the classical bit index used in the process.


        Example:
        self._process_measure_index = {
            0: 3,
            1: 2,
            2: 1
        }
        """
        self._process_measure_index = {}  # type: dict[int, int]




    def reorder_bit_string(self,bitstring:str)-> str:
        """
        Given the measurement bitstring according to the measurement index,
        Return the measurement bitstring according to the classical bit index used in the process.


        For example, 
        self._process_measure_index = {
            0: 1,
            1: 2,
            2: 0
        }

        And the measured bitstring is(Use Little Endian encoding):

        110

        The actual classical bits should be:

        bit index:  0 2 1

        So the actual reordered bitstring should be:

        101

        """
        reordered_bitstring = ['0'] * len(bitstring)
        for meas_index, clbit_index in self._process_measure_index.items():
            reordered_bitstring[clbit_index] = bitstring[len(bitstring) - 1 - meas_index]
        return ''.join(reordered_bitstring)



    def reorder_count(self,process_counts:dict)-> dict:
        """
        Given the measurement counts according to the measurement index,
        Return the measurement counts according to the classical bit index used in the process.
        """
        reordered_counts = {}
        for bitstring, count in process_counts.items():
            reorder_bitstring = self.reorder_bit_string(bitstring)
            reordered_counts[reorder_bitstring] = count
        return reordered_counts





    def process_str(self)-> str:
        """
        Print the standard representation of the process. Example:

        q = alloc_data(3)
        s = alloc_helper(3)
        set_shot(1000)
        H q2
        CNOT q1, q2
        CNOT q0, s0
        CNOT q1, s1
        X q1
        CNOT q1, s2
        c0 = MEASURE s0
        c1 = MEASURE s1
        c2 = MEASURE s2
        deallocate_data(q)
        deallocate_helper(s)    

        """
        outputstr = ""
        outputstr+= f"q = alloc_data({self._num_data_qubits})\n"
        outputstr+= f"s = alloc_helper({self._num_syndrome_qubits})\n"
        outputstr += f"set_shot({self._shots})\n"


        for inst in self._instruction_list:
            if not isinstance(inst, instruction):
                continue
            type = inst.get_type()
            virindex=inst.get_virtual_address_indices()
            if type == Instype.H:
                outputstr+= f"H {virindex[0]}\n"
            elif type == Instype.X:
                outputstr+= f"X {virindex[0]}\n"
            elif type == Instype.Y:
                outputstr+= f"Y {virindex[0]}\n"
            elif type == Instype.Z:
                outputstr+= f"Z {virindex[0]}\n"
            elif type == Instype.T:
                outputstr+= f"T {virindex[0]}\n"
            elif type == Instype.Tdg:
                outputstr+= f"Tdg {virindex[0]}\n"
            elif type == Instype.S:
                outputstr+= f"S {virindex[0]}\n"
            elif type == Instype.Sdg:
                outputstr+= f"Sdg {virindex[0]}\n"
            elif type == Instype.SX:
                outputstr+= f"SX {virindex[0]}\n"
            elif type == Instype.RZ:
                params=inst.get_params()
                outputstr+= f"RZ({params[0]}) {virindex[0]}\n"
            elif type == Instype.RX:
                params=inst.get_params()
                outputstr+= f"RX({params[0]}) {virindex[0]}\n"
            elif type == Instype.RY:
                params=inst.get_params()
                outputstr+= f"RY({params[0]}) {virindex[0]}\n"
            elif type == Instype.U:
                params=inst.get_params()
                outputstr+= f"U({params[0]}, {params[1]}, {params[2]}) {virindex[0]}\n"
            elif type == Instype.U3:
                params=inst.get_params()
                outputstr+= f"U3({params[0]}, {params[1]}, {params[2]}) {virindex[0]}\n"
            elif type == Instype.Toffoli:
                outputstr+= f"Toffoli {virindex[0]}, {virindex[1]}, {virindex[2]}\n"
            elif type == Instype.CNOT:
                outputstr+= f"CNOT {virindex[0]}, {virindex[1]}\n"
            elif type == Instype.CH:
                outputstr+= f"CH {virindex[0]}, {virindex[1]}\n"
            elif type == Instype.SWAP:
                outputstr+= f"SWAP {virindex[0]}, {virindex[1]}\n"
            elif type == Instype.CSWAP:
                outputstr+= f"CSWAP {virindex[0]}, {virindex[1]}, {virindex[2]}\n"
            elif type == Instype.CP:
                params=inst.get_params()
                outputstr+= f"CP({params[0]}) {virindex[0]}, {virindex[1]}\n"
            elif type == Instype.RESET:
                outputstr+= f"RESET {virindex[0]}\n"
            elif type == Instype.MEASURE:
                classical_address=inst.get_classical_address()
                outputstr+= f"c{classical_address} = MEASURE {virindex[0]}\n"
            else:
                raise ValueError(f"Unsupported instruction")
            
        outputstr+= f"deallocate_data(q)\n"
        outputstr+= f"deallocate_helper(s)\n"
        return outputstr




    def get_total_shots(self):
        return self._shots


    def finish_shot(self):
        """
        Return true if all the shots of this process has been consumed
        """
        return self._remaining_shots == 0


    def consume_shot(self, shot_num: int):
        self._remaining_shots -= int(shot_num)


    def get_remaining_shots(self) -> int:
        return self._remaining_shots


    def reset_mapping(self):
        """
        Reset all mappings in the process.
        But notice that the remaining shots and consumed QPU time are not reset.
        """
        self._status = ProcessStatus.WAIT_TO_START
        self._executed = {}
        self._is_done = False
        self._next_instruction_label = 0
        self._consumed_qpu_time = 0
        self._data_qubit_virtual_hardware_mapping = {}  # type: dict[virtualAddress, int]
        self._syndrome_qubit_virtual_hardware_mapping = {}  # type: dict[virtualAddress, int]
        self._syndrome_qubit_is_allocated = {}  # type: dict[virtualAddress, bool]
        for inst in self._instruction_list:
            if isinstance(inst, instruction):
                inst.reset_mapping()




    def analyze_data_qubit_connectivity(self):
        """
        This is used to help setting up the initial mapping of data qubits.
        """
        self._data_qubit_inner_connectivity = {addr: {daddr:0 for daddr in  self._virtual_data_addresses} for addr in self._virtual_data_addresses}
        for inst in self._instruction_list:
            if isinstance(inst, instruction) and inst.get_type() == Instype.CNOT:
                qubit_addresses = inst.get_qubitaddress()
                control, target = qubit_addresses
                if not control.is_syndrome() and not target.is_syndrome():
                     self._data_qubit_inner_connectivity[control][target] += 1
                     self._data_qubit_inner_connectivity[target][control] += 1
        return self._data_qubit_inner_connectivity


    def calc_data_mapping_cost(self,distance_matrix,num_physical_qubits):
        """
        TODO: Calculate the mapping cost for all data qubits in the process.
        We will use a greedy algorithm to find a good initial mapping of data qubits.
        """
        self._data_mapping_cost = {addr: {} for addr in self._virtual_data_addresses}
        for dataaddr in self._virtual_data_addresses:
            for physical_qubit in range(num_physical_qubits):
                cost = self.data_mapping_cost(dataaddr, physical_qubit, distance_matrix)
                self._data_mapping_cost[dataaddr][physical_qubit] = cost
        return self._data_mapping_cost


    def analyze_syndrome_connectivity(self):
        """
        Analyze the connectivity of syndrome qubits in the process.
        This function count the number of CNOT gates between all syndrome qubits and data qubits.
        The return value is a dictionary, the key is the data qubit virtual address,the value us the number of CNOT gates.
        """
        self._connectivity = {addr: {daddr:0 for daddr in  self._virtual_data_addresses} for addr in self._virtual_syndrome_addresses}

        for inst in self._instruction_list:
            if isinstance(inst, instruction) and inst.get_type() == Instype.CNOT:
                qubit_addresses = inst.get_qubitaddress()
                control, target = qubit_addresses
                if control.is_syndrome() and not target.is_syndrome():
                     self._connectivity[control][target] += 1
                elif target.is_syndrome() and not control.is_syndrome():
                     self._connectivity[target][control] += 1

        return  self._connectivity


    def calc_syn_mapping_cost(self,distance_matrix,num_physical_qubits):
        """
        Calculate the mapping cost for all syndrome qubits in the process.
        The mapping cost is defined as the sum of the distances between the physical qubit mapped to the syndrome qubit and the physical qubits mapped to the data qubits it interacts with.
        The distance is weighted by the number of CNOT gates between the syndrome qubit and each data qubit.
        The return value is a dictionary, the key is the syndrome qubit virtual address, the value is another dictionary, whose key is the physical qubit address, and value is the mapping cost.
        """
        self._syn_mapping_cost = {addr: {} for addr in self._virtual_syndrome_addresses}
        for synaddr in self._virtual_syndrome_addresses:
            for physical_qubit in range(num_physical_qubits):
                cost = self.syn_mapping_cost(synaddr, physical_qubit, distance_matrix)
                self._syn_mapping_cost[synaddr][physical_qubit] = cost
        return self._syn_mapping_cost



    def data_mapping_cost(self, dataaddr:virtualAddress, selected_qubit: int ,distance_matrix):
        """
        Return the cost if we map a data qubit to selected_qubit
        We assume here that all other data qubits have already been mapped to physical qubits
        Some data qubits may not have been mapped yet, we skip them in the cost calculation
        This is used for greedy initial mapping of data qubits
        """
        cost = 0
        for data_addr in self._virtual_data_addresses:
            if data_addr == dataaddr:
                continue
            physical_data_qubit = self.get_data_qubit_virtual_hardware_mapping(data_addr)
            if physical_data_qubit == -1:
                continue
            num_cnot = self._data_qubit_inner_connectivity[dataaddr][data_addr]
            cost += num_cnot * distance_matrix[selected_qubit][physical_data_qubit]
        return cost


    def syn_mapping_cost(self,synaddr:virtualAddress,selected_qubit: int ,distance_matrix):
        """
        Return the cost if we map a syndrome qubit to selected_qubit
        We assume here that all data qubits have already been mapped to physical qubits
        """
        cost = 0
        for data_addr in self._virtual_data_addresses:
            physical_data_qubit = self.get_data_qubit_virtual_hardware_mapping(data_addr)
            num_cnot = self._connectivity[synaddr][data_addr]
            cost += num_cnot * distance_matrix[selected_qubit][physical_data_qubit]
        return cost


    def ranked_syn_mapping_cost(self,synaddr:virtualAddress, current_avalible:dict[int,bool]):
        """
        Return the ranked mapping cost for a given syndrome qubit.
        """
        result = sorted(
                ((addr, cost) for addr, cost in self._syn_mapping_cost[synaddr].items() if current_avalible[addr]), 
                key=lambda item: item[1]
                )
        return result


    def ranked_data_mapping_cost(self,dataaddr:virtualAddress, current_avalible:dict[int,bool]):
        """
        Return the ranked mapping cost for a given data qubit.
        Used in the initial mapping of data qubits.
        """
        result = sorted(
                ((addr, cost) for addr, cost in self._data_mapping_cost[dataaddr].items() if current_avalible[addr]), 
                key=lambda item: item[1]
                )
        return result



    def set_status(self, status: ProcessStatus):
        self._status = status


    def get_status(self) -> ProcessStatus:
        return self._status

    def get_processID(self) -> int:
        return self._processID


    def empty_syndrome_qubit_mappings(self,addr):
        """
        Empty the mapping from virtual address to physical address for a given syndrome qubit.
        This is used for dynamic allocation and deallocation of syndrome qubits.
        
         Args:
            qubitaddress (virtualAddress): The virtual address of the qubit.
        """
        if not self.syndrome_qubit_is_allocated(addr):
            raise ValueError("The requested syndrome qubit is not allocated.")
        self._syndrome_qubit_virtual_hardware_mapping[addr] = -1
        self._syndrome_qubit_is_allocated[addr] = False


    def syndrome_qubit_is_allocated(self, qubitaddress: virtualAddress) -> bool:
        """
        Check if a given syndrome qubit (by its virtual address) is currently allocated.
        
        Returns:
            bool: True if the syndrome qubit is allocated, False otherwise.
        """
        return self._syndrome_qubit_is_allocated.get(qubitaddress)


    def get_syndrome_qubit_virtual_hardware_mapping(self,qubitaddress: virtualAddress) -> int:
        """
        Get the mapping from virtual address to physical address for syndrome qubits.
        
        Returns:
            int: The physical address mapped to the given virtual address.
        """
        if not self.syndrome_qubit_is_allocated(qubitaddress):
            raise ValueError("The requested syndrome qubit is not allocated.")
        return self._syndrome_qubit_virtual_hardware_mapping[qubitaddress]


    def set_syndrome_qubit_virtual_hardware_mapping(self, qubitaddress: virtualAddress, physicaladdress: int):
        """
        Set the mapping from virtual address to physical address for syndrome qubits.
        
         Args:
            qubitaddress (virtualAddress): The virtual address of the qubit.
            physicaladdress (int): The physical address to which the qubit is mapped.
        """
        self._syndrome_qubit_virtual_hardware_mapping[qubitaddress] = physicaladdress
        self._syndrome_qubit_is_allocated[qubitaddress] = True


    def set_data_qubit_virtual_hardware_mapping(self, qubitaddress: virtualAddress, physicaladdress: int):
        """
        Set the mapping from virtual address to physical address for data qubits.
        
         Args:
            qubitaddress (virtualAddress): The virtual address of the qubit.
            physicaladdress (int): The physical address to which the qubit is mapped.
        """
        self._data_qubit_virtual_hardware_mapping[qubitaddress] = physicaladdress


    def get_data_qubit_virtual_hardware_mapping(self,qubitaddress: virtualAddress) -> int:
        """
        Get the mapping from virtual address to physical address for data qubits.
        
        Returns:
            int: The physical address mapped to the given virtual address.
            If the data qubit is not allocated, return -1.
        """
        return self._data_qubit_virtual_hardware_mapping.get(qubitaddress, -1)


    def is_data_qubit(self, qubitaddress: virtualAddress) -> bool:
        """
        Check if a given virtual address is a data qubit.
        
        Returns:
            bool: True if the virtual address is a data qubit, False otherwise.
        """
        return qubitaddress in self._virtual_data_addresses

    def is_syndrome_qubit(self, qubitaddress: virtualAddress) -> bool:
        """
        Check if a given virtual address is a syndrome qubit.
        
        Returns:
            bool: True if the virtual address is a syndrome qubit, False otherwise.
        """
        return qubitaddress in self._virtual_syndrome_addresses


    def parse_from_stim_program(self, stim_program):
        """
        Parse a stim program and add instructions to the process.
        This is a placeholder for future implementation.
        """
        raise NotImplementedError("This method needs to be implemented for parsing stim programs.")


    def parse_from_qasm_program(self, qasm_program):
        """
        Parse a QASM program and add instructions to the process.
        This is a placeholder for future implementation.
        """
        raise NotImplementedError("This method needs to be implemented for parsing QASM programs.")


    def get_start_time(self) -> int:
        return self._start_time

    def get_num_data_qubits(self) -> int:
        """
        Get the number of data qubits used by the process.
        """
        return self._num_data_qubits

    def get_num_syndrome_qubits(self) -> int:
        """
        Get the number of syndrome qubits used by the process.
        """
        return self._num_syndrome_qubits


    def add_instruction(self, type:Instype, qubitaddress:List[virtualAddress],classical_address: int=None, params: List[float]=None):
        """
        Add an instruction to the process.
        """
        time = self._current_time
        self._current_time += get_clocktime(type)  # Increment time for the next instruction 

        if type == Instype.H:
            inst = instruction(type=Instype.H, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.X:
            inst = instruction(type=Instype.X, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.Y:
            inst = instruction(type=Instype.Y, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.Z:
            inst = instruction(type=Instype.Z, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.T:
            inst = instruction(type=Instype.T, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.Tdg:
            inst = instruction(type=Instype.Tdg, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.S:
            inst = instruction(type=Instype.S, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.Sdg:
            inst = instruction(type=Instype.Sdg, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.SX:
            inst = instruction(type=Instype.SX, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.RZ:
            inst = instruction(type=Instype.RZ, qubitaddress=qubitaddress, processID=self._processID, time=0, params=params)
        elif type == Instype.RX:
            inst = instruction(type=Instype.RX, qubitaddress=qubitaddress, processID=self._processID, time=0, params=params)
        elif type == Instype.RY:
            inst = instruction(type=Instype.RY, qubitaddress=qubitaddress, processID=self._processID, time=0, params=params)
        elif type == Instype.U:
            inst = instruction(type=Instype.U, qubitaddress=qubitaddress, processID=self._processID, time=0, params=params)
        elif type == Instype.U3:
            inst = instruction(type=Instype.U3, qubitaddress=qubitaddress, processID=self._processID, time=0, params=params)
        elif type == Instype.Toffoli:
            inst = instruction(type=Instype.Toffoli, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.CNOT:
            inst = instruction(type=Instype.CNOT, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.CH:
            inst = instruction(type=Instype.CH, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.SWAP:
            inst = instruction(type=Instype.SWAP, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.CSWAP:
            inst = instruction(type=Instype.CSWAP, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.CP:
            inst = instruction(type=Instype.CP, qubitaddress=qubitaddress, processID=self._processID, time=0, params=params)
        elif type == Instype.RESET:
            inst = instruction(type=Instype.RESET, qubitaddress=qubitaddress, processID=self._processID, time=0)
        elif type == Instype.MEASURE:
            """
            Be careful also need to change the measurement index mapping in the process
            """
            inst = instruction(type=Instype.MEASURE, qubitaddress=qubitaddress, processID=self._processID, time=0, classical_address=classical_address)
            self._process_measure_index[self._num_measurement] = classical_address
        else:
            raise ValueError(f"Unsupported instruction")


        #inst=instruction(type, qubitaddress, self._processID,time)
        self._instruction_list.append(inst)
        self._executed[inst] = False
        if type == Instype.MEASURE:
            self._num_measurement += 1
        for addr in qubitaddress:
            if addr.is_syndrome():
                if addr not in self._virtual_syndrome_addresses:
                    self._virtual_syndrome_addresses.append(addr)
                if addr not in self._syndrome_qubit_is_allocated:
                    self._syndrome_qubit_is_allocated[addr] = False
            else:
                if addr not in self._virtual_data_addresses:
                    self._virtual_data_addresses.append(addr)


    def get_instruction_list(self):
        return self._instruction_list


    def get_virtual_data_addresses(self):
        return self._virtual_data_addresses

    def get_virtual_syndrome_addresses(self):
        return self._virtual_syndrome_addresses

    def add_syscall(self, syscallinst: syscall):
        """
        Add a syscall to the process.
        The process must have allocated the necessary resources first.
        """
        if isinstance(syscallinst, syscall):
            self._syscall_list.append(syscallinst)
            self._instruction_list.append(syscallinst)
            if isinstance(syscallinst, syscall_allocate_data_qubits):
                if syscallinst._size > self._vdataspace.get_size():
                    raise ValueError("Requested size exceeds virtual data space size.")
                self._vdataspace.allocate_range(0, syscallinst._size - 1)
                self._num_data_qubits += syscallinst._size
            elif isinstance(syscallinst, syscall_allocate_syndrome_qubits):
                if syscallinst._size > self._vsyndromespace.get_size():
                    raise ValueError("Requested size exceeds virtual syndrome space size.")
                self._vsyndromespace.allocate_range(0, syscallinst._size - 1)
                self._num_syndrome_qubits += syscallinst._size
            elif isinstance(syscallinst, syscall_deallocate_data_qubits):
                self._vdataspace.free_range(0, syscallinst._size - 1)
            elif isinstance(syscallinst, syscall_deallocate_syndrome_qubits):
                self._vsyndromespace.free_range(0, syscallinst._size - 1)
            elif isinstance(syscallinst, syscall_magic_state_distillation):
                for addr in syscallinst.get_address():
                    if self._num_syndrome_qubits<addr.get_index()+1:
                        self._num_syndrome_qubits=addr.get_index()+1
        else:
            raise TypeError("Expected a syscall instance.")


    def __str__(self):
        outputstr = f"Process ID: {self._processID}\n"
        outputstr += "Instructions:\n"
        for inst in self._instruction_list:
            if isinstance(inst, instruction):
                outputstr += str(inst) + "\n"
            elif isinstance(inst, syscall):
                outputstr += f"Syscall: {inst}\n"
        return outputstr


    def is_done(self) -> bool:
        """
        Check if the process has completed all instructions.
        """
        return self._is_done



    def get_next_instruction(self) -> instruction:
        """
        Get the next instruction to be executed in the process.
        """
        if self._next_instruction_label < len(self._instruction_list):
            return self._instruction_list[self._next_instruction_label]
        else:
            return None


    def get_addresses_of_next_instruction(self) -> List[virtualAddress]:
        """
        Get the qubit addresses of the next instruction to be executed.
        """
        if self._next_instruction_label < len(self._instruction_list):
            inst = self._instruction_list[self._next_instruction_label]
            return inst.get_qubitaddress()
        else:
            return []


    def execute_instruction(self, hardwaretime=0) -> instruction:
        """
        Virtually execute the next instruction in the process.
        This method help the scheduling algorithm to make decisions.
        Meanwhile, it also updates the scheduled hardware time.
        """
        if self._next_instruction_label < len(self._instruction_list):
            if self._next_instruction_label == len(self._instruction_list)-1:
                self._is_done = True
                self._status = ProcessStatus.FINISHED
            inst = self._instruction_list[self._next_instruction_label]
            self._executed[inst] = True
            self._next_instruction_label += 1
            if isinstance(inst, syscall):
                inst.set_scheduled_time(hardwaretime)
                self._consumed_qpu_time += get_syscall_time(inst)
                return inst
            elif isinstance(inst, instruction):
                inst.set_scheduled_time(hardwaretime)
                self._consumed_qpu_time += get_clocktime(inst.get_type())
                return inst
        return None

    def get_consumed_qpu_time(self) -> int:
        return self._consumed_qpu_time




    def construct_qiskit_circuit(self, add_syscall_gates=False) -> QuantumCircuit:
        """
        Construct a qiskit circuit from the instruction list.
        Also help to visualize the circuit.
        """
        dataqubit = QuantumRegister(self._num_data_qubits, self._vdataspace.get_label())

        # Second part: 2 qubits named 's'
        syndromequbit = QuantumRegister(self._num_syndrome_qubits,self._vsyndromespace.get_label())

        # Classical registers (optional, if you want measurements)
        classicalbits = ClassicalRegister(self._num_measurement, "c")

        # Combine them into one circuit
        qiskit_circuit = QuantumCircuit(dataqubit,  syndromequbit, classicalbits)

        for inst in self._instruction_list:
            if isinstance(inst, instruction):
                inst_name = get_gate_type_name(inst.get_type())
                addresses = inst.get_qubitaddress()
                qiskitaddress=[]
                for addr in addresses:
                    if addr.is_syndrome():
                        qiskitaddress.append(syndromequbit[addr.get_index()])
                    else:
                        qiskitaddress.append(dataqubit[addr.get_index()])
                match inst.get_type():
                    case Instype.H:
                        qiskit_circuit.h(qiskitaddress[0])
                    case Instype.X:
                        qiskit_circuit.x(qiskitaddress[0]) 
                    case Instype.Y:
                        qiskit_circuit.y(qiskitaddress[0]) 
                    case Instype.Z:
                        qiskit_circuit.z(qiskitaddress[0])  
                    case Instype.T:
                        qiskit_circuit.t(qiskitaddress[0])
                    case Instype.Tdg:
                        qiskit_circuit.tdg(qiskitaddress[0])
                    case Instype.S:
                        qiskit_circuit.s(qiskitaddress[0])
                    case Instype.Sdg:
                        qiskit_circuit.sdg(qiskitaddress[0])
                    case Instype.SX:
                        qiskit_circuit.sx(qiskitaddress[0])
                    case Instype.RZ:
                        params=inst.get_params()
                        qiskit_circuit.rz(params[0], qiskitaddress[0])
                    case Instype.RX:
                        params=inst.get_params()
                        qiskit_circuit.rx(params[0], qiskitaddress[0])
                    case Instype.RY:
                        params=inst.get_params()
                        qiskit_circuit.ry(params[0], qiskitaddress[0])
                    case Instype.U3:
                        params=inst.get_params()
                        qiskit_circuit.u3(params[0], params[1], params[2], qiskitaddress[0])
                    case Instype.U:
                        params=inst.get_params()
                        qiskit_circuit.u(params[0], params[1], params[2], qiskitaddress[0])
                    case Instype.Toffoli:
                        qiskit_circuit.ccx(qiskitaddress[0], qiskitaddress[1], qiskitaddress[2])
                    case Instype.CNOT:
                        qiskit_circuit.cx(qiskitaddress[0], qiskitaddress[1])
                    case Instype.CH:
                        qiskit_circuit.ch(qiskitaddress[0], qiskitaddress[1])
                    case Instype.SWAP:
                        qiskit_circuit.swap(qiskitaddress[0], qiskitaddress[1])
                    case Instype.CSWAP:
                        qiskit_circuit.cswap(qiskitaddress[0], qiskitaddress[1], qiskitaddress[2])
                    case Instype.CP:
                        params=inst.get_params()
                        qiskit_circuit.cp(params[0], qiskitaddress[0], qiskitaddress[1])
                    case Instype.RESET:
                        qiskit_circuit.reset(qiskitaddress[0])
                    case Instype.MEASURE:
                        classical_address=inst.get_classical_address()
                        qiskit_circuit.measure(qiskitaddress[0], classical_address)

            elif isinstance(inst, syscall):
                continue

        self._qiskit_circuit = qiskit_circuit
        return qiskit_circuit


    def simulate_circuit(self,shots=1000):
        # Choose simulator and transpile for it
        sim = AerSimulator()
        tqc = transpile(self._qiskit_circuit, sim)

        # Run with 1000 shots
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts(tqc)
        #print(counts)
        return counts



    def construct_qiskit_diagram(self):
        style = {
            "fontsize": 15  # increase/decrease as needed
        }
        fig = circuit_drawer(self._qiskit_circuit, output="mpl", fold=-1, style=style) 
        fig.savefig(f"circuit_P{self._processID}.png", dpi=300, bbox_inches="tight")




def parse_qasm_instruction(shots: int,process_ID: int,instruction_str: str) -> process:
    """
    Parse a QASM instruction string and return an Instruction object.
    
    Args:
        instruction_str (str): The QASM instruction string.
    """

    circuit = relabel_clbits(instruction_str)
    qubit_number = circuit.num_qubits

    vdata = virtualSpace(size=qubit_number, label="vdata")    
    vdata.allocate_range(0, qubit_number - 1)
    vsyn = virtualSpace(size=0, label="vsyn", is_syndrome=True)
    proc = process(processID=process_ID, start_time=0, vdataspace=vdata, vsyndromespace=vsyn, shots=shots)
    proc.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata.get_address(0)],size=qubit_number,processID=process_ID))  # Allocate 2 data qubits
    proc.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=None,size=0,processID=process_ID))  # Allocate 1 syndrome qubit


    for instr, qargs, cargs in circuit.data:
        name = instr.name.lower()
        # Map QASM instruction names to Instruction types
        #print(instr)
        if name == "h":
            proc.add_instruction(Instype.H, [vdata.get_address(qargs[0]._index)])
        elif name == "x":
            proc.add_instruction(Instype.X, [vdata.get_address(qargs[0]._index)])
        elif name == "y":
            proc.add_instruction(Instype.Y, [vdata.get_address(qargs[0]._index)])
        elif name == "z":
            proc.add_instruction(Instype.Z, [vdata.get_address(qargs[0]._index)])
        elif name == "t":
            proc.add_instruction(Instype.T, [vdata.get_address(qargs[0]._index)])
        elif name == "tdg":
            proc.add_instruction(Instype.Tdg, [vdata.get_address(qargs[0]._index)])
        elif name == "s":
            proc.add_instruction(Instype.S, [vdata.get_address(qargs[0]._index)])
        elif name == "sdg":
            proc.add_instruction(Instype.Sdg, [vdata.get_address(qargs[0]._index)])
        elif name == "sx":
            proc.add_instruction(Instype.SX, [vdata.get_address(qargs[0]._index)])
        elif name == "rz":
            proc.add_instruction(Instype.RZ, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "rx":
            #print(instr)
            proc.add_instruction(Instype.RX, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "ry":
            proc.add_instruction(Instype.RY, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "u":
            proc.add_instruction(Instype.U, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "u3":
            proc.add_instruction(Instype.U3, [vdata.get_address(qargs[0]._index)], params=instr.params)
        elif name == "ccx":
            proc.add_instruction(Instype.Toffoli, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index), vdata.get_address(qargs[2]._index)])
        elif name == "cx":
            proc.add_instruction(Instype.CNOT, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)])
        elif name == "ch":
            proc.add_instruction(Instype.CH, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)])
        elif name == "swap":
            proc.add_instruction(Instype.SWAP, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)])
        elif name == "cswap":
            proc.add_instruction(Instype.CSWAP, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index), vdata.get_address(qargs[2]._index)])
        elif name == "cp":
            proc.add_instruction(Instype.CP, [vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)], params=instr.params)
        elif name == "reset":
            proc.add_instruction(Instype.RESET, [vdata.get_address(qargs[0]._index)])
        elif name == "measure":
            proc.add_instruction(Instype.MEASURE, [vdata.get_address(qargs[0]._index)], classical_address=cargs[0]._index)
        elif name == "barrier":
            continue
        else:
            raise ValueError(f"Unsupported instruction: {name}")

    proc.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata.get_address(0)],size=qubit_number ,processID=process_ID))  # Allocate 2 data qubits
    proc.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=None,size=0,processID=process_ID))  # Allocate 2 syndrome qubits

    return proc



label_name_map = {
    1: "adder_n4",
    2: "basis_trotter_n4",
    3: "bb84_n8",
    4: "bell_n4",
    5: "cat_state_n4",
    6: "deutsch_n2",
    7: "dnn_n2",
    8: "dnn_n8",
    9: "error_correctiond3_n5",
    10: "fredkin_n3",
    11: "grover_n2",
    12: "hs4_n4",
    13: "ising_n10",
    14: "iswap_n2",
    15: "lpn_n5",
    16: "qaoa_n3",
    17: "qaoa_n6",
    18: "qec_en_n5",
    19: "qft_n4",
    20: "qrng_n4",
    21: "simon_n6",
    22: "teleportation_n3",
    23: "toffoli_n3",
    24: "vqe_n4",
    25: "wstate_n3"
}

if __name__ == "__main__":

    label_name = label_name_map[1]

    for label_name in label_name_map.values():
        file_path = f"C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\smallqasm\\{label_name}.qasm"

        with open(file_path, "r") as file:
            qasm_code = file.read()  
            proc=parse_qasm_instruction(shots=1000, process_ID=1, instruction_str=qasm_code)

            circ=proc.construct_qiskit_circuit(add_syscall_gates=False)
            print(circ)

