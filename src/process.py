#Fault-tolerant process on a fault-tolerant quantum computer

from instruction import *
from virtualSpace import virtualSpace, virtualAddress
from syscall import *
import qiskit
from qiskit.circuit import Gate
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator  # <- use AerSimulator (Qiskit 2.x)



class ProcessStatus(Enum):
    WAIT_TO_START = 0
    RUNNING = 1
    WAIT_FOR_ANSILLA = 2
    WAIT_FOR_T_GATE = 3
    FINISHED = 4




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
        Connectivity analysis and mapping cost
        """
        self._connectivity=None
        self._mapping_cost=None


    def analyze_data_qubit_connectivity(self):
        """
        TODO: Analyze the connectivity between data qubits in the process.
        This is used to help setting up the initial mapping of data qubits.
        """
        pass

    def calc_data_qubit_mapping_cost(self,distance_matrix,num_physical_qubits):
        """
        TODO: Calculate the mapping cost for all data qubits in the process.
        We will use a greedy algorithm to find a good initial mapping of data qubits.
        """
        pass


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


    def calc_mapping_cost(self,distance_matrix,num_physical_qubits):
        """
        Calculate the mapping cost for all syndrome qubits in the process.
        The mapping cost is defined as the sum of the distances between the physical qubit mapped to the syndrome qubit and the physical qubits mapped to the data qubits it interacts with.
        The distance is weighted by the number of CNOT gates between the syndrome qubit and each data qubit.
        The return value is a dictionary, the key is the syndrome qubit virtual address, the value is another dictionary, whose key is the physical qubit address, and value is the mapping cost.
        """
        self._mapping_cost = {addr: {} for addr in self._virtual_syndrome_addresses}
        for synaddr in self._virtual_syndrome_addresses:
            for physical_qubit in range(num_physical_qubits):
                cost = self.mapping_cost(synaddr, physical_qubit, distance_matrix)
                self._mapping_cost[synaddr][physical_qubit] = cost
        return self._mapping_cost



    def mapping_cost(self,synaddr:virtualAddress,selected_qubit: int ,distance_matrix):
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


    def ranked_mapping_cost(self,synaddr:virtualAddress, current_avalible:dict[int,bool]):
        """
        Return the ranked mapping cost for a given syndrome qubit.
        """
        result = sorted(
                ((addr, cost) for addr, cost in self._mapping_cost[synaddr].items() if current_avalible[addr]), 
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
        """
        return self._data_qubit_virtual_hardware_mapping[qubitaddress] 


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


    def add_instruction(self, type:Instype, qubitaddress:List[virtualAddress]):
        """
        Add an instruction to the process.
        """
        time = self._current_time
        self._current_time += get_clocktime(type)  # Increment time for the next instruction 
        inst=instruction(type, qubitaddress, self._processID,time)
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


        current_measurement = 0
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
                    case Instype.CNOT:
                        qiskit_circuit.cx(qiskitaddress[0], qiskitaddress[1])
                    case Instype.X:
                        qiskit_circuit.x(qiskitaddress[0])                
                    case Instype.Y:
                        qiskit_circuit.y(qiskitaddress[0]) 
                    case Instype.Z:
                        qiskit_circuit.z(qiskitaddress[0])   
                    case Instype.H: 
                        qiskit_circuit.h(qiskitaddress[0])
                    case Instype.RESET:
                        qiskit_circuit.reset(qiskitaddress[0]) 
                    case Instype.MEASURE:
                        qiskit_circuit.measure(qiskitaddress[0], current_measurement)
                        current_measurement += 1
                        #Add reset after measurement
                        qiskit_circuit.reset(qiskitaddress[0])

            elif isinstance(inst, syscall):
                if not add_syscall_gates:
                    continue
                if isinstance(inst, syscall_magic_state_distillation):
                    addresses = inst.get_address()
                    qiskitaddress=[]
                    for addr in addresses:
                        print(addr)
                        
                        qiskitaddress.append(syndromequbit[addr.get_index()])
                    qubit_num = len(addresses)
                    my_gate = Gate(name=f"MAGIC", num_qubits=qubit_num, params=[])
                    qiskit_circuit.append(my_gate, qiskitaddress)
                else:
                    qubit_num = self._num_data_qubits + self._num_syndrome_qubits
                    my_gate = Gate(name=f"{inst.get_simple_name()}, P{self._processID}", num_qubits=qubit_num, params=[])
                    qiskit_circuit.append(my_gate, list(range(qubit_num)))
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


if __name__ == "__main__":

    vdata1 = virtualSpace(size=10, label="vdata1")
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=5, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,4)
    proc1 = process(processID=1,start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1),vsyn1.get_address(2)],size=3,processID=1))  # Allocate 2 syndrome qubits
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    proc1.add_syscall(syscallinst=syscall_magic_state_distillation(address=[vsyn1.get_address(2),vsyn1.get_address(3)],processID=1))  # Magic state distillation

    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(2)])  # CNOT operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1),vsyn1.get_address(2)],size=3,processID=1))  # Allocate 2 syndrome qubits


    proc1.analyze_syndrome_connectivity()
    print(proc1._connectivity)


    #proc1.construct_qiskit_diagram()

    #proc1.construct_qiskit_circuit()
    #proc1.simulate_circuit()