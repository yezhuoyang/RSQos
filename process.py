#Fault-tolerant process on a fault-tolerant quantum computer

from instruction import *
from virtualSpace import virtualSpace, virtualAddress
from syscall import *


class ProcessStatus(Enum):
    WAIT_TO_START = 0
    RUNNING = 1
    WAIT_FOR_ANSILLA = 2
    WAIT_FOR_T_GATE = 3
    FINISHED = 4




class process:

    def __init__(self, processID: int , start_time: int, vdataspace: virtualSpace, vsyndromespace: virtualSpace):
        self._start_time=start_time
        self._processID = processID
        self._instruction_list = []
        self._syscall_list = []
        self._current_time = start_time
        self._status = ProcessStatus.WAIT_TO_START
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
        for addr in qubitaddress:
            if addr.is_syndrome():
                if addr not in self._virtual_syndrome_addresses:
                    self._virtual_syndrome_addresses.append(addr)
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





if __name__ == "__main__":

    vdata = virtualSpace(size=10, label="vdata")
    vsyn = virtualSpace(size=5, label="vsyn", is_syndrome=True)
    process_instance = process(processID=1,start_time=0, vdataspace=vdata, vsyndromespace=vsyn)

    process_instance.add_syscall(syscallinst=syscall_allocate_data_qubits(size=3,processID=1))  # Allocate 2 data qubits
    process_instance.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=3,processID=1))  # Allocate 2 syndrome qubits

    print(f"Process data qubit num is {process_instance._num_data_qubits}" )
    print(f"Process syndrome qubit num is {process_instance._num_syndrome_qubits}" )


    print(vdata.get_address(1))

    process_instance.add_instruction(Instype.CNOT, [vdata.get_address(0), vsyn.get_address(0)])  # CNOT operation

    process_instance.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(1)])  # CNOT operation



    process_instance.add_instruction(Instype.MEASURE, [vsyn.get_address(0)])  # Measure operation
    process_instance.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(2)])  # CNOT operation


    print(process_instance)
