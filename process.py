#Fault-tolerant process on a fault-tolerant quantum computer

from instruction import *
from virtualSpace import virtualSpace
from syscall import *



class process:

    def __init__(self, processID, start_time):
        self._start_time=start_time
        self._processID = processID
        self._instruction_list = []
        self._syscall_list = []
        self._current_time = start_time
        """
        Keep tract of the execution of all instruction
        Help the kernel and schedule to estimate the resources
        consumed_qpu_time keep track of how long this process has been executed on the hardware
        """
        self._executed = {}
        self._consumed_qpu_time = 0


    def add_instruction(self, type:Instype, qubitaddress:List[virtualAddress]):
        """
        Add an instruction to the process.
        """
        time = self._current_time
        self._current_time += get_clocktime(type)  # Increment time for the next instruction 
        inst=instruction(type, qubitaddress, time)
        self._instruction_list.append(inst)
        self._executed[inst] = False

    def get_instruction_list(self):
        return self._instruction_list


    def add_syscall(self, syscallinst: syscall):
        """
        Add a syscall to the process.
        """
        if isinstance(syscallinst, syscall):
            self._syscall_list.append(syscallinst)
            self._instruction_list.append(syscallinst)
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


    def execute_instruction(self, inst: instruction):
        self._executed[inst] = True
        self._consumed_qpu_time += get_clocktime(inst.type) 






if __name__ == "__main__":
    process_instance = process(processID=1)
    vdata = virtualSpace(size=10, label="vdata")
    vsyn = virtualSpace(size=5, label="vsyn")

    process_instance.add_syscall(syscallinst=syscall_allocate_data_qubits(size=2,processID=1))  # Allocate 2 data qubits
    process_instance.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=2,processID=1))  # Allocate 2 syndrome qubits


    process_instance.add_instruction(Instype.CNOT, [vdata.get_address(0), vsyn.get_address(0)])  # CNOT operation
    process_instance.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(1)])  # CNOT operation
    process_instance.add_instruction(Instype.MEASURE, [vsyn.get_address(0)])  # Measure operation
    process_instance.add_instruction(Instype.CNOT, [vdata.get_address(1), vsyn.get_address(2)])  # CNOT operation

    print(process_instance)
