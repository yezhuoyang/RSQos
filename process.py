#Fault-tolerant process on a fault-tolerant quantum computer

from instruction import Instype, instruction
from virtualSpace import virtualSpace
from syscall import syscall, syscalltype



class process:

    def __init__(self, processID):
        self._processID = processID
        self._instruction_list = []
        self._syscall_list = []

    @property
    def add_instruction(self, instruction: instruction):
        """
        Add an instruction to the process.
        """
        if isinstance(instruction, instruction):
            self._instruction_list.append(instruction)
        else:
            raise TypeError("Expected an instruction instance.")


    @property
    def add_syscall(self, syscall: syscall):
        """
        Add a syscall to the process.
        """
        if isinstance(syscall, syscall):
            self._syscall_list.append(syscall)
        else:
            raise TypeError("Expected a syscall instance.")



    def execute(self):
        # Placeholder for execution logic
        print(f"Executing {self.process_type} process: {self.name} with parameters {self.parameters}")







if __name__ == "__main__":
    process_instance = process(name="ExampleProcess", process_type="quantum", param1=42, param2="test")
    print(process_instance)