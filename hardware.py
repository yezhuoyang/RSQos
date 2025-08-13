import stim
from process import *
from instruction import *
from kernel import *
from virtualSpace import *

class virtualHardware:


    def __init__(self, qubit_number:int, error_rate:float):
        """
        Initialize the virtual hardware with configuration parameters.
        """
        self._physical_qubits = qubit_number
        self._error_rate = error_rate



class virtualHardwareMapping:


    def __init__(self, hardware_instance: virtualHardware):
        """
        Initialize the virtual hardware manager with a given hardware instance.
        """
        self._hardware = hardware_instance
        self._mapping = {}


    def add_mapping(self, virtual_address: virtualAddress, physical_qubit: int):
        """
        Add a mapping from a virtual address to a physical qubit.
        """
        if physical_qubit >= self._hardware._physical_qubits:
            raise ValueError("Invalid physical qubit number.")
        self._mapping[virtual_address] = physical_qubit


    def get_physical_qubit(self, virtual_address: virtualAddress) -> int:
        """
        Get the physical qubit mapped to a virtual address.
        """
        return self._mapping.get(virtual_address)

        # For example: self._mapping[virtual_address] = physical_qubit


    def transpile(self,process_instance):
        """
        Transpile the process instructions to stim program given the mapping
        """
        circuit = stim.Circuit()
        for inst in process_instance._instruction_list:
            if isinstance(inst, instruction):
                match inst.get_type():
                    case Instype.CNOT:
                        qubit1 = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        qubit2 = self.get_physical_qubit(inst.get_qubitaddress()[1])
                        circuit.append("CNOT", [qubit1, qubit2])
                    case Instype.H:
                        qubit = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        circuit.append("H", [qubit])
                    case Instype.MEASURE:
                        qubit = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        circuit.append("M", [qubit])
                    case Instype.RESET:
                        qubit = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        circuit.append("R", [qubit])
                    case _:
                        raise ValueError("Unknown instruction type")

        return circuit


    def transpile(self,process_instance):
        """
        Transpile the process instructions to stim program given the mapping
        """
        circuit = stim.Circuit()
        for inst in process_instance._instruction_list:
            if isinstance(inst, instruction):
                match inst.get_type():
                    case Instype.CNOT:
                        qubit1 = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        qubit2 = self.get_physical_qubit(inst.get_qubitaddress()[1])
                        circuit.append("CNOT", [qubit1, qubit2])
                    case Instype.H:
                        qubit = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        circuit.append("H", [qubit])
                    case Instype.MEASURE:
                        qubit = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        circuit.append("M", [qubit])
                    case Instype.RESET:
                        qubit = self.get_physical_qubit(inst.get_qubitaddress()[0])
                        circuit.append("R", [qubit])
                    case _:
                        raise ValueError("Unknown instruction type")

        return circuit


if __name__ == "__main__":


    virtual_hardware = virtualHardware(qubit_number=4, error_rate=0.01)
    virtual_hardware_mapping = virtualHardwareMapping(virtual_hardware)


    vdata1 = virtualSpace(size=10, label="vdata")
    vsyn1 = virtualSpace(size=5, label="vsyn")

    process_instance1 = process(processID=1)
    process_instance1.add_syscall(syscallinst=syscall_allocate_data_qubits(size=2,processID=1))  # Allocate 2 data qubits
    process_instance1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=2,processID=1))  # Allocate 2 syndrome qubits
    process_instance1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    process_instance1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])  # CNOT operation
    process_instance1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation


    vdata2 = virtualSpace(size=10, label="vdata")
    vsyn2 = virtualSpace(size=5, label="vsyn")

    process_instance2 = process(processID=2)
    process_instance2.add_syscall(syscallinst=syscall_allocate_data_qubits(size=2,processID=2))  # Allocate 2 data qubits
    process_instance2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(size=2,processID=2))  # Allocate 2 syndrome qubits
    process_instance2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])  # CNOT operation
    process_instance2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])  # CNOT operation
    process_instance2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation

    virtual_hardware_mapping.add_mapping(vdata1.get_address(0), 0)
    virtual_hardware_mapping.add_mapping(vdata1.get_address(1), 1)
    virtual_hardware_mapping.add_mapping(vsyn1.get_address(0), 2)
    virtual_hardware_mapping.add_mapping(vsyn1.get_address(1), 3)

    stim_circuit = virtual_hardware_mapping.transpile(process_instance1)

    print(stim_circuit)
