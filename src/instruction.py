from enum import Enum
from typing import List
from virtualSpace import virtualSpace, virtualAddress  
import qiskit.qasm2
"""
instruction.py

This module defines the atomic structure of a user process: the Instruction class
Each instruction has a type (e.g., H, CNOT, MEASURE), a list of target qubits, and a timestamp.

Classes:
    Instruction -- Represents a quantum gate operation or measurement

Example:
    >>> inst = Instruction(type=Instype.H, qubit=[0], time=5)
    >>> print(inst)
    H gate on qubit 0 at time 5
"""

class Instype(Enum):
    H = 1
    X = 2
    Y = 3
    Z = 3
    T = 4
    Tdg = 5
    S = 6
    Sdg = 7
    SX = 8
    RZ = 9
    RX = 10
    RY = 11
    U3 = 12
    Toffoli = 13
    CNOT = 14
    SWAP = 15
    CSWAP = 16
    CU1 = 17
    RESET = 18
    MEASURE = 19
    BARRIER = 20


"""
Constants for clock times associated with different quantum operations.
These constants define the time taken for single-qubit gates, two-qubit gates, reset,
and measurement operations in a quantum circuit simulation.
"""
SINGLE_QUBIT_CLOCKTIME = 1
TWO_QUBIT_CLOCKTIME = 1
THREE_QUBIT_CLOCKTIME = 8
RESET_CLOCKTIME = 100
MEASURE_CLOCKTIME = 100
T_STATE_CLOCKTIME = 500

def get_clocktime(type: Instype) -> int:
    """
    Get the clock time associated with a given instruction type.
    
    Args:
        type (Instype): The type of the instruction.
    
    Returns:
        int: The clock time for the instruction type.
    """
    match type:
        case Instype.H | Instype.X | Instype.Y | Instype.Z | Instype.T | Instype.Tdg | Instype.S | Instype.Sdg | Instype.SX | Instype.RZ | Instype.RX | Instype.RY | Instype.U3:
            return SINGLE_QUBIT_CLOCKTIME
        case Instype.CNOT | Instype.SWAP | Instype.CU1:
            return TWO_QUBIT_CLOCKTIME
        case Instype.Toffoli | Instype.CSWAP:
            return THREE_QUBIT_CLOCKTIME
        case Instype.RESET:
            return RESET_CLOCKTIME
        case Instype.MEASURE:
            return MEASURE_CLOCKTIME
        case Instype.BARRIER:
            return 0  # Barrier operations do not consume clock time
        case _:
            raise ValueError("Unknown instruction type")



def get_gate_type_name(type: Instype) -> str:
    """
    Get the name of the gate type.
    
    Args:
        type (Instype): The type of the instruction.
    
    Returns:
        str: The name of the instruction type.
    """
    match type:
        case Instype.H:
            return "H"
        case Instype.X:
            return "X"  
        case Instype.Y:
            return "Y"
        case Instype.Z:
            return "Z"
        case Instype.T:
            return "T"
        case Instype.Tdg:
            return "Tdg"
        case Instype.S:
            return "S"
        case Instype.Sdg:
            return "Sdg"
        case Instype.SX:
            return "SX"
        case Instype.RZ:
            return "RZ"
        case Instype.RX:
            return "RX"
        case Instype.RY:
            return "RY"
        case Instype.U3:
            return "U3"
        case Instype.Toffoli:
            return "Toffoli"
        case Instype.CNOT:
            return "CNOT"
        case Instype.SWAP:
            return "SWAP"
        case Instype.CSWAP:
            return "CSWAP"
        case Instype.CU1:   
            return "CU1"
        case Instype.RESET:
            return "RESET"
        case Instype.MEASURE:
            return "MEASURE"
        case Instype.BARRIER:
            return "BARRIER"
        case _:
            raise ValueError("Unknown instruction type")




class instruction:

    """
    Initialize a new instruction.
    """
    def __init__(self, type:Instype, qubitaddress:List[virtualAddress],processID: int,time: int,classical_address: int=None, params: List[float]=None) -> None:
        self._type=type
        self._time=time # This is the tim when the instruction is executed in the virtual machine
        self._scheduled_time=None  # This is the real time an instruction is performed in hardware after scheduling
        self._clock_time=get_clocktime(type)
        self._qubitaddress=qubitaddress
        self._scheduled_mapped_address={} # This is the physical address after scheduling
        self._processID=processID
        self._params=params if params is not None else [] # The rotation angles for rotation gates, for example, for RZ(0.2*pi), params=[0.2]
        if self._type==Instype.MEASURE:
            if classical_address is None:
                raise ValueError("Classical address must be provided for MEASURE instruction.")
            self._classical_address=classical_address


    def get_params(self) -> List[float]:
        """
        Return the parameters associated with the instruction.
        
        Returns:
            List[float]: The list of parameters for the instruction.
        """
        return self._params
    

    def get_classical_address(self) -> int:
        """
        Get the classical address associated with the instruction.
        
        Returns:
            int: The classical address for the instruction.
        """
        if self._type != Instype.MEASURE:
            raise ValueError("Classical address is only applicable for MEASURE instructions.")
        return self._classical_address


    def reset_mapping(self):
        self._scheduled_mapped_address={} # This is the physical address after scheduling
        self._scheduled_time=None  # This is the real time an instruction is performed in hardware after scheduling


    def is_reset(self) -> bool:
        """
        Check if the instruction is a reset operation.
        
        Returns:
            bool: True if the instruction is a reset, False otherwise.
        """
        return self._type == Instype.RESET



    def is_measurement(self) -> bool:
        """
        Check if the instruction is a measurement operation.
        
        Returns:
            bool: True if the instruction is a measurement, False otherwise.
        """
        return self._type == Instype.MEASURE



    def set_scheduled_mapped_address(self, qubitaddress: virtualAddress, physicaladdress: int):
        """
        Set the scheduled mapped address for a given qubit address.
        
        Args:
            qubitaddress (virtualAddress): The virtual address of the qubit.
            physicaladdress (int): The physical address to which the qubit is mapped.
        """
        self._scheduled_mapped_address[qubitaddress] = physicaladdress


    def get_scheduled_mapped_address(self, qubitaddress: virtualAddress) -> int:
        """
        Get the scheduled mapped addresses for the instruction.
        
        Returns:
            dict: A dictionary mapping virtual qubit addresses to physical addresses.
        """
        return self._scheduled_mapped_address[qubitaddress]


    def get_processID(self) -> int:
        """
        Get the process ID associated with the instruction.
        """
        return self._processID


    def get_type(self) -> Instype:
        """
        Get the type of the instruction.
        """
        return self._type

    def get_qubitaddress(self) -> List[virtualAddress]:
        """
        Get the list of qubit addresses associated with the instruction.
        """
        return self._qubitaddress


    def get_scheduled_time(self) -> int:
        """
        Get the scheduled time for the instruction.
        This is the time when the instruction is actually executed in hardware.
        """
        if self._scheduled_time is None:
            print(self._type)
            raise ValueError("Instruction has not been scheduled yet.")
        return self._scheduled_time
    

    def set_scheduled_time(self, scheduled_time: int):
        """
        Set the scheduled time for the instruction.
        This is the time when the instruction is actually executed in hardware.
        
        Args:
            scheduled_time (int): The time when the instruction is scheduled to be executed.
        """
        self._scheduled_time = scheduled_time



    def get_time(self) -> int:
        """
        Get the time associated with the instruction.
        This is only know and resolved after scheduling
        """
        return self._time


    def get_clock_time(self) -> int:
        """
        Get the clock time associated with the instruction.
        """
        return self._clock_time


    def __str__(self):
        outputstr=""
        match self._type:
            case Instype.H:
                outputstr+="H"
            case Instype.X:
                outputstr+="X"
            case Instype.Y:
                outputstr+="Y"
            case Instype.Z:
                outputstr+="Z"
            case Instype.T:
                outputstr+="T"
            case Instype.Tdg:
                outputstr+="Tdg"
            case Instype.S:
                outputstr+="S"
            case Instype.Sdg:
                outputstr+="Sdg"
            case Instype.SX:
                outputstr+="SX"
            case Instype.RZ:
                outputstr+="RZ("+str(self._params[0])+"*pi)"
            case Instype.RX:
                outputstr+="RX("+str(self._params[0])+"*pi)"
            case Instype.RY:
                outputstr+="RY("+str(self._params[0])+"*pi)"
            case Instype.U3:
                outputstr+="U3("+str(self._params[0])+"*pi, "+str(self._params[1])+"*pi, "+str(self._params[2])+"*pi)"
            case Instype.Toffoli:
                outputstr+="Toffoli"    
            case Instype.CNOT:
                outputstr+="CNOT"
            case Instype.SWAP:
                outputstr+="SWAP"
            case Instype.CSWAP:
                outputstr+="CSWAP"
            case Instype.CU1:
                outputstr+="CU1("+str(self._params[0])+"*pi)"
            case Instype.RESET:
                outputstr+="RESET"
            case Instype.MEASURE:
                outputstr+="c" + str(self._classical_address) + "=MEASURE"
        outputstr+=" on qubit(" + ", ".join(map(str, self._qubitaddress)) + ") at time " + str(self._time)
        return outputstr

    def __repr__(self):
        return self.__str__()





def parse_qasm_instruction(process_ID: int,instruction_str: str) -> List[instruction]:
    """
    Parse a QASM instruction string and return an Instruction object.
    
    Args:
        instruction_str (str): The QASM instruction string.
    """
    circuit = qiskit.qasm2.loads(instruction_str)
    qubit_number = circuit.num_qubits

    vdata = virtualSpace(size=qubit_number, label="vdata")
    vdata.allocate_range(0, qubit_number - 1)

    inst_list = []
    for instr, qargs, cargs in circuit.data:
        name = instr.name.lower()
        # Map QASM instruction names to Instruction types
        if name == "h":
            inst = instruction(type=Instype.H, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "x":
            inst = instruction(type=Instype.X, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "y":
            inst = instruction(type=Instype.Y, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "z":
            inst = instruction(type=Instype.Z, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "t":
            inst = instruction(type=Instype.T, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "tdg":
            inst = instruction(type=Instype.Tdg, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "s":
            inst = instruction(type=Instype.S, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "sdg":
            inst = instruction(type=Instype.Sdg, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "sx":
            inst = instruction(type=Instype.SX, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "rz":
            inst = instruction(type=Instype.RZ, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0, params=[cargs[1]])
        elif name == "rx":
            inst = instruction(type=Instype.RX, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0, params=[cargs[1]])
        elif name == "ry":
            inst = instruction(type=Instype.RY, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0, params=[cargs[1]])
        elif name == "u3":
            inst = instruction(type=Instype.U3, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0, params=cargs[1:])
        elif name == "ccx":
            inst = instruction(type=Instype.Toffoli, qubitaddress=[vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index), vdata.get_address(qargs[2]._index)], processID=process_ID, time=0)
        elif name == "cx":
            inst = instruction(type=Instype.CNOT, qubitaddress=[vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)], processID=process_ID, time=0)
        elif name == "swap":
            inst = instruction(type=Instype.SWAP, qubitaddress=[vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)], processID=process_ID, time=0)
        elif name == "cswap":
            inst = instruction(type=Instype.CSWAP, qubitaddress=[vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index), vdata.get_address(qargs[2]._index)], processID=process_ID, time=0)
        elif name == "cu1":
            inst = instruction(type=Instype.CU1, qubitaddress=[vdata.get_address(qargs[0]._index), vdata.get_address(qargs[1]._index)], processID=process_ID, time=0, params=[cargs[1]])
        elif name == "reset":
            inst = instruction(type=Instype.RESET, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0)
        elif name == "measure":
            inst = instruction(type=Instype.MEASURE, qubitaddress=[vdata.get_address(qargs[0]._index)], processID=process_ID, time=0, classical_address=cargs[0]._index)
        else:
            raise ValueError(f"Unsupported instruction: {name}")

        inst_list.append(inst)

    return inst_list


if __name__ == "__main__":

   file_path = "C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\smallqasm\\adder_n4.qasm"
   with open(file_path, "r") as file:
       qasm_code = file.read()


   parse_qasm_instruction(0,qasm_code)
