from enum import Enum
from typing import List
from virtualSpace import virtualSpace, virtualAddress  
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
    CNOT = 4
    RESET = 5
    MEASURE = 6
    BARRIER = 7  


"""
Constants for clock times associated with different quantum operations.
These constants define the time taken for single-qubit gates, two-qubit gates, reset,
and measurement operations in a quantum circuit simulation.
"""
SINGLE_QUBIT_CLOCKTIME = 1
TWO_QUBIT_CLOCKTIME = 1
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
        case Instype.H | Instype.X | Instype.Y | Instype.Z:
            return SINGLE_QUBIT_CLOCKTIME
        case Instype.CNOT:
            return TWO_QUBIT_CLOCKTIME
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
        case Instype.CNOT:
            return "CNOT"
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
    def __init__(self, type:Instype, qubitaddress:List[virtualAddress],processID: int,time: int):
        self._type=type
        self._time=time # This is the tim when the instruction is executed in the virtual machine
        self._scheduled_time=None  # This is the real time an instruction is performed in hardware after scheduling
        self._clock_time=get_clocktime(type)
        self._qubitaddress=qubitaddress
        self._scheduled_mapped_address={} # This is the physical address after scheduling
        self._processID=processID


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
            case Instype.CNOT:
                outputstr+="CNOT"
            case Instype.RESET:
                outputstr+="RESET"
            case Instype.MEASURE:
                outputstr+="MEASURE"
        outputstr+=" on qubit(" + ", ".join(map(str, self._qubitaddress)) + ") at time " + str(self._time)
        return outputstr


    def __repr__(self):
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
            case Instype.CNOT:
                outputstr+="CNOT"
            case Instype.RESET:
                outputstr+="RESET"
            case Instype.MEASURE:
                outputstr+="MEASURE"
        outputstr+=" on qubit(" + ", ".join(map(str, self._qubitaddress)) + ") at time " + str(self._time)
        return outputstr


if __name__ == "__main__":

    inst = instruction(type=Instype.H, qubitaddress=[0], time=5)
    print(inst)