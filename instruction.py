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


"""
Constants for clock times associated with different quantum operations.
These constants define the time taken for single-qubit gates, two-qubit gates, reset,
and measurement operations in a quantum circuit simulation.
"""
SINGLE_QUBIT_CLOCKTIME = 1
TWO_QUBIT_CLOCKTIME = 1
RESET_CLOCKTIME = 100
MEASURE_CLOCKTIME = 100


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
        case _:
            raise ValueError("Unknown instruction type")




class instruction:

    """
    Initialize a new instruction.
    """
    def __init__(self, type:Instype, qubitaddress:List[virtualAddress],time: int):
        self._type=type
        self._time=time
        self._clock_time=get_clocktime(type)
        self._qubitaddress=qubitaddress


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

    def get_time(self) -> int:
        """
        Get the time associated with the instruction.
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