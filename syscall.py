from enum import Enum
from typing import List  
from virtualSpace import virtualSpace
"""
syscall.py

This module defines the system call interface for out fault-tolerant quantum kernel.

Classes:
    syscall

Example:


"""


class syscalltype(Enum):
    MAGIC_STATE_DISTILLATION = 1
    ALLOCATE_DATA_QUBITS = 2
    DEALLOCATE_DATA_QUBITS = 3
    ALLOCATE_SYNDROME_QUBITS = 4
    DEALLOCATE_SYNDROME_QUBITS = 5
    ALLOCATE_T_FACTORY = 6
    DEALLOCATE_T_FACTORY = 7


def get_syscall_time(call: 'syscall')-> int:
    """
    Get the clock time associated with a given syscall type.
    
    Args:
        call (syscall): The syscall instance.
    
    Returns:
        int: The clock time for the syscall type.
    """
    match call._syscall_type:
        case syscalltype.MAGIC_STATE_DISTILLATION:
            return 1000  # Example time for magic state distillation
        case syscalltype.ALLOCATE_DATA_QUBITS | syscalltype.ALLOCATE_SYNDROME_QUBITS | syscalltype.ALLOCATE_T_FACTORY:
            return 500  # Example time for allocation syscalls
        case syscalltype.DEALLOCATE_DATA_QUBITS | syscalltype.DEALLOCATE_SYNDROME_QUBITS | syscalltype.DEALLOCATE_T_FACTORY:
            return 1  # Example time for deallocation syscalls
        case _:
            raise ValueError("Unknown syscall type")


class syscall:
    """
    Initialize a new syscall.
    Each syscall has a type and may have a process ID.
    """
    def __init__(self, syscall_type: syscalltype, processID: int = None):
        self._syscall_type = syscall_type
        self._processID = processID


class syscall_magic_state_distillation(syscall):
    """
    Initialize a syscall for magic state distillation.
    """
    def __init__(self, vspace: virtualSpace, address: List[int], processID: int = None):
        """
        Initialize a syscall for magic state distillation.
        :param vspace: The virtual space in which the syscall operates.
        :param args: Arguments for the syscall, e.g., exact address.
        """
        super().__init__(syscall_type=syscalltype.MAGIC_STATE_DISTILLATION, processID=processID)
        self._address = address


    def __str__(self):
        return f"MSD"


class syscall_allocate_data_qubits(syscall):
    """
    Initialize a syscall for allocating data qubits.
    """
    def __init__(self, size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.ALLOCATE_DATA_QUBITS, processID=processID)
        self._size = size

    def __str__(self):
        return f"ADQ: {self._syscall_type.name}, Size: {self._size}"



class syscall_deallocate_data_qubits(syscall):
    """
    Initialize a syscall for deallocating data qubits.
    """
    def __init__(self, virtual_space: virtualSpace, processID: int = None):
        super().__init__(syscall_type=syscalltype.DEALLOCATE_DATA_QUBITS, processID=processID)
        self._virtual_space = virtual_space
        self._size = virtual_space.get_size()

    def __str__(self):
        return f"DDQ: {self._syscall_type.name}, Virtual Space: {self._virtual_space}"


class syscall_allocate_syndrome_qubits(syscall):
    """
    Initialize a syscall for allocating syndrome qubits.
    """
    def __init__(self, size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.ALLOCATE_SYNDROME_QUBITS, processID=processID)
        self._size = size

    def __str__(self):
        return f"ASQ: {self._syscall_type.name}, Size: {self._size}"


class syscall_deallocate_syndrome_qubits(syscall):
    """
    Initialize a syscall for deallocating syndrome qubits.
    """
    def __init__(self, virtual_space: virtualSpace, processID: int = None):
        super().__init__(syscall_type=syscalltype.DEALLOCATE_SYNDROME_QUBITS, processID=processID)
        self._virtual_space = virtual_space
        self._size = virtual_space.get_size()

    def __str__(self):
        return f"DSQ: {self._syscall_type.name}, Space: {self._virtual_space}"



class syscall_allocate_T_factory(syscall):
    """
    Initialize a T gate factory.
    """
    def __init__(self, size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.ALLOCATE_T_FACTORY, processID=processID)
        self._size = size

    def __str__(self):
        return f"ATF: {self._syscall_type.name}, Size: {self._size}"


class syscall_deallocate_T_factory(syscall):
    """
    Initialize a syscall for deallocating a T gate factory.
    """
    def __init__(self, virtual_space: virtualSpace, processID: int = None):
        super().__init__(syscall_type=syscalltype.DEALLOCATE_T_FACTORY, processID=processID)
        self._virtual_space = virtual_space
        self._size = virtual_space.get_size()

    def __str__(self):
        return f"DTF: {self._syscall_type.name}, Space: {self._virtual_space}"



if __name__ == "__main__":

    syscall_instance = syscall(syscall_type=syscalltype.MAGIC_STATE_DISTILLATION, args=[1, 2, 3])

