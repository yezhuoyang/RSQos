from enum import Enum
from typing import List  
from virtualSpace import virtualSpace, virtualAddress
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



def get_syscall_type_name(call: 'syscall') -> str:
    """
    Get the name of the syscall type.
    
    Args:
        call (syscall): The syscall instance.
    
    Returns:
        str: The name of the syscall type.
    """
    return call._name






class syscall:
    """
    Initialize a new syscall.
    Each syscall has a type and may have a process ID.
    """
    def __init__(self, syscall_type: syscalltype, processID: int = None, time: int = None):
        self._syscall_type = syscall_type
        self._processID = processID
        self._name=None
        self._size=None
        self._time = time  # This is the time when the syscall is executed in the virtual machine
        self._scheduled_time = None  # This is the real time a syscall is performed in
        self._address = None  # This will be set in specific syscall subclasses
        self._scheduled_mapped_address={} # This is the physical address after scheduling


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


    def set_address(self, address: List[virtualAddress]):
        """
        Set the address associated with the syscall.
        
        Args:
            address (List[virtualAddress]): The address to associate with the syscall.
        """
        self._address = address


    def get_address(self) -> List[virtualAddress]:
        """
        Get the address associated with the syscall.
        
        Returns:
            List[virtualAddress]: The address associated with the syscall.
        """
        return self._address

    def get_name(self) -> str:
        """
        Get the name of the syscall.
        
        Returns:
            str: The name of the syscall.
        """
        return self._name


    def get_processID(self) -> int:
        """
        Get the process ID associated with the syscall.
        
        Returns:
            int: The process ID, or None if not set.
        """
        return self._processID


    def get_time(self) -> int:
        """
        Get the time associated with the syscall.
        This is only known and resolved after scheduling.
        """
        return self._time
    

    def get_scheduled_time(self) -> int:
        """
        Get the scheduled time associated with the syscall.
        This is the real time a syscall is performed in hardware after scheduling.
        """
        return self._scheduled_time
    

    def set_scheduled_time(self, time: int):
        """
        Set the scheduled time for the syscall.
        
        Args:
            time (int): The time when the syscall is scheduled to be executed.
        """
        self._scheduled_time = time


class syscall_magic_state_distillation(syscall):
    """
    Initialize a syscall for magic state distillation.
    """
    def __init__(self, address: List[virtualAddress], processID: int = None):
        """
        Initialize a syscall for magic state distillation.
        :param vspace: The virtual space in which the syscall operates.
        :param args: Arguments for the syscall, e.g., exact address.
        """
        super().__init__(syscall_type=syscalltype.MAGIC_STATE_DISTILLATION, processID=processID)
        self._name="MAGIC_STATE_DISTILLATION"
        self.set_address(address)

    def __str__(self):
        return f"MSD"


class syscall_allocate_data_qubits(syscall):
    """
    Initialize a syscall for allocating data qubits.
    """
    def __init__(self, address: List[virtualAddress],size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.ALLOCATE_DATA_QUBITS, processID=processID)
        self._size = size
        self._name="ALLOCATE_DATA_QUBITS"
        self.set_address(address)

    def __str__(self):
        return f"ADQ: {self._syscall_type.name}, Size: {self._size}"



class syscall_deallocate_data_qubits(syscall):
    """
    Initialize a syscall for deallocating data qubits.
    """
    def __init__(self,address: List[virtualAddress],size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.DEALLOCATE_DATA_QUBITS, processID=processID)
        self.set_address(address)
        self._size = size
        self._name="DEALLOCATE_DATA_QUBITS"

    def __str__(self):
        return f"DDQ: {self._syscall_type.name}, Size: {self._size}"


class syscall_allocate_syndrome_qubits(syscall):
    """
    Initialize a syscall for allocating syndrome qubits.
    """
    def __init__(self,address: List[virtualAddress],size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.ALLOCATE_SYNDROME_QUBITS, processID=processID)
        self.set_address(address)
        self._size = size
        self._name="ALLOCATE_SYNDROME_QUBITS"

    def __str__(self):
        return f"ASQ: {self._syscall_type.name}, Size: {self._size}"


class syscall_deallocate_syndrome_qubits(syscall):
    """
    Initialize a syscall for deallocating syndrome qubits.
    """
    def __init__(self,address: List[virtualAddress],size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.DEALLOCATE_SYNDROME_QUBITS, processID=processID)
        self.set_address(address)
        self._size = size
        self._name="DEALLOCATE_SYNDROME_QUBITS"

    def __str__(self):
        return f"DSQ: {self._syscall_type.name}, Size: {self._size}"



class syscall_allocate_T_factory(syscall):
    """
    Initialize a T gate factory.
    """
    def __init__(self, address: List[virtualAddress],size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.ALLOCATE_T_FACTORY, processID=processID)
        self.set_address(address)
        self._name="ALLOCATE_T_FACTORY"
        self._size = size
    def __str__(self):
        return f"ATF: {self._syscall_type.name}, Size: {self._size}"


class syscall_deallocate_T_factory(syscall):
    """
    Initialize a syscall for deallocating a T gate factory.
    """
    def __init__(self, address: List[virtualAddress],size: int, processID: int = None):
        super().__init__(syscall_type=syscalltype.DEALLOCATE_T_FACTORY, processID=processID)
        self.set_address(address)
        self._name="DEALLOCATE_T_FACTORY"
        self._size = size
    def __str__(self):
        return f"DTF: {self._syscall_type.name}, Size: {self._size}"



if __name__ == "__main__":

    syscall_instance = syscall(syscall_type=syscalltype.MAGIC_STATE_DISTILLATION, args=[1, 2, 3])

