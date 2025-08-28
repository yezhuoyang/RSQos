"""
virtualAddress.py

This module defines the virtual address space for the fault-tolerant quantum kernel.

Classes:
    virtualSpace -- Represents a virtual space in the quantum kernel
    virtualAddress -- Represents a virtual address in the quantum kernel
Example:


"""


class virtualSpace:


    def __init__(self, size: int, label: str = "default", is_syndrome=False, is_T_factory=False):
        """
        Initialize a new virtual space with a given maximum size.
        """
        self._size = size
        self._label = label
        self._addresses = [virtualAddress(self, i, is_syndrome) for i in range(size)]
        self._is_allocated = [False] * size
        self._is_syndrome = is_syndrome
        self._is_T_factory = is_T_factory

    def get_label(self) -> str:
        return self._label

    def is_syndrome(self) -> bool:
        return self._is_syndrome
    
    def is_T_factory(self) -> bool:
        return self._is_T_factory

    def get_size(self) -> int:
        return self._size

    def __str__(self):
        """
        Return a string representation of the virtual space.
        """
        return f"VSpace '{self._label}' with size {self._size}"

    def __repr__(self):
        """
        Return a string representation of the virtual space for debugging.
        """
        return f"virtualSpace(size={self._size}, label='{self._label}')"


    def allocate_by_index(self, index: int) -> 'virtualAddress':
        """
        Allocate a virtual address in the virtual space.
        """
        if 0 <= index < self._size:
            self._is_allocated[index] = True
            return self._addresses[index]
        else:
            raise IndexError("Index out of bounds.")


    def allocate_range(self, begin_index: int, end_index: int):
        """
        Allocate a range of virtual addresses in the virtual space.
        """
        if 0 <= begin_index < end_index <= self._size:
            for i in range(begin_index, end_index+1):
                self._is_allocated[i] = True
        else:
            raise IndexError("Index out of bounds.")

    def free_range(self, begin_index: int, end_index: int):
        """
        Free a range of virtual addresses in the virtual space.
        """
        if 0 <= begin_index < end_index <= self._size:
            for i in range(begin_index, end_index+1):
                self._is_allocated[i] = False
        else:
            raise IndexError("Index out of bounds.")

    def free_by_index(self, index: int):
        """
        Free a virtual address in the virtual space.
        """
        if 0 <= index < self._size:
            self._is_allocated[index] = False
        else:
            raise IndexError("Index out of bounds.")

    def get_address(self, index: int) -> 'virtualAddress':
        """
        Get the virtual address at the specified index.
        """
        if 0 <= index < self._size:
            if not self._is_allocated[index]:
                raise IndexError("Qubit not allocated!")
            return self._addresses[index]
        else:
            raise IndexError("Index out of bounds.")



class virtualAddress:


    def __init__(self, virtual_space: virtualSpace, index: int, is_syndrome=False, is_T_factory=False):
        """
        Initialize a new virtual address with a given index.
        """
        self._index = index
        self._virtual_space = virtual_space
        self._is_syndrome =  is_syndrome
        self._is_T_factory = is_T_factory

    def is_syndrome(self) -> bool:
        return self._is_syndrome

    def __str__(self):
        """
        Return a string representation of the virtual address.
        """
        return f" {self._virtual_space._label}[{self._index}]"

    def __repr__(self):
        """
        Return a string representation of the virtual address.
        """
        return f" {self._virtual_space._label}[{self._index}]"


    def get_index(self) -> int:
        """
        Get the index of the virtual address.
        """
        return self._index




if __name__ == "__main__":
    vspace = virtualSpace(size=10, label="TestSpace")
    print(vspace)
    vspace.allocate_by_index(4)
    address = vspace.get_address(2)
    print(address)
    print(f"Address index: {address.get_index()}")


    vlist=[ vspace.get_address(2), vspace.get_address(3)]

    print(str(vlist))