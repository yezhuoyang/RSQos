"""
virtualAddress.py

This module defines the virtual address space for the fault-tolerant quantum kernel.

Classes:
    virtualSpace -- Represents a virtual space in the quantum kernel
    virtualAddress -- Represents a virtual address in the quantum kernel
Example:


"""


class virtualSpace:


    def __init__(self, size: int, label: str = "default"):
        """
        Initialize a new virtual space with a given size.
        """
        self._size = size
        self._label = label
        self._addresses = [virtualAddress(self, i) for i in range(size)]


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


    def get_address(self, index: int) -> 'virtualAddress':
        """
        Get the virtual address at the specified index.
        """
        if 0 <= index < self._size:
            return self._addresses[index]
        else:
            raise IndexError("Index out of bounds.")



class virtualAddress:


    def __init__(self, virtual_space: virtualSpace, index: int):
        """
        Initialize a new virtual address with a given index.
        """
        self._index = index
        self._virtual_space = virtual_space

    def __str__(self):
        """
        Return a string representation of the virtual address.
        """
        return f" {self._virtual_space._label}[{self._index}]"

    def __repr__(self):
        """
        Return a string representation of the virtual address.
        """
        return f"virtualAddress(virtual_space={self._virtual_space._label}, index={self._index})"


    def get_index(self) -> int:
        """
        Get the index of the virtual address.
        """
        return self._index




if __name__ == "__main__":
    vspace = virtualSpace(size=10, label="TestSpace")
    print(vspace)
    address = vspace.get_address(2)
    print(address)
    print(f"Address index: {address.get_index()}")


    vlist=[ vspace.get_address(2), vspace.get_address(3)]

    print(str(vlist))