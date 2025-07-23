#Fault-tolerant process on a fault-tolerant quantum computer

from tqec import BlockGraph, compile_block_graph, NoiseModel
import sinter
import os
import stim



class process:

    def __init__(self, name, process_type, **kwargs):
        self.name = name
        self.process_type = process_type
        self.parameters = kwargs
        self._circuit_list=[]


    @property
    def add_circuit(self, circuit:BlockGraph):
        """
        Add a BlockGraph circuit to the process.
        """
        if isinstance(circuit, BlockGraph):
            self._circuit_list.append(circuit)
        else:
            raise TypeError("Expected a BlockGraph instance.") 


    def __repr__(self):
        return f"Process(name={self.name}, type={self.process_type}, parameters={self.parameters})"

    def execute(self):
        # Placeholder for execution logic
        print(f"Executing {self.process_type} process: {self.name} with parameters {self.parameters}")







if __name__ == "__main__":
    process_instance = process(name="ExampleProcess", process_type="quantum", param1=42, param2="test")
    print(process_instance)