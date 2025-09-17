from qiskit import QuantumCircuit
from typing import List
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister


class RepetitionCode:
    """
    The repetition code that only protect against bit-flip errors.
    """




    def __init__(self) -> None:
        self._dataqubits = QuantumRegister(3, 'data')
        self._ancillaqubits = QuantumRegister(2, 'ancilla')
        self._synclbits = ClassicalRegister(2)
        self._logiclbit = ClassicalRegister(3)
        self._circuit = QuantumCircuit(self._dataqubits,  self._ancillaqubits ,self._synclbits, self._logiclbit)




    def construct_circuit(self):


        self._circuit.x(self._dataqubits[2])  # Introduce an X error on the first data qubit
        self._circuit.cx(self._dataqubits[0],self._ancillaqubits[0])
        self._circuit.cx(self._dataqubits[1],self._ancillaqubits[0])


        self._circuit.cx(self._dataqubits[1],self._ancillaqubits[1])
        self._circuit.cx(self._dataqubits[2],self._ancillaqubits[1])


        self._circuit.measure(self._ancillaqubits[0], self._synclbits[0])
        self._circuit.measure(self._ancillaqubits[1], self._synclbits[1])


        with self._circuit.if_test((self._synclbits, 0b01)):
            self._circuit.x(self._dataqubits[0])
        with self._circuit.if_test((self._synclbits, 0b10)):
            self._circuit.x(self._dataqubits[2])
        with self._circuit.if_test((self._synclbits, 0b11)):
            self._circuit.x(self._dataqubits[1])


        self._circuit.measure(self._dataqubits, self._logiclbit)



    def draw_circuit(self):
        """
        Save the circuit to a file circuit.png
        """
        self._circuit.draw('mpl', filename='circuit.png')



    def simulate(self):
        simulator = AerSimulator()
        
        
        result = simulator.run(self._circuit, shots=1024).result()

        counts = result.get_counts(self._circuit)
        # print("\n=== Counts(Fake hardware) ===")
        #print(counts)   
        
        
        return counts


    def create_processor(self):
        pass



if __name__ == "__main__":

    rep_code = RepetitionCode()
    rep_code.construct_circuit()
    rep_code.draw_circuit()

    counts = rep_code.simulate()

    print("Simulation results:", counts)