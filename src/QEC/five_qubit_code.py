from qiskit import QuantumCircuit
from typing import List
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister



class FiveQubitCode:
    """
    Five qubit code implementation.
    Use dynamic circuit to implement the five qubit code decoder
    """




    def __init__(self) -> None:
        self._dataqubits = QuantumRegister(5, 'data')
        self._ancillaqubits = QuantumRegister(4, 'ancilla')
        self._logicreadout_qubit= QuantumRegister(1, 'readout')  
        self._synclbits = ClassicalRegister(4)
        self._logicbits = ClassicalRegister(2)      
        self._circuit = QuantumCircuit(self._dataqubits, self._logicreadout_qubit, self._ancillaqubits ,self._synclbits, self._logicbits)



    def syndrome_measurement(self):

        self._circuit.reset(self._ancillaqubits[0])
        self._circuit.reset(self._ancillaqubits[1])
        self._circuit.reset(self._ancillaqubits[2])
        self._circuit.reset(self._ancillaqubits[3])
        

        self._circuit.h(self._ancillaqubits[0])
        self._circuit.h(self._ancillaqubits[1])
        self._circuit.h(self._ancillaqubits[2])
        self._circuit.h(self._ancillaqubits[3])
        
    


        self._circuit.cx(self._ancillaqubits[0], self._dataqubits[0])
        self._circuit.cz(self._ancillaqubits[0], self._dataqubits[1])
        self._circuit.cz(self._ancillaqubits[0], self._dataqubits[2])
        self._circuit.cx(self._ancillaqubits[0], self._dataqubits[3])


        self._circuit.cx(self._ancillaqubits[1], self._dataqubits[1])
        self._circuit.cz(self._ancillaqubits[1], self._dataqubits[2])
        self._circuit.cz(self._ancillaqubits[1], self._dataqubits[3])
        self._circuit.cx(self._ancillaqubits[1], self._dataqubits[4])


        self._circuit.cx(self._ancillaqubits[2], self._dataqubits[0])
        self._circuit.cx(self._ancillaqubits[2], self._dataqubits[2])
        self._circuit.cz(self._ancillaqubits[2], self._dataqubits[3])
        self._circuit.cz(self._ancillaqubits[2], self._dataqubits[4])       


        self._circuit.cz(self._ancillaqubits[3], self._dataqubits[0])
        self._circuit.cx(self._ancillaqubits[3], self._dataqubits[1])
        self._circuit.cx(self._ancillaqubits[3], self._dataqubits[3])
        self._circuit.cz(self._ancillaqubits[3], self._dataqubits[4])       


        self._circuit.h(self._ancillaqubits[0])
        self._circuit.h(self._ancillaqubits[1])
        self._circuit.h(self._ancillaqubits[2])
        self._circuit.h(self._ancillaqubits[3])


        self._circuit.measure(self._ancillaqubits[0], self._synclbits[0])
        self._circuit.measure(self._ancillaqubits[1], self._synclbits[1])
        self._circuit.measure(self._ancillaqubits[2], self._synclbits[2])
        self._circuit.measure(self._ancillaqubits[3], self._synclbits[3])

        #TODO:Check the binary values for the if_test statements, are these conditions consistent with the syndrome measurements bits?
        with self._circuit.if_test((self._synclbits, 0b0001)):
            self._circuit.x(self._dataqubits[1])   
        with self._circuit.if_test((self._synclbits, 0b0010)):
            self._circuit.z(self._dataqubits[4])   
        with self._circuit.if_test((self._synclbits, 0b0011)):
            self._circuit.x(self._dataqubits[2])   
        with self._circuit.if_test((self._synclbits, 0b0100)):
            self._circuit.z(self._dataqubits[2])  
        with self._circuit.if_test((self._synclbits, 0b0101)):
            self._circuit.z(self._dataqubits[0])
        with self._circuit.if_test((self._synclbits, 0b0110)):
            #self._circuit.x(self._dataqubits[3])
            pass
        with self._circuit.if_test((self._synclbits, 0b0111)):
            self._circuit.y(self._dataqubits[2])
        with self._circuit.if_test((self._synclbits, 0b1000)):
            self._circuit.x(self._dataqubits[0])
        with self._circuit.if_test((self._synclbits, 0b1001)):
            self._circuit.z(self._dataqubits[3])
        with self._circuit.if_test((self._synclbits, 0b1010)):
            self._circuit.z(self._dataqubits[1])
        with self._circuit.if_test((self._synclbits, 0b1011)):
            self._circuit.y(self._dataqubits[0])
        with self._circuit.if_test((self._synclbits, 0b1100)):
            self._circuit.x(self._dataqubits[4])     
        with self._circuit.if_test((self._synclbits, 0b1101)):
            self._circuit.y(self._dataqubits[1])
        with self._circuit.if_test((self._synclbits, 0b1110)):
            self._circuit.y(self._dataqubits[4])
        with self._circuit.if_test((self._synclbits, 0b1111)):
            self._circuit.y(self._dataqubits[3]) 



    def logical_readout(self,k:int=0):
        self._circuit.reset(self._logicreadout_qubit[0])
        self._circuit.cx(self._dataqubits[0],self._logicreadout_qubit[0])
        self._circuit.cx(self._dataqubits[1],self._logicreadout_qubit[0])
        self._circuit.cx(self._dataqubits[2],self._logicreadout_qubit[0])
        self._circuit.cx(self._dataqubits[3],self._logicreadout_qubit[0])
        self._circuit.cx(self._dataqubits[4],self._logicreadout_qubit[0])
        self._circuit.measure(self._logicreadout_qubit[0], self._logicbits[k])



    def construct_circuit(self):


        self.syndrome_measurement()
        self.logical_readout(0)
        self.syndrome_measurement()        
        self.logical_readout(1)




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


    def construct_decoding_table(self):
        pass




    def construct_process(self):
        pass



if __name__ == "__main__":

    five_qubit_code = FiveQubitCode()
    five_qubit_code.construct_circuit()
    five_qubit_code.draw_circuit()


    counts = five_qubit_code.simulate()

    print("Simulation results:", counts)