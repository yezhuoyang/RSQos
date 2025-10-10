from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService



APIKEY ="API"

if __name__ == "__main__":


    # Create a new circuit with two qubits
    qc = QuantumCircuit(100)
    
    # Add a Hadamard gate to qubit 0
    qc.h(range(100))
    # Perform a controlled-X gate on qubit 1, controlled by qubit 0
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    qc.cx(5, 6)
    qc.cx(6, 7)
    qc.cx(7, 8)
    qc.cx(8, 9)
    qc.cx(9, 10)
    qc.cx(10, 11)
    qc.cx(11, 12)
    qc.cx(12, 13)
    qc.cx(13, 14)
    qc.cx(14, 15)
    qc.cx(15, 16)
    qc.cx(16, 17)
    qc.cx(17, 18)
    qc.cx(18, 19)
    qc.cx(19, 20)
    qc.cx(20, 21)
    qc.cx(21, 22)
    qc.cx(22, 23)
    qc.cx(23, 24)
    qc.cx(24, 25)
    qc.cx(25, 26)
    qc.cx(26, 27)
    qc.cx(27, 28)
    # Perform a controlled-X gate on qubit 1, controlled by qubit 0
    qc.cx(0, 1)
    qc.measure_all()


    # Return a drawing of the circuit using MatPlotLib ("mpl").
    # These guides are written by using Jupyter notebooks, which
    # display the output of the last line of each cell.
    # If you're running this in a script, use `print(qc.draw())` to
    # print a text drawing.
    #qc.draw("mpl")


    # Set up six different observables.
    
    #observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
    #observables = [SparsePauliOp(label) for label in observables_labels]


    service = QiskitRuntimeService(channel="ibm_cloud",token=APIKEY)
    
    #backend = service.least_busy(simulator=False, operational=True)
    backend = service.backend("ibm_torino")

    print(f">>> Using backend: {backend}")

    # Convert to an ISA circuit and layout-mapped observables.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    
    # run SamplerV2 on the chosen backend
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 4000

    job = sampler.run([isa_circuit])
    pub = job.result()[0]                 # first (and only) PUB result
    counts = pub.join_data().get_counts() # convenient counts helper
    print(counts)

