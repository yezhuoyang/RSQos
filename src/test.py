from pathlib import Path

from tqec import BlockGraph, compile_block_graph, NoiseModel
import sinter
import os
import stim













if __name__ == "__main__":

    # # 0. Path to the logical_cnot.dae file
    # cnot_dae_filepath = Path.cwd() / "assets" / "logical_cnot.dae"
    # if not cnot_dae_filepath.exists():
    #     print(f"The file '{cnot_dae_filepath}' does not exists.")

    # # 1. Construct the logical computation
    # block_graph = BlockGraph.from_dae_file(cnot_dae_filepath)

    # # 2. Get the correlation surfaces of interest and compile the computation
    # correlation_surfaces = block_graph.find_correlation_surfaces()
    # compiled_computation = compile_block_graph(block_graph, observables=[correlation_surfaces[1]])

    # # 3. Generate the `stim.Circuit` of target code distance
    # circuit = compiled_computation.generate_stim_circuit(
    #     # k = (d-1)/2 is the scale factor
    #     # Large values will take a lot of time.
    #     k=2,
    #     # The noise applied and noise levels can be changed.
    #     noise_model=NoiseModel.uniform_depolarizing(0.001),
    # )


    # # # 4. Write the circuit to a text file
    output_path = Path("cnot_stim.txt")
    # with output_path.open("w") as f:
    #     f.write(str(circuit))

    # Read the circuit from file
    with output_path.open("r") as f:
        circuit = f.read()

    circuit=stim.Circuit(circuit)


    print(f"Circuit has been written to {output_path}")


    pvalue=0.001
    samplebudget=500000000
    mytask=sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        'p': pvalue,
                        'd': 0,
                    },
                )            

    samples = sinter.collect(
        num_workers=os.cpu_count(),
        max_shots=samplebudget,
        max_errors=100,
        tasks=[mytask],
        decoders=['pymatching'],
    )


    print(samples)

    num_LER=samples[0].errors

    sample_used=samples[0].shots

    LER= num_LER / sample_used

    print(f"Number of logical errors: {num_LER}")
    print(f"Number of samples used: {sample_used}")
    print(f"Logical error rate: {LER:.4f}")