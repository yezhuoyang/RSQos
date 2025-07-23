from pathlib import Path

from tqec import BlockGraph, compile_block_graph, NoiseModel

# 0. Path to the logical_cnot.dae file
cnot_dae_filepath = Path.cwd() / "assets" / "logical_cnot.dae"
if not cnot_dae_filepath.exists():
    print(f"The file '{cnot_dae_filepath}' does not exists.")

# 1. Construct the logical computation
block_graph = BlockGraph.from_dae_file(cnot_dae_filepath)

# 2. Get the correlation surfaces of interest and compile the computation
correlation_surfaces = block_graph.find_correlation_surfaces()
compiled_computation = compile_block_graph(block_graph, observables=[correlation_surfaces[1]])

# 3. Generate the `stim.Circuit` of target code distance
circuit = compiled_computation.generate_stim_circuit(
    # k = (d-1)/2 is the scale factor
    # Large values will take a lot of time.
    k=2,
    # The noise applied and noise levels can be changed.
    noise_model=NoiseModel.uniform_depolarizing(0.001),
)