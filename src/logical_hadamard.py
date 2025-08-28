from fractions import Fraction

from pyzx import EdgeType, VertexType
from pyzx.graph.graph_s import GraphS

from tqec.interop import block_synthesis
from tqec.utils.position import Position3D
from tqec import BlockGraph, compile_block_graph, NoiseModel



if __name__ == "__main__":
    # Example usage of the block synthesis
    # Define a ZX graph

    g_zx = GraphS()
    g_zx.add_vertices(5)
    g_zx.set_type(1, VertexType.Z)
    g_zx.set_type(3, VertexType.Z)
    g_zx.set_type(4, VertexType.Z)
    g_zx.set_phase(4, Fraction(1, 2))
    g_zx.add_edges([(0, 1), (1, 2), (1, 3), (3, 4)])
    g_zx.set_inputs((0,))
    g_zx.set_outputs((2,))

    positions = {
        0: Position3D(0, 0, 0),
        1: Position3D(0, 0, 1),
        2: Position3D(0, 0, 2),
        3: Position3D(1, 0, 1),
        4: Position3D(1, 0, 2),
    }

    g = block_synthesis(g_zx, positions=positions)

    # 2. Get the correlation surfaces of interest and compile the computation
    correlation_surfaces = g.find_correlation_surfaces()
    compiled_computation = compile_block_graph(g, observables=[correlation_surfaces[0]])

    # 3. Generate the `stim.Circuit` of target code distance
    circuit = compiled_computation.generate_stim_circuit(
        # k = (d-1)/2 is the scale factor
        # Large values will take a lot of time.
        k=2,
        # The noise applied and noise levels can be changed.
        noise_model=NoiseModel.uniform_depolarizing(0.001),
    )

    circuit=stim.Circuit(circuit)


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
