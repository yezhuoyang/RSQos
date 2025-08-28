#Explicit construction of Logical CNOT gate




from tqec import BlockGraph
from tqec.utils.position import Position3D
from tqec import BlockGraph, compile_block_graph, NoiseModel
from tqec.computation.cube import (
    Cube,
    Port,
    YHalfCube,
    ZXCube,
    cube_kind_from_string,
)
import sinter
import os
import stim
from pathlib import Path






if __name__ == "__main__":


    g = BlockGraph("CNOT")
    cubes = [
        (Position3D(0, 0, 0), "P", "In_Control"),
        (Position3D(0, 0, 1), "ZXZ", ""),    
        (Position3D(0, 0, 2), "P", "Out_Target"),
    ]
    for pos, kind, label in cubes:
        g.add_cube(pos, kind, label)

    pipes = [(0, 1), (1, 2)]

    for p0, p1 in pipes:
        g.add_pipe(cubes[p0][0], cubes[p1][0])
    
    
    filled_gs = g.fill_ports_for_minimal_simulation()
    assert len(filled_gs) == 2
    g = filled_gs[0].graph


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