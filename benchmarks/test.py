from pathlib import Path


import sinter
import os
import stim













if __name__ == "__main__":


    # # # 4. Write the circuit to a text file
    output_path = Path("schedule")

    # Read the circuit from file
    with output_path.open("r") as f:
        circuit = f.read()

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
        count_observable_error_combos=True,
        max_errors=4000,
        tasks=[mytask],
        decoders=['pymatching'],
    )


    print(samples)

    num_LER=samples[0].errors
    sample_used=samples[0].shots
    
    process1Fail=samples[0].custom_counts['obs_mistake_mask=E_']+samples[0].custom_counts['obs_mistake_mask=EE']
    process2Fail=samples[0].custom_counts['obs_mistake_mask=_E']+samples[0].custom_counts['obs_mistake_mask=EE']    


    p1LER= process1Fail / sample_used
    p2LER= process2Fail / sample_used    

    print(f"P1 LER: {p1LER:.7f}")

    print(f"P2 LER: {p2LER:.7f}")




    output_path = Path("single")
    # Read the circuit from file
    with output_path.open("r") as f:
        circuit = f.read()

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
        count_observable_error_combos=True,
        max_errors=4000,
        tasks=[mytask],
        decoders=['pymatching'],
    )

    num_LER=samples[0].errors
    sample_used=samples[0].shots
    LER= num_LER / sample_used    

    print(f"Single LER: {LER:.7f}")