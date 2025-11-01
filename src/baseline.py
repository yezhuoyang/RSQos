# Run 1 circuit at a time as baseline
# checklist before running:
# 1. small only or small+med
# 2. which quantum computer we are using
# 3. which access token we are using
from HypervisorBackend import *
from vm_executable import *
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJobFailureError
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qasmbench import QASMBenchmark


# for real machine, just submit jobs, get results later

real_job_queue = []
# add automatic job queue maintainance
for i in range(len(circ_list)):
    if circ_name_list[i] in exclude_tests:
        continue
    circ_transpiled = transpile(circ_list[i], backend)
    if len(real_job_queue) == 3: # IBM Quantum allows at most 3 jobs in the queue
        try:
            res = real_job_queue[0].result() # use result to block
        except RuntimeJobFailureError:
            print('failed job:', real_job_queue[0].job_id())
        real_job_queue.pop(0)
        cal_file.write(str(score_all(hypervisor, vm_coupling_map)) + '\n')

    job = real_sampler.run([circ_transpiled])
    print(job.job_id(), circ_name_list[i])
    real_job_queue.append(job)

while len(real_job_queue):
    try:
        res = real_job_queue[0].result() # use result to block
    except RuntimeJobFailureError:
        print('failed job:', real_job_queue[0].job_id())
    real_job_queue.pop(0)
    # write calibration data when a job finishes
    cal_file.write(str(score_all(hypervisor, vm_coupling_map)) + '\n')
    
cal_file.close()