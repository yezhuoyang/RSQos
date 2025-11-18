"""
Simulate a possion distribution
Get the throughput, average waiting time
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import random
import copy
import math
from typing import Any, Iterable
import matplotlib.pyplot as plt

from collections import deque
from dataclasses import dataclass
import random
from qiskit import QuantumCircuit, transpile
# visualize_layout.py
from qiskit.visualization import plot_coupling_map
import matplotlib
matplotlib.use("Agg")              # call BEFORE importing pyplot
import matplotlib.pyplot as plt
from qiskit.transpiler import CouplingMap
from scheduler import *
from process import process, parse_qasm_instruction
import numpy as np
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error,pauli_error
from fakeHardware import construct_10_qubit_hardware, construct_20_qubit_hardware, plot_process_schedule_on_20_qubit_hardware, save_circuit_png, plot_process_schedule_on_10_qubit_hardware, build_noise_model,distribution_fidelity
from instruction import *
from qiskit.qasm2 import dumps



label_name_map = {
    1: "adder_n4",
    2: "basis_trotter_n4",
    3: "bb84_n8",
    4: "bell_n4",
    5: "cat_state_n4",
    6: "deutsch_n2",
    7: "dnn_n2",
    8: "dnn_n8",
    9: "error_correctiond3_n5",
    10: "fredkin_n3",
    11: "grover_n2",
    12: "hs4_n4",
    13: "ising_n10",
    14: "iswap_n2",
    15: "lpn_n5",
    16: "qaoa_n3",
    17: "qaoa_n6",
    18: "qec_en_n5",
    19: "qft_n4",
    20: "qrng_n4",
    21: "simon_n6",
    22: "teleportation_n3",
    23: "toffoli_n3",
    24: "vqe_n4",
    25: "wstate_n3"
}


# -------------------------------------------------------------------
# Optional: a small helper to extract processes from your Kernel object
# -------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessTemplate:
    pid: int
    label: str
    template: Any  # your parsed process object (from parse_qasm_instruction)

class ProcessPool:
    """Holds parsed process templates and returns safe-to-use copies."""
    def __init__(self, kernel_instance: Any):
        # Try a few common attribute names to get the process list
        candidates = [
            getattr(kernel_instance, "processes", None),
            getattr(kernel_instance, "procs", None),
            getattr(kernel_instance, "process_list", None),
            getattr(kernel_instance, "proc_list", None),
        ]
        procs = next((c for c in candidates if c is not None), None)
        if procs is None:
            # Fall back: try a method
            getter = getattr(kernel_instance, "get_processes", None)
            if callable(getter):
                procs = getter()
        if procs is None:
            raise AttributeError("Cannot find process list on kernel_instance. "
                                 "Expected attribute 'processes'/'process_list' or method 'get_processes()'.")

        # Heuristically extract pid and label from each process object
        self.pool: list[ProcessTemplate] = []
        for p in procs:
            # Common attributes people use:
            pid = getattr(p, "pid", getattr(p, "process_ID", getattr(p, "id", None)))
            label = getattr(p, "label", getattr(p, "name", f"proc_{pid if pid is not None else len(self.pool)}"))
            if pid is None:
                # If no explicit pid, assign a stable one
                pid = len(self.pool) + 1
            self.pool.append(ProcessTemplate(pid=pid, label=str(label), template=p))

        if not self.pool:
            raise ValueError("ProcessPool is empty: kernel has no processes.")

    def pick(self, rng: random.Random) -> ProcessTemplate:
        """Return a random template (not copied—just the template’s identity)."""
        return rng.choice(self.pool)

    def instantiate(self, tmpl: ProcessTemplate) -> Any:
        """Deep copy to create a runnable instance for the queue."""
        return copy.deepcopy(tmpl.template)

# -------------------------------------------------------------------
# Poisson arrival simulator that draws jobs from a ProcessPool
# -------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    id: int
    arrival_time: float
    process_label: str
    process_obj: Any  # deep-copied instance

class PoissonArrivalSimulator:
    """
    Event-driven Poisson arrival simulator that enqueues random parsed processes.

    - rate:      lambda (λ), expected arrivals per unit time (> 0)
    - add_prob:  probability an arrival is actually enqueued (0..1)
    - seed:      optional RNG seed for reproducibility
    """
    def __init__(self, rate: float, process_pool: ProcessPool,
                 add_prob: float = 1.0, seed: int | None = None):
        if rate <= 0:
            raise ValueError("rate (λ) must be > 0")
        if not (0.0 <= add_prob <= 1.0):
            raise ValueError("add_prob must be in [0,1]")
        self.rate = rate
        self.add_prob = add_prob
        self.pool = process_pool
        self.rng = random.Random(seed)

        self.t = 0.0
        self._next_arrival = self.t + self._exp()
        self.queue: deque[Job] = deque()
        self._next_id = 0

        # stats
        self.total_arrivals = 0
        self.total_enqueued = 0
        self.dropped = 0

        # event log: list of dicts with keys:
        # time, event ('arrive','enqueue','drop','dequeue'), job_id, process_label
        self.log: list[dict] = []

    # --- internals ---
    def _exp(self) -> float:
        return self.rng.expovariate(self.rate)

    def _log(self, **kwargs):
        rec = {"time": self.t} | kwargs
        self.log.append(rec)

    # --- public API ---
    def advance_to(self, t_end: float) -> None:
        """Advance the clock to t_end, processing all arrivals in (t, t_end]."""
        if t_end < self.t:
            raise ValueError("t_end must be >= current time")

        while self._next_arrival <= t_end:
            self.t = self._next_arrival
            self.total_arrivals += 1
            self._log(event="arrive", job_id=None, process_label=None)

            # Sample a process from the pool (template + deep copy for instance)
            tmpl = self.pool.pick(self.rng)
            proc_inst = self.pool.instantiate(tmpl)

            if self.rng.random() <= self.add_prob:
                job = Job(
                    id=self._next_id,
                    arrival_time=self.t,
                    process_label=tmpl.label,
                    process_obj=proc_inst
                )
                self._next_id += 1
                self.queue.append(job)
                self.total_enqueued += 1
                self._log(event="enqueue", job_id=job.id, process_label=tmpl.label)
            else:
                self.dropped += 1
                self._log(event="drop", job_id=None, process_label=tmpl.label)

            self._next_arrival = self.t + self._exp()

        self.t = t_end

    def run(self, duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must be >= 0")
        self.advance_to(self.t + duration)

    def pop(self) -> Job | None:
        """Pop next job (FIFO)."""
        if not self.queue:
            return None
        job = self.queue.popleft()
        self._log(event="dequeue", job_id=job.id, process_label=job.process_label)
        return job

    def peek(self) -> Job | None:
        return self.queue[0] if self.queue else None

    def __len__(self) -> int:
        return len(self.queue)

    def metrics(self) -> dict:
        return {
            "time": self.t,
            "queue_len": len(self.queue),
            "total_arrivals": self.total_arrivals,
            "total_enqueued": self.total_enqueued,
            "dropped": self.dropped,
            "enqueue_rate_est": (self.total_enqueued / self.t) if self.t > 0 else 0.0,
        }

    # --- plotting ---
    def plot_log(self, show: bool = True, figsize=(8, 6)) -> None:
        """
        Plot cumulative arrivals/enqueues/drops and queue length over time.
        Creates two axes stacked: counts (top) and queue length (bottom).
        """
        if not self.log:
            print("No events to plot yet.")
            return

        # Build time series
        times = []
        cum_arr, cum_enq, cum_drop = [], [], []
        qlens = []

        a = e = d = 0
        q = 0
        for rec in self.log:
            t = rec["time"]
            ev = rec["event"]
            if ev == "arrive":
                a += 1
            elif ev == "enqueue":
                e += 1
                q += 1
            elif ev == "drop":
                d += 1
            elif ev == "dequeue":
                q = max(0, q - 1)

            times.append(t)
            cum_arr.append(a)
            cum_enq.append(e)
            cum_drop.append(d)
            qlens.append(q)

        # Make them look like step curves by duplicating points (left stairs)
        def stepify(x: list[float], y: list[int]):
            xs, ys = [], []
            prev_x = x[0] if x else 0.0
            prev_y = 0
            for xi, yi in zip(x, y):
                xs.extend([prev_x, xi])
                ys.extend([prev_y, prev_y])
                prev_x = xi
                prev_y = yi
            # final point
            xs.append(prev_x)
            ys.append(prev_y)
            return xs, ys

        tsa, ya = stepify(times, cum_arr)
        tse, ye = stepify(times, cum_enq)
        tsd, yd = stepify(times, cum_drop)
        tsq, yq = stepify(times, qlens)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # Top: cumulative counts
        ax1.plot(tsa, ya, label="Arrivals")
        ax1.plot(tse, ye, label="Enqueued")
        ax1.plot(tsd, yd, label="Dropped")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Cumulative count")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: queue length
        ax2.plot(tsq, yq, label="Queue length")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Queue length")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()


# ===================== KERNEL-SERVER QUEUING LAYER =====================

class PoissonKernelServer:
    """
    Single-server system:
      - Arrivals: Poisson (handled internally)
      - Server holds exactly one running 'kernel' at a time.
      - When server is idle and queue is non-empty, it starts a new kernel
        with up to `kernel_capacity` processes from the head of the queue.
      - Kernel service time is from:
          * service_time_fn(batch)  [if provided], or
          * Exp(mu)                 [if mu > 0 provided]
      - Collects throughput and waiting-time stats.

    You can reuse PoissonArrivalSimulator's arrival mechanics by delegating
    to it internally, or use this as a replacement. Here we integrate both.
    """
    def __init__(
        self,
        rate: float,
        process_pool: ProcessPool,
        add_prob: float = 1.0,
        seed: int | None = None,
        kernel_capacity: int = 5,
        mu: float | None = None,
        service_time_fn: callable | None = None,
    ):
        if service_time_fn is None and not (mu and mu > 0):
            raise ValueError("Provide either service_time_fn or a positive mu for Exp service.")
        if kernel_capacity <= 0:
            raise ValueError("kernel_capacity must be >= 1")

        # Arrival side (same as before)
        self.arr = PoissonArrivalSimulator(rate=rate, process_pool=process_pool,
                                           add_prob=add_prob, seed=seed)
        # Service side
        self.kernel_capacity = kernel_capacity
        self.mu = mu
        self.service_time_fn = service_time_fn
        self.rng = self.arr.rng  # shared RNG

        # Server state
        self.server_busy = False
        self.current_batch: list[Job] = []
        self.next_departure: float | None = None

        # Stats
        self.completed_jobs: list[Job] = []
        self.waiting_times: list[float] = []   # per job
        self.start_times: dict[int, float] = {}  # job_id -> service start time

        # Log extends arrival log with 'start_kernel' and 'finish_kernel'
        # Reuse self.arr.log for a unified event history.

    # ---- service time models ----
    def _exp_service(self) -> float:
        # Mean service time = 1/mu
        return self.rng.expovariate(self.mu)

    def _kernel_service_time(self, batch: list[Job]) -> float:
        if self.service_time_fn is not None:
            return float(self.service_time_fn(batch))
        return self._exp_service()

    # ---- helpers ----
    def _maybe_start_kernel(self):
        """If server idle and queue has jobs, start a kernel with up to K jobs."""
        if self.server_busy:
            return
        if not len(self.arr.queue):
            return

        # Start a batch
        K = min(self.kernel_capacity, len(self.arr.queue))
        batch = []
        for _ in range(K):
            job = self.arr.pop()  # pops from queue and logs 'dequeue'
            batch.append(job)

        self.current_batch = batch
        t = self.arr.t

        # record start time and waiting time (start - arrival)
        for job in batch:
            self.start_times[job.id] = t
            self.waiting_times.append(t - job.arrival_time)

        st = self._kernel_service_time(batch)
        self.next_departure = t + st
        self.server_busy = True
        self.arr.log.append({
            "time": t,
            "event": "start_kernel",
            "job_ids": [j.id for j in batch],
            "labels": [j.process_label for j in batch],
            "service_time": st
        })

    def _finish_kernel(self):
        """Complete the running kernel, mark jobs as done, free server."""
        if not self.server_busy or self.next_departure is None:
            return
        t = self.next_departure
        self.arr.t = t  # advance time to departure instant

        # mark completion
        for job in self.current_batch:
            self.completed_jobs.append(job)
        self.arr.log.append({
            "time": t,
            "event": "finish_kernel",
            "job_ids": [j.id for j in self.current_batch],
            "labels": [j.process_label for j in self.current_batch],
        })

        # reset server
        self.current_batch = []
        self.next_departure = None
        self.server_busy = False

    # ---- core advancement ----
    def advance_to(self, t_end: float):
        """
        Event-driven advancement to absolute time t_end.
        Competes arrivals vs (next) departure.
        """
        if t_end < self.arr.t:
            raise ValueError("t_end must be >= current time")

        # Loop over next imminent event (arrival or departure), up to t_end
        while True:
            next_arrival = self.arr._next_arrival
            next_departure = self.next_departure

            # Choose the earliest event before or at t_end
            candidates = []
            if next_arrival is not None and next_arrival <= t_end:
                candidates.append(("arrival", next_arrival))
            if next_departure is not None and next_departure <= t_end:
                candidates.append(("departure", next_departure))

            if not candidates:
                # no event before t_end: just advance time
                self.arr.t = t_end
                break

            ev_type, ev_time = min(candidates, key=lambda x: x[1])

            if ev_type == "arrival":
                # process all arrivals exactly at ev_time using arrival engine
                self.arr.advance_to(ev_time)  # enqueues & logs arrivals
                # after arrivals, try to start kernel if idle
                self._maybe_start_kernel()

            else:  # departure
                # jump to departure and finish kernel
                self.arr.t = ev_time
                self._finish_kernel()
                # after finishing, immediately try to start next kernel if queue nonempty
                self._maybe_start_kernel()

        # After time limit reached, do nothing (partially processed kernel can be running)

    def run(self, duration: float):
        if duration < 0:
            raise ValueError("duration must be >= 0")
        self.advance_to(self.arr.t + duration)

    # ---- metrics ----
    def metrics(self) -> dict:
        t = self.arr.t if self.arr.t > 0 else 1.0
        completed = len(self.completed_jobs)
        avg_wait = (sum(self.waiting_times) / len(self.waiting_times)) if self.waiting_times else 0.0
        return {
            "time": self.arr.t,
            "queue_len": len(self.arr),
            "server_busy": self.server_busy,
            "completed": completed,
            "throughput": completed / t,        # per unit time
            "avg_waiting_time": avg_wait,       # per process
            "total_arrivals": self.arr.total_arrivals,
            "total_enqueued": self.arr.total_enqueued,
            "dropped": self.arr.dropped,
        }

    def plot_log(self, show: bool = True, figsize=(9, 7), save_path: str | None = None) -> None:
        """Plot arrivals, queue length, and kernel start/finish markers."""
        # Reuse the arrival plots first (don’t show yet)
        self.arr.plot_log(show=False, figsize=figsize)

        fig = plt.gcf()
        ax1, ax2 = fig.axes  # two axes: cumulative counts, queue length

        # Overlay kernel start/finish markers
        starts = [r for r in self.arr.log if r.get("event") == "start_kernel"]
        ends   = [r for r in self.arr.log if r.get("event") == "finish_kernel"]

        ax1.vlines([r["time"] for r in starts], *ax1.get_ylim(),
                linestyles="dashed", linewidth=1, label="kernel start")
        ax1.vlines([r["time"] for r in ends], *ax1.get_ylim(),
                linestyles="dotted", linewidth=1, label="kernel finish")

        # De-duplicate legend labels
        handles, labels = ax1.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax1.legend(uniq.values(), uniq.keys())

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

def generate_random_kernel(num_proc,max_shot):
    """
    Randomly generate a kernel with specified number of processes.
    All those processes are selected randomly from the small QASM benchmark set.
    Args:
        num_proc (int): Number of processes in the kernel.
        max_shot (int): Maximum number of shots for each process.
    """

    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], 
                [0,5], [1,6], [2,7], [3,8], [4,9],
                [5,6], [6,7],[7,8],[8,9],
                [5,10], [6, 11], [7, 12], [8, 13], [9, 14],
                [10,11],[11,12], [12,13], [13,14], 
                [10,15], [11,16], [12,17], [13,18], [14,19],
                [15,16], [16,17], [17,18], [18,19]]  


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    name_list = []
    for pid in range(1, num_proc + 1):
        label_id = np.random.randint(1, 26)  # Assuming there are 25 benchmark QASM files
        label_name = label_name_map[label_id]
        name_list.append(label_name)
        file_path = f"C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\smallqasm\\{label_name}.qasm"
        with open(file_path, "r") as file:
            qasm_code = file.read()   
        shots = np.random.randint(100, max_shot + 1)
        proc_instance = parse_qasm_instruction(shots=shots, process_ID=pid, instruction_str=qasm_code)
        kernel_instance.add_process(proc_instance)

    virtual_hardware = virtualHardware(qubit_number=20, error_rate=0.001,edge_list=COUPLING)
    print("Generated processes:", name_list)
    return kernel_instance, virtual_hardware



if __name__ == "__main__":
    # 1) Build your pool as before
    kernel, vhw = generate_random_kernel(num_proc=10, max_shot=2000)
    pool = ProcessPool(kernel)

    # 2) Arrival rate λ=2; service rate μ=0.5  (mean service time = 2.0 time units per kernel)
    sys = PoissonKernelServer(
        rate=2.0,
        process_pool=pool,
        add_prob=1.0,
        seed=7,
        kernel_capacity=5,
        mu=0.5,                 # <- exponential service
        service_time_fn=None    # <- not needed when using mu
    )

    # --- run the simulation ---
    sys.run(30.0)

    # --- metrics ---
    print("\n=== Metrics ===")
    for k, v in sys.metrics().items():
        print(f"{k:>18s}: {v}")

    # --- plotting ---
    # If you kept `matplotlib.use("Agg")`, nothing will display; so also save to a file.
    sys.plot_log(show=True, save_path="arrival_kernel_timeline.png")
    print("Saved timeline plot to: arrival_kernel_timeline.png")

    # --- arrival log: print and save ---
    arrivals = [r for r in sys.arr.log if r.get("event") == "arrive"]
    print(f"\n=== Arrival events ({len(arrivals)}) ===")
    for i, rec in enumerate(arrivals, 1):
        print(f"{i:03d}  t={rec['time']:.6f}  event={rec['event']}")

    # Save arrivals to CSV (stdlib only)
    import csv
    with open("arrival_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "event"])
        writer.writeheader()
        for rec in arrivals:
            writer.writerow({"time": rec["time"], "event": rec["event"]})
    print("Saved arrival log to: arrival_log.csv")

    # Optional: also save full log (all events) for debugging
    with open("event_log.csv", "w", newline="") as f:
        # collect all keys that appear in any record
        import itertools
        all_keys = sorted(set(itertools.chain.from_iterable(r.keys() for r in sys.arr.log)))
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for rec in sys.arr.log:
            writer.writerow({k: rec.get(k) for k in all_keys})
    print("Saved full event log to: event_log.csv")