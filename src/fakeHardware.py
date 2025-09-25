from qiskit.providers.fake_provider import GenericBackendV2  # lives here
from qiskit import QuantumCircuit, transpile
# visualize_layout.py
from qiskit.visualization import plot_coupling_map
import matplotlib
matplotlib.use("Agg")              # call BEFORE importing pyplot
import matplotlib.pyplot as plt
from qiskit.transpiler import CouplingMap
from scheduler import *
from process import process
import numpy as np
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error,pauli_error



def construct_10_qubit_hardware():
    NUM_QUBITS = 10
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain
    BASIS = ["cx", "id", "rz", "sx", "x"]  # add more *only* if truly native
    SINGLE_QUBIT_GATE_LENGTH_NS = 32       # example: 0.222 ns timestep
    SINGLE_QUBIT_GATE_LENGTH_NS = 88       # example: 0.222 ns timestep
    READOUT_LENGTH_NS = 2584     # example measurement timestep

    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,         # optional
        coupling_map=COUPLING,     # strongly recommended
        control_flow=True,        # set True if you want dynamic circuits            
        seed=1234,                 # reproducible auto-generated props
        noise_info=True            # attach plausible noise/durations
    )

    return backend    


def get_10_qubit_hardware_coords() -> list[tuple[float, float]]:
    edge_length = 1
    coords = [ ]
    for i in range(10):
        if i<5:
            coords.append( (float(i*edge_length), 0.0) )
        else:
            coords.append( (float((i-5)*edge_length), -edge_length))
    return coords



def construct_20_qubit_hardware():
    NUM_QUBITS = 20
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], 
                [0,5], [1,6], [2,7], [3,8], [4,9],
                [5,6], [6,7],[7,8],[8,9],
                [5,10], [6, 11], [7, 12], [8, 13], [9, 14],
                [10,11],[11,12], [12,13], [13,14], 
                [10,15], [11,16], [12,17], [13,18], [14,19],
                [15,16], [16,17], [17,18], [18,19]]  # linear chain
    BASIS = ["cx", "id", "rz", "sx", "x"]  # add more *only* if truly native
    SINGLE_QUBIT_GATE_LENGTH_NS = 32       # example: 0.222 ns timestep
    SINGLE_QUBIT_GATE_LENGTH_NS = 88       # example: 0.222 ns timestep
    READOUT_LENGTH_NS = 2584     # example measurement timestep

    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,         # optional
        coupling_map=COUPLING,     # strongly recommended
        control_flow=True,        # set True if you want dynamic circuits            
        seed=1234,                 # reproducible auto-generated props
        noise_info=True            # attach plausible noise/durations
    )

    return backend    


def get_20_qubit_hardware_coords() -> list[tuple[float, float]]:
    edge_length = 1
    coords = [ ]
    for i in range(20):
        if i<5:
            coords.append( (float(i*edge_length), 0.0) )
        elif i<10:
            coords.append( (float((i-5)*edge_length), -edge_length))
        elif i<15:
            coords.append( (float((i-10)*edge_length), -2*edge_length))
        else:
            coords.append( (float((i-15)*edge_length), -3*edge_length))
    return coords


def plot_process_schedule_on_20_qubit_hardware(coupling_edges: list[list[int]],
                               syndrome_qubit_history: dict[int, list[int]],
                               process_list: list,
                               out_png: str = "hardware_mapping.png",
                               figsize=(12, 4.5)):      # a bit shorter is fine; width stays large
    coords = get_20_qubit_hardware_coords()
    cm = CouplingMap(coupling_edges)

    pairs = cm.get_edges()
    undirected = sorted(set(tuple(sorted((a, b))) for a, b in pairs))

    # Better layout engine than tight_layout for drawings
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # edges
    for a, b in undirected:
        xa, ya = coords[a]; xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5, alpha=0.7, color="#20324d")

    # nodes
    xs = [xy[0] for xy in coords]; ys = [xy[1] for xy in coords]
    ax.scatter(xs, ys, s=620, color="#0b1e3f", zorder=3)

    # indices
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white",
                zorder=4, clip_on=False)  # avoid text clipping

    # process outlines
    colors = plt.cm.tab10(np.linspace(0, 1, len(process_list)))
    for proc, color in zip(process_list, colors):
        vaddress = proc.get_virtual_data_addresses()
        label_physical_map = {}
        for vaddr in vaddress:
            phys = proc.get_data_qubit_virtual_hardware_mapping(vaddr)
            if phys is not None:
                label_physical_map[phys] = str(vaddr)

        for phys, label in label_physical_map.items():
            x, y = coords[phys]
            ax.scatter([x], [y], s=780, facecolors="none", edgecolors=color,
                       linewidths=2.6, zorder=5)
            ax.text(x, y + 0.3, label, ha="center", va="bottom",
                    fontsize=6, color=color, weight="bold", zorder=6, clip_on=False)

    # syndrome labels
    for phys in syndrome_qubit_history.keys():
        x, y = coords[phys]
        vaddress_list = syndrome_qubit_history[phys]
        label = ",".join([str(vaddr) for vaddr in vaddress_list[:2]])
        if len(vaddress_list) > 2:
            label += ",..."
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors="orange",
                   linewidths=2.6, zorder=5)
        ax.text(x, y + 0.2, label, ha="center", va="bottom",
                fontsize=6, color="blue", weight="bold", zorder=6, clip_on=False)

    ax.set_aspect("equal", adjustable="datalim")

    # ---- Key fix: add padding around data limits ----
    pad = 0.75                     # increase if labels still feel tight
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.axis("off")

    # Avoid overly tight cropping; keep a little page margin
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_process_schedule_on_10_qubit_hardware(coupling_edges: list[list[int]],
                               syndrome_qubit_history: dict[int, list[int]],
                               process_list: list,
                               out_png: str = "hardware_mapping.png",
                               figsize=(12, 4.5)):      # a bit shorter is fine; width stays large
    coords = get_10_qubit_hardware_coords()
    cm = CouplingMap(coupling_edges)

    pairs = cm.get_edges()
    undirected = sorted(set(tuple(sorted((a, b))) for a, b in pairs))

    # Better layout engine than tight_layout for drawings
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # edges
    for a, b in undirected:
        xa, ya = coords[a]; xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5, alpha=0.7, color="#20324d")

    # nodes
    xs = [xy[0] for xy in coords]; ys = [xy[1] for xy in coords]
    ax.scatter(xs, ys, s=620, color="#0b1e3f", zorder=3)

    # indices
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white",
                zorder=4, clip_on=False)  # avoid text clipping

    # process outlines
    colors = plt.cm.tab10(np.linspace(0, 1, len(process_list)))
    for proc, color in zip(process_list, colors):
        vaddress = proc.get_virtual_data_addresses()
        label_physical_map = {}
        for vaddr in vaddress:
            phys = proc.get_data_qubit_virtual_hardware_mapping(vaddr)
            if phys is not None:
                label_physical_map[phys] = str(vaddr)

        for phys, label in label_physical_map.items():
            x, y = coords[phys]
            ax.scatter([x], [y], s=780, facecolors="none", edgecolors=color,
                       linewidths=2.6, zorder=5)
            ax.text(x, y + 0.1, label, ha="center", va="bottom",
                    fontsize=8, color=color, weight="bold", zorder=6, clip_on=False)

    # syndrome labels
    for phys in syndrome_qubit_history.keys():
        x, y = coords[phys]
        vaddress_list = syndrome_qubit_history[phys]
        label = ",".join([str(vaddr) for vaddr in vaddress_list[:2]])
        if len(vaddress_list) > 2:
            label += ",..."
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors="orange",
                   linewidths=2.6, zorder=5)
        ax.text(x, y + 0.1, label, ha="center", va="bottom",
                fontsize=8, color="blue", weight="bold", zorder=6, clip_on=False)

    ax.set_aspect("equal", adjustable="datalim")

    # ---- Key fix: add padding around data limits ----
    pad = 0.75                     # increase if labels still feel tight
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.axis("off")

    # Avoid overly tight cropping; keep a little page margin
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)




def construct_fake_ibm_pittsburgh():
    NUM_QUBITS = 156


    # Directed edges (bidirectional 0<->1 and 1->2)
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]

    BASIS = ["cz","id","rx","rz","rzz","sx","x"]  # add more *only* if truly native

    SINGLE_QUBIT_GATE_LENGTH_NS = 32       # example: 0.222 ns timestep
    SINGLE_QUBIT_GATE_LENGTH_NS = 88       # example: 0.222 ns timestep
    READOUT_LENGTH_NS = 2584     # example measurement timestep


    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,         # optional
        coupling_map=COUPLING,     # strongly recommended
        control_flow=True,        # set True if you want dynamic circuits            
        seed=1234,                 # reproducible auto-generated props
        noise_info=True            # attach plausible noise/durations
    )

    return backend


def pittsburgh_qubit_coords() -> list[tuple[float, float]]:
    coords = [(0.0, 0.0)] * 156

    # Long rows: each has 16 nodes, at x=0..15
    long_starts = [0, 20, 40, 60, 80, 100, 120, 140]
    for r, start in enumerate(long_starts):
        y = -2.0 * r
        for k in range(16):
            coords[start + k] = (float(k), y)

    # Short rows: each has 4 nodes, alternating column anchors
    short_starts = [16, 36, 56, 76, 96, 116, 136]
    anchors_odd  = [3, 7, 11, 15]  # short rows 1,3,5,7
    anchors_even = [1, 5, 9, 13]   # short rows 2,4,6
    for s, start in enumerate(short_starts):
        y = -(2.0 * s + 1.0)
        xs = anchors_odd if (s % 2 == 0) else anchors_even
        for j, x in enumerate(xs):
            coords[start + j] = (float(x), y)

    return coords



def plot_pittsburgh_with_annotations(coupling_edges, selected=None, custom_labels=None, figsize=(11, 9)):
    coords = pittsburgh_qubit_coords()
    cm = CouplingMap(coupling_edges)
    selected = selected or []
    custom_labels = custom_labels or {}

    # Try Graphviz first
    try:
        fig = plot_coupling_map(
            coupling_map=cm,
            qubit_coordinates=coords,
            label_qubits=True,
            plot_directed=False,
            figsize=figsize,
        )
        ax = fig.axes[0]
        for q in selected:
            x, y = coords[q]
            ax.scatter([x],[y], s=650, facecolors="none", edgecolors="crimson", linewidths=2.6, zorder=5)
            ax.text(x, y+0.36, custom_labels.get(q, f"Q{q}"),
                    ha="center", va="bottom", fontsize=11, color="crimson", weight="bold", zorder=6)
        plt.tight_layout(); plt.show()
        return fig
    except Exception as e:
        print(f"[Graphviz not available or plotting failed: {e}] Falling back to Matplotlib.")

    # Matplotlib fallback (undirected look)
    pairs = cm.get_edges()
    undirected = set(tuple(sorted((a, b))) for a, b in pairs)

    fig, ax = plt.subplots(figsize=figsize)
    for a, b in undirected:
        xa, ya = coords[a]; xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5, alpha=0.7)
    xs = [xy[0] for xy in coords]; ys = [xy[1] for xy in coords]
    ax.scatter(xs, ys, s=620, color="#0b1e3f", zorder=3)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white", zorder=4)
    for q in selected:
        x, y = coords[q]
        ax.scatter([x],[y], s=780, facecolors="none", edgecolors="crimson", linewidths=2.6, zorder=5)
        ax.text(x, y+0.38, custom_labels.get(q, f"Q{q}"),
                ha="center", va="bottom", fontsize=10, color="crimson", weight="bold", zorder=6)
    ax.set_aspect("equal"); ax.axis("off"); plt.tight_layout(); plt.show()
    return fig




def plot_process_schedule_on_pittsburgh(coupling_edges: list[list[int]],
                               syndrome_qubit_history: list[int],
                               process_list: list[process],
                               out_png: str = "hardware_mapping.png",
                               figsize=(11, 9)):
    """
    Plot the layout and mapping of multiple processes on the Pittsburgh hardware.
    """
    coords = pittsburgh_qubit_coords()
    cm = CouplingMap(coupling_edges)

    # undirected edges for a clean look
    pairs = cm.get_edges()
    undirected = sorted(set(tuple(sorted((a, b))) for a, b in pairs))

    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for a, b in undirected:
        xa, ya = coords[a]; xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5, alpha=0.7, color="#20324d")

    # nodes
    xs = [xy[0] for xy in coords]; ys = [xy[1] for xy in coords]
    ax.scatter(xs, ys, s=620, color="#0b1e3f", zorder=3)

    # small index inside each node (physical index)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white", zorder=4)


    # --- give each process a unique color ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(process_list)))  # up to 10 distinct
    for proc, color in zip(process_list, colors):
        vaddress = proc.get_virtual_data_addresses()
        label_physical_map = {}
        for vaddr in vaddress:
            phys = proc.get_data_qubit_virtual_hardware_mapping(vaddr)
            if phys is not None:
                label_physical_map[phys] = str(vaddr)

        for phys, label in label_physical_map.items():
            x, y = coords[phys]
            ax.scatter([x], [y], s=780, facecolors="none", edgecolors=color,
                       linewidths=2.6, zorder=5)
            ax.text(x, y + 0.38, label , ha="center", va="bottom",
                    fontsize=5, color=color, weight="bold", zorder=6)


    for phys in syndrome_qubit_history.keys():
        x, y = coords[phys]
        vaddress_list = syndrome_qubit_history[phys]
        label=",".join([str(vaddr) for vaddr in vaddress_list[:2]])
        if len(vaddress_list)>2:
            label+=",..."
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors="orange", linewidths=2.6, zorder=5)
        ax.text(x, y + 0.38, label, ha="center", va="bottom",
                fontsize=3, color="blue", weight="bold", zorder=6)



    ax.set_aspect("equal"); ax.axis("off"); plt.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mapping_on_pittsburgh(coupling_edges: list[list[int]],
                               virt2phys: dict[int, int],
                               out_png: str = "hardware_mapping.png",
                               figsize=(11, 9)):
    """
    Draw the coupling graph with Matplotlib and annotate each *physical* node
    with the logical index that maps to it (e.g. '0' near phys 5 if 0->5).
    """
    coords = pittsburgh_qubit_coords()
    cm = CouplingMap(coupling_edges)

    # undirected edges for a clean look
    pairs = cm.get_edges()
    undirected = sorted(set(tuple(sorted((a, b))) for a, b in pairs))

    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for a, b in undirected:
        xa, ya = coords[a]; xb, yb = coords[b]
        ax.plot([xa, xb], [ya, yb], linewidth=1.5, alpha=0.7, color="#20324d")

    # nodes
    xs = [xy[0] for xy in coords]; ys = [xy[1] for xy in coords]
    ax.scatter(xs, ys, s=620, color="#0b1e3f", zorder=3)

    # small index inside each node (physical index)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="white", zorder=4)

    # annotations: for every mapping logical->physical, write the logical near the physical node
    for v, p in virt2phys.items():
        x, y = coords[p]
        ax.scatter([x], [y], s=780, facecolors="none", edgecolors="crimson", linewidths=2.6, zorder=5)
        ax.text(x, y + 0.38, f"{v}", ha="center", va="bottom",
                fontsize=10, color="crimson", weight="bold", zorder=6)

    ax.set_aspect("equal"); ax.axis("off"); plt.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------- dynamic circuit example ----------
def build_dynamic_circuit() -> QuantumCircuit:
    """
    3-qubit toy circuit with:
      - mid-circuit measure of q1 into c0
      - reset of q1
      - feedback: if c0==1 then X on q1
      - another mid-circuit measure of q2 into c1 with feedback Z on q0 if 1
      - final readout of all qubits into c2,c3,c4
    """
    qc = QuantumCircuit(3, 5)   # 3 qubits, 5 classical bits
    # your base ops
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # mid-circuit measurement + reset + feedback
    qc.measure(1, 0)            # c0 <= measure(q1)
    qc.reset(1)
    with qc.if_test((qc.clbits[0], 1)):  # if c0 == 1
        qc.x(1)

    # another measurement + feedback path
    qc.measure(2, 1)            # c1 <= measure(q2)
    with qc.if_test((qc.clbits[1], 1)):  # if c1 == 1
        qc.z(0)

    # finish and read out all three data qubits
    qc.cx(1, 2)
    qc.measure(0, 2)            # c2
    qc.measure(1, 3)            # c3
    qc.measure(2, 4)            # c4
    return qc


# ---------- dynamic circuit (15 qubits) ----------
def build_dynamic_circuit_15() -> QuantumCircuit:
    """
    15-qubit dynamic circuit with multiple mid-circuit measurements,
    resets, and measurement-conditioned feedback.
    """
    n = 15
    c_total = 2 * n                      # plenty of classical bits (first half mid-circ, last half final RO)
    qc = QuantumCircuit(n, c_total)

    # --- Prepare a long entangled chain ---
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    # --- Mid-circuit measure/reset on a sparse subset with feedback to neighbors ---
    meas1_qubits = [2, 5, 8, 11, 14]     # 5 taps along the chain
    for k, q in enumerate(meas1_qubits):
        qc.measure(q, k)                 # store in c[k]
        qc.reset(q)                      # recycle that qubit
        with qc.if_test((qc.clbits[k], 1)):
            # simple feedback: kick the right neighbor (or the last qubit if at end)
            target = min(q + 1, n - 1)
            qc.x(target)

    # --- A second measurement/feedback layer to show multiple dynamic regions ---
    meas2_qubits = [3, 9, 12]
    offset = len(meas1_qubits)
    for j, q in enumerate(meas2_qubits):
        bit = offset + j                 # c[offset + j]
        qc.measure(q, bit)
        with qc.if_test((qc.clbits[bit], 1)):
            qc.z(0)                      # global feedback to the chain head

    # --- Some more gates post-feedback (mix things up a bit) ---
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)

    # --- Final measurements into the *second half* of the classical register ---
    final_start = c_total - n
    for q in range(n):
        qc.measure(q, final_start + q)

    return qc





# ---------- 1) save abstract circuit as PNG ----------
def save_circuit_png(qc: QuantumCircuit, path: str = "abstract_circuit.png"):
    fig = qc.draw(output="mpl", fold=-1)     # pure Matplotlib
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def extract_virt2phys(tqc) -> dict[int, int]:
    """
    Return {logical_qubit_index: physical_qubit_index} from a transpiled circuit,
    working across Qiskit versions.
    """
    L = getattr(tqc, "layout", None)
    if L is None:
        raise ValueError("Transpiled circuit has no layout")

    # 1) Newer API: final_index_layout() may be a dict OR a list
    if hasattr(L, "final_index_layout") and callable(L.final_index_layout):
        fil = L.final_index_layout()
        if isinstance(fil, dict):
            # already {logical:int -> physical:int}
            return {int(k): int(v) for k, v in fil.items()}
        if isinstance(fil, (list, tuple)):
            # list: position = logical index; value = physical int (or None)
            return {i: int(p) for i, p in enumerate(fil) if p is not None}

    # 2) Fallback: use the final Layout object
    if hasattr(L, "final_layout"):
        FL = L.final_layout()
        # Try virtual->physical first
        if hasattr(FL, "get_virtual_bits"):
            v2p = {}
            for vq, phys in FL.get_virtual_bits().items():  # {Qubit->int}
                idx = getattr(vq, "_index", getattr(vq, "index", None))
                if idx is not None:
                    v2p[int(idx)] = int(phys)
            if v2p:
                return v2p
        # Or physical->virtual
        if hasattr(FL, "get_physical_bits"):
            p2v = {}
            for phys, vq in FL.get_physical_bits().items():  # {int->Qubit}
                idx = getattr(vq, "_index", getattr(vq, "index", None))
                if idx is not None:
                    p2v[int(idx)] = int(phys)
            if p2v:
                return p2v

    # 3) Last-resort: derive from wires (not ideal, but works when mapped circuit is packed)
    try:
        # If the transpiled circuit has been permuted so that qubit order equals physical indices:
        return {i: tqc.find_bit(tqc.qubits[i]).index for i in range(len(tqc.qubits))}
    except Exception:
        pass

    raise RuntimeError("Could not extract logical→physical mapping from layout")





class HardwareManager:
    def __init__(self, backend: GenericBackendV2):
        self._backend = backend
        self._num_qubits = backend.num_qubits
        self._coupling_map = backend.coupling_map
        self._basis_gates = backend.basis_gates
        # You can access more properties if needed, e.g. gate durations, noise model, etc.


        self._syndrome_qubits = None
        self._data_qubits = None





    def get_qubit_num(self) -> int:
        return self._num_qubits

    def get_coupling_map(self) -> list[list[int]]:
        return self._coupling_map

    def get_basis_gates(self) -> list[str]:
        return self._basis_gates

    def get_backend(self) -> GenericBackendV2:
        return self._backend






def generate_example_ppt10_on_10_qubit_device():
    vdata1 = virtualSpace(size=3, label="vdata1")
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=2, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,1)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))
    proc1.add_instruction(Instype.H, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0),vsyn1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))


    vdata2 = virtualSpace(size=3, label="vdata2")
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=2, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,1)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3,processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))
    proc2.add_instruction(Instype.H, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0), vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3 ,processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))

    # ---- proc3 ----
    vdata3 = virtualSpace(size=3, label="vdata3")
    vdata3.allocate_range(0,2)
    vsyn3 = virtualSpace(size=2, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0,1)
    proc3 = process(processID=3, start_time=0, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0),vdata3.get_address(1),vdata3.get_address(2)],size=3,processID=3))
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0),vsyn3.get_address(1)],size=2,processID=3))
    proc3.add_instruction(Instype.H, [vsyn3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vsyn3.get_address(0),vsyn3.get_address(1)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0), vsyn3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(1), vsyn3.get_address(1)])
    proc3.add_instruction(Instype.CNOT, [vsyn3.get_address(0), vsyn3.get_address(1)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(1)])
    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0),vdata3.get_address(1),vdata3.get_address(2)],size=3 ,processID=3))
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0),vsyn3.get_address(1)],size=2,processID=3))



    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)

    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware



def generate_simples_example_for_test_2():
    vdata1 = virtualSpace(size=5, label="vdata1")    
    vdata1.allocate_range(0,4)
    vsyn1 = virtualSpace(size=3, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,2)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)],size=5,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 1 syndrome qubit
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(1)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(2)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(3)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(4)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1),vdata1.get_address(3)])   
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vdata1.get_address(4)])
    proc1.add_instruction(Instype.X, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation    
    proc1.add_instruction(Instype.X, [vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])  # Measure operation    
    proc1.add_instruction(Instype.X, [vsyn1.get_address(2)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(2)])  # Measure operation    
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation        
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=3,processID=1))  # Allocate 2 syndrome qubits

    vdata2 = virtualSpace(size=3, label="vdata2")    
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=3, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,2)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 1 syndrome qubit
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.X, [vdata2.get_address(1)])
    proc2.add_instruction(Instype.X, [vdata2.get_address(2)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])  # Measure operation    
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation  
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])  # Measure operation      
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(2),vsyn2.get_address(2)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(2)])  # Measure operation           
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=3,processID=2))  # Allocate 2 syndrome qubits




    vdata3 = virtualSpace(size=1, label="vdata3")    
    vdata3.allocate_range(0,0)
    vsyn3 = virtualSpace(size=1, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0,0)
    proc3 = process(processID=3, start_time=0, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0)],size=1,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=1,processID=3))  # Allocate 1 syndrome qubit
    proc3.add_instruction(Instype.X, [vdata3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0),vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vdata3.get_address(0)])  # Measure operation    
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])  # Measure operation        
    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0)],size=1 ,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=1,processID=3))  # Allocate 2 syndrome qubits


    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], 
                [0,5], [1,6], [2,7], [3,8], [4,9],
                [5,6], [6,7],[7,8],[8,9],
                [5,10], [6, 11], [7, 12], [8, 13], [9, 14],
                [10,11],[11,12], [12,13], [13,14], 
                [10,15], [11,16], [12,17], [13,18], [14,19],
                [15,16], [16,17], [17,18], [18,19]]  # linear chain

    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)

    virtual_hardware = virtualHardware(qubit_number=20, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware



def generate_simples_example_for_test():
    vdata1 = virtualSpace(size=1, label="vdata1")    
    vdata1.allocate_range(0,0)
    vsyn1 = virtualSpace(size=1, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,0)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0)],size=1,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=1,processID=1))  # Allocate 1 syndrome qubit
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.X, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.H, [vdata1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0),vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vdata1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation        
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0)],size=1 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0)],size=1,processID=1))  # Allocate 2 syndrome qubits

    vdata2 = virtualSpace(size=1, label="vdata2")    
    vdata2.allocate_range(0,0)
    vsyn2 = virtualSpace(size=1, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,0)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0)],size=1,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=1,processID=2))  # Allocate 1 syndrome qubit
    proc2.add_instruction(Instype.X, [vdata2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0),vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vdata2.get_address(0)])  # Measure operation    
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation        
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0)],size=1 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0)],size=1,processID=2))  # Allocate 2 syndrome qubits




    vdata3 = virtualSpace(size=1, label="vdata3")    
    vdata3.allocate_range(0,0)
    vsyn3 = virtualSpace(size=1, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0,0)
    proc3 = process(processID=3, start_time=0, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0)],size=1,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=1,processID=3))  # Allocate 1 syndrome qubit
    proc3.add_instruction(Instype.X, [vdata3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0),vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vdata3.get_address(0)])  # Measure operation    
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])  # Measure operation        
    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0)],size=1 ,processID=3))  # Allocate 2 data qubits
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0)],size=1,processID=3))  # Allocate 2 syndrome qubits


    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)

    virtual_hardware = virtualHardware(qubit_number=10, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware



def generate_example_ppt_real():
    vdata1 = virtualSpace(size=3, label="vdata1")
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=2, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,1)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))  # Allocate 2 syndrome qubits
    proc1.add_instruction(Instype.H, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0),vsyn1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])  # CNOT operation
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0), vsyn1.get_address(1)])  # CNOT operation    
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])  # Measure operation
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])  # Measure operation
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))  # Allocate 2 data qubits
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))  # Allocate 2 syndrome qubits
    proc1.construct_qiskit_diagram()


    vdata2 = virtualSpace(size=3, label="vdata2")
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=2, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,1)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))  # Allocate 2 syndrome qubits
    proc2.add_instruction(Instype.H, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])  # CNOT operation
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])  # CNOT operation
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0), vsyn2.get_address(1)])  # CNOT operation    
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])  # Measure operation
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])  # Measure operation
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3 ,processID=2))  # Allocate 2 data qubits
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))  # Allocate 2 syndrome qubits


    proc2.construct_qiskit_diagram()


    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=156, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware







def generate_example_ppt10_real():
    vdata1 = virtualSpace(size=3, label="vdata1")
    vdata1.allocate_range(0,2)
    vsyn1 = virtualSpace(size=2, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0,1)
    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3,processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))
    proc1.add_instruction(Instype.H, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0),vsyn1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(0), vsyn1.get_address(0)])
    proc1.add_instruction(Instype.CNOT, [vdata1.get_address(1), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.CNOT, [vsyn1.get_address(0), vsyn1.get_address(1)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(0)])
    proc1.add_instruction(Instype.MEASURE, [vsyn1.get_address(1)])
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata1.get_address(0),vdata1.get_address(1),vdata1.get_address(2)],size=3 ,processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn1.get_address(0),vsyn1.get_address(1)],size=2,processID=1))
    proc1.construct_qiskit_diagram()

    vdata2 = virtualSpace(size=3, label="vdata2")
    vdata2.allocate_range(0,2)
    vsyn2 = virtualSpace(size=2, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0,1)
    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3,processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))
    proc2.add_instruction(Instype.H, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0),vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(0), vsyn2.get_address(0)])
    proc2.add_instruction(Instype.CNOT, [vdata2.get_address(1), vsyn2.get_address(1)])
    proc2.add_instruction(Instype.CNOT, [vsyn2.get_address(0), vsyn2.get_address(1)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(0)])
    proc2.add_instruction(Instype.MEASURE, [vsyn2.get_address(1)])
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata2.get_address(0),vdata2.get_address(1),vdata2.get_address(2)],size=3 ,processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn2.get_address(0),vsyn2.get_address(1)],size=2,processID=2))
    proc2.construct_qiskit_diagram()

    # ---- proc3 ----
    vdata3 = virtualSpace(size=3, label="vdata3")
    vdata3.allocate_range(0,2)
    vsyn3 = virtualSpace(size=2, label="vsyn3", is_syndrome=True)
    vsyn3.allocate_range(0,1)
    proc3 = process(processID=3, start_time=0, vdataspace=vdata3, vsyndromespace=vsyn3)
    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata3.get_address(0),vdata3.get_address(1),vdata3.get_address(2)],size=3,processID=3))
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn3.get_address(0),vsyn3.get_address(1)],size=2,processID=3))
    proc3.add_instruction(Instype.H, [vsyn3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vsyn3.get_address(0),vsyn3.get_address(1)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(0), vsyn3.get_address(0)])
    proc3.add_instruction(Instype.CNOT, [vdata3.get_address(1), vsyn3.get_address(1)])
    proc3.add_instruction(Instype.CNOT, [vsyn3.get_address(0), vsyn3.get_address(1)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(0)])
    proc3.add_instruction(Instype.MEASURE, [vsyn3.get_address(1)])
    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata3.get_address(0),vdata3.get_address(1),vdata3.get_address(2)],size=3 ,processID=3))
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn3.get_address(0),vsyn3.get_address(1)],size=2,processID=3))
    proc3.construct_qiskit_diagram()

    # ---- proc4 ----
    vdata4 = virtualSpace(size=3, label="vdata4")
    vdata4.allocate_range(0,2)
    vsyn4 = virtualSpace(size=2, label="vsyn4", is_syndrome=True)
    vsyn4.allocate_range(0,1)
    proc4 = process(processID=4, start_time=0, vdataspace=vdata4, vsyndromespace=vsyn4)
    proc4.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata4.get_address(0),vdata4.get_address(1),vdata4.get_address(2)],size=3,processID=4))
    proc4.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn4.get_address(0),vsyn4.get_address(1)],size=2,processID=4))
    proc4.add_instruction(Instype.H, [vsyn4.get_address(0)])
    proc4.add_instruction(Instype.CNOT, [vsyn4.get_address(0),vsyn4.get_address(1)])
    proc4.add_instruction(Instype.CNOT, [vdata4.get_address(0), vsyn4.get_address(0)])
    proc4.add_instruction(Instype.CNOT, [vdata4.get_address(1), vsyn4.get_address(1)])
    proc4.add_instruction(Instype.CNOT, [vsyn4.get_address(0), vsyn4.get_address(1)])
    proc4.add_instruction(Instype.MEASURE, [vsyn4.get_address(0)])
    proc4.add_instruction(Instype.MEASURE, [vsyn4.get_address(1)])
    proc4.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata4.get_address(0),vdata4.get_address(1),vdata4.get_address(2)],size=3 ,processID=4))
    proc4.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn4.get_address(0),vsyn4.get_address(1)],size=2,processID=4))
    proc4.construct_qiskit_diagram()

    # ---- proc5 ----
    vdata5 = virtualSpace(size=3, label="vdata5")
    vdata5.allocate_range(0,2)
    vsyn5 = virtualSpace(size=2, label="vsyn5", is_syndrome=True)
    vsyn5.allocate_range(0,1)
    proc5 = process(processID=5, start_time=0, vdataspace=vdata5, vsyndromespace=vsyn5)
    proc5.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata5.get_address(0),vdata5.get_address(1),vdata5.get_address(2)],size=3,processID=5))
    proc5.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn5.get_address(0),vsyn5.get_address(1)],size=2,processID=5))
    proc5.add_instruction(Instype.H, [vsyn5.get_address(0)])
    proc5.add_instruction(Instype.CNOT, [vsyn5.get_address(0),vsyn5.get_address(1)])
    proc5.add_instruction(Instype.CNOT, [vdata5.get_address(0), vsyn5.get_address(0)])
    proc5.add_instruction(Instype.CNOT, [vdata5.get_address(1), vsyn5.get_address(1)])
    proc5.add_instruction(Instype.CNOT, [vsyn5.get_address(0), vsyn5.get_address(1)])
    proc5.add_instruction(Instype.MEASURE, [vsyn5.get_address(0)])
    proc5.add_instruction(Instype.MEASURE, [vsyn5.get_address(1)])
    proc5.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata5.get_address(0),vdata5.get_address(1),vdata5.get_address(2)],size=3 ,processID=5))
    proc5.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn5.get_address(0),vsyn5.get_address(1)],size=2,processID=5))
    proc5.construct_qiskit_diagram()

    # ---- proc6 ----
    vdata6 = virtualSpace(size=3, label="vdata6")
    vdata6.allocate_range(0,2)
    vsyn6 = virtualSpace(size=2, label="vsyn6", is_syndrome=True)
    vsyn6.allocate_range(0,1)
    proc6 = process(processID=6, start_time=0, vdataspace=vdata6, vsyndromespace=vsyn6)
    proc6.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata6.get_address(0),vdata6.get_address(1),vdata6.get_address(2)],size=3,processID=6))
    proc6.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn6.get_address(0),vsyn6.get_address(1)],size=2,processID=6))
    proc6.add_instruction(Instype.H, [vsyn6.get_address(0)])
    proc6.add_instruction(Instype.CNOT, [vsyn6.get_address(0),vsyn6.get_address(1)])
    proc6.add_instruction(Instype.CNOT, [vdata6.get_address(0), vsyn6.get_address(0)])
    proc6.add_instruction(Instype.CNOT, [vdata6.get_address(1), vsyn6.get_address(1)])
    proc6.add_instruction(Instype.CNOT, [vsyn6.get_address(0), vsyn6.get_address(1)])
    proc6.add_instruction(Instype.MEASURE, [vsyn6.get_address(0)])
    proc6.add_instruction(Instype.MEASURE, [vsyn6.get_address(1)])
    proc6.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata6.get_address(0),vdata6.get_address(1),vdata6.get_address(2)],size=3 ,processID=6))
    proc6.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn6.get_address(0),vsyn6.get_address(1)],size=2,processID=6))
    proc6.construct_qiskit_diagram()

    # ---- proc7 ----
    vdata7 = virtualSpace(size=3, label="vdata7")
    vdata7.allocate_range(0,2)
    vsyn7 = virtualSpace(size=2, label="vsyn7", is_syndrome=True)
    vsyn7.allocate_range(0,1)
    proc7 = process(processID=7, start_time=0, vdataspace=vdata7, vsyndromespace=vsyn7)
    proc7.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata7.get_address(0),vdata7.get_address(1),vdata7.get_address(2)],size=3,processID=7))
    proc7.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn7.get_address(0),vsyn7.get_address(1)],size=2,processID=7))
    proc7.add_instruction(Instype.H, [vsyn7.get_address(0)])
    proc7.add_instruction(Instype.CNOT, [vsyn7.get_address(0),vsyn7.get_address(1)])
    proc7.add_instruction(Instype.CNOT, [vdata7.get_address(0), vsyn7.get_address(0)])
    proc7.add_instruction(Instype.CNOT, [vdata7.get_address(1), vsyn7.get_address(1)])
    proc7.add_instruction(Instype.CNOT, [vsyn7.get_address(0), vsyn7.get_address(1)])
    proc7.add_instruction(Instype.MEASURE, [vsyn7.get_address(0)])
    proc7.add_instruction(Instype.MEASURE, [vsyn7.get_address(1)])
    proc7.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata7.get_address(0),vdata7.get_address(1),vdata7.get_address(2)],size=3 ,processID=7))
    proc7.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn7.get_address(0),vsyn7.get_address(1)],size=2,processID=7))
    proc7.construct_qiskit_diagram()

    # ---- proc8 ----
    vdata8 = virtualSpace(size=3, label="vdata8")
    vdata8.allocate_range(0,2)
    vsyn8 = virtualSpace(size=2, label="vsyn8", is_syndrome=True)
    vsyn8.allocate_range(0,1)
    proc8 = process(processID=8, start_time=0, vdataspace=vdata8, vsyndromespace=vsyn8)
    proc8.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata8.get_address(0),vdata8.get_address(1),vdata8.get_address(2)],size=3,processID=8))
    proc8.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn8.get_address(0),vsyn8.get_address(1)],size=2,processID=8))
    proc8.add_instruction(Instype.H, [vsyn8.get_address(0)])
    proc8.add_instruction(Instype.CNOT, [vsyn8.get_address(0),vsyn8.get_address(1)])
    proc8.add_instruction(Instype.CNOT, [vdata8.get_address(0), vsyn8.get_address(0)])
    proc8.add_instruction(Instype.CNOT, [vdata8.get_address(1), vsyn8.get_address(1)])
    proc8.add_instruction(Instype.CNOT, [vsyn8.get_address(0), vsyn8.get_address(1)])
    proc8.add_instruction(Instype.MEASURE, [vsyn8.get_address(0)])
    proc8.add_instruction(Instype.MEASURE, [vsyn8.get_address(1)])
    proc8.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata8.get_address(0),vdata8.get_address(1),vdata8.get_address(2)],size=3 ,processID=8))
    proc8.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn8.get_address(0),vsyn8.get_address(1)],size=2,processID=8))
    proc8.construct_qiskit_diagram()

    # ---- proc9 ----
    vdata9 = virtualSpace(size=3, label="vdata9")
    vdata9.allocate_range(0,2)
    vsyn9 = virtualSpace(size=2, label="vsyn9", is_syndrome=True)
    vsyn9.allocate_range(0,1)
    proc9 = process(processID=9, start_time=0, vdataspace=vdata9, vsyndromespace=vsyn9)
    proc9.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata9.get_address(0),vdata9.get_address(1),vdata9.get_address(2)],size=3,processID=9))
    proc9.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn9.get_address(0),vsyn9.get_address(1)],size=2,processID=9))
    proc9.add_instruction(Instype.H, [vsyn9.get_address(0)])
    proc9.add_instruction(Instype.CNOT, [vsyn9.get_address(0),vsyn9.get_address(1)])
    proc9.add_instruction(Instype.CNOT, [vdata9.get_address(0), vsyn9.get_address(0)])
    proc9.add_instruction(Instype.CNOT, [vdata9.get_address(1), vsyn9.get_address(1)])
    proc9.add_instruction(Instype.CNOT, [vsyn9.get_address(0), vsyn9.get_address(1)])
    proc9.add_instruction(Instype.MEASURE, [vsyn9.get_address(0)])
    proc9.add_instruction(Instype.MEASURE, [vsyn9.get_address(1)])
    proc9.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata9.get_address(0),vdata9.get_address(1),vdata9.get_address(2)],size=3 ,processID=9))
    proc9.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn9.get_address(0),vsyn9.get_address(1)],size=2,processID=9))
    proc9.construct_qiskit_diagram()

    # ---- proc10 ----
    vdata10 = virtualSpace(size=3, label="vdata10")
    vdata10.allocate_range(0,2)
    vsyn10 = virtualSpace(size=2, label="vsyn10", is_syndrome=True)
    vsyn10.allocate_range(0,1)
    proc10 = process(processID=10, start_time=0, vdataspace=vdata10, vsyndromespace=vsyn10)
    proc10.add_syscall(syscallinst=syscall_allocate_data_qubits(address=[vdata10.get_address(0),vdata10.get_address(1),vdata10.get_address(2)],size=3,processID=10))
    proc10.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(address=[vsyn10.get_address(0),vsyn10.get_address(1)],size=2,processID=10))
    proc10.add_instruction(Instype.H, [vsyn10.get_address(0)])
    proc10.add_instruction(Instype.CNOT, [vsyn10.get_address(0),vsyn10.get_address(1)])
    proc10.add_instruction(Instype.CNOT, [vdata10.get_address(0), vsyn10.get_address(0)])
    proc10.add_instruction(Instype.CNOT, [vdata10.get_address(1), vsyn10.get_address(1)])
    proc10.add_instruction(Instype.CNOT, [vsyn10.get_address(0), vsyn10.get_address(1)])
    proc10.add_instruction(Instype.MEASURE, [vsyn10.get_address(0)])
    proc10.add_instruction(Instype.MEASURE, [vsyn10.get_address(1)])
    proc10.add_syscall(syscallinst=syscall_deallocate_data_qubits(address=[vdata10.get_address(0),vdata10.get_address(1),vdata10.get_address(2)],size=3 ,processID=10))
    proc10.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(address=[vsyn10.get_address(0),vsyn10.get_address(1)],size=2,processID=10))
    proc10.construct_qiskit_diagram()


    proc2.construct_qiskit_diagram()


    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]


    #print(proc2)
    kernel_instance = Kernel(config={'max_virtual_logical_qubits': 1000, 'max_physical_qubits': 10000, 'max_syndrome_qubits': 1000})
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)
    kernel_instance.add_process(proc4)
    kernel_instance.add_process(proc5)
    kernel_instance.add_process(proc6)
    kernel_instance.add_process(proc7)
    kernel_instance.add_process(proc8)
    kernel_instance.add_process(proc9)
    kernel_instance.add_process(proc10)

    virtual_hardware = virtualHardware(qubit_number=156, error_rate=0.001,edge_list=COUPLING)

    return kernel_instance, virtual_hardware



def generate_example_two_procs_50d_30s():
    # ---------- proc1 ----------
    vdata1 = virtualSpace(size=50, label="vdata1")
    vdata1.allocate_range(0, 49)
    vsyn1  = virtualSpace(size=30, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 29)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)

    # allocate
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(50)], size=50, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(30)], size=30, processID=1))

    # scalable gate pattern: for each syndrome si, hit with H, then CNOT from two data qubits, then MEASURE
    for i in range(30):
        s  = vsyn1.get_address(i)
        d0 = vdata1.get_address((2*i)   % 50)
        d1 = vdata1.get_address((2*i+1) % 50)

        proc1.add_instruction(Instype.H,     [s])
        proc1.add_instruction(Instype.CNOT,  [d0, s])
        proc1.add_instruction(Instype.CNOT,  [d1, s])
        proc1.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(50)], size=50, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(30)], size=30, processID=1))

    # Optional: draw a circuit image if you want (comment out if slow)
    # proc1.construct_qiskit_diagram()

    # ---------- proc2 ----------
    vdata2 = virtualSpace(size=50, label="vdata2")
    vdata2.allocate_range(0, 49)
    vsyn2  = virtualSpace(size=30, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 29)

    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)

    # allocate
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(50)], size=50, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(30)], size=30, processID=2))

    # same scalable pattern with different data pairing (phase-shifted) to diversify
    for i in range(30):
        s  = vsyn2.get_address(i)
        d0 = vdata2.get_address((3*i)   % 50)
        d1 = vdata2.get_address((3*i+7) % 50)

        proc2.add_instruction(Instype.H,     [s])
        proc2.add_instruction(Instype.CNOT,  [d0, s])
        proc2.add_instruction(Instype.CNOT,  [d1, s])
        proc2.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(50)], size=50, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(30)], size=30, processID=2))

    # Optional: draw a circuit image if you want (comment out if slow)
    # proc2.construct_qiskit_diagram()

    # ---------- simple large hardware (linear chain) ----------
    # 300-vertex line as a safe default; adjust to your real layout
    N = 156
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]


    kernel_instance = Kernel(config={
        'max_virtual_logical_qubits': 10_000,
        'max_physical_qubits': 100_000,
        'max_syndrome_qubits': 10_000
    })
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=N, error_rate=0.001, edge_list=COUPLING)

    return kernel_instance, virtual_hardware




def generate_example_two_procs_30d_30s():
    # ---------- proc1 ----------
    vdata1 = virtualSpace(size=30, label="vdata1")
    vdata1.allocate_range(0, 29)
    vsyn1  = virtualSpace(size=30, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 29)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)

    # allocate
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(30)], size=30, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(30)], size=30, processID=1))

    # per-syndrome pattern
    for i in range(30):
        s  = vsyn1.get_address(i)
        d0 = vdata1.get_address((2*i)   % 30)
        d1 = vdata1.get_address((2*i+1) % 30)
        proc1.add_instruction(Instype.H,       [s])
        proc1.add_instruction(Instype.CNOT,    [d0, s])
        proc1.add_instruction(Instype.CNOT,    [d1, s])
        proc1.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(30)], size=30, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(30)], size=30, processID=1))

    # Optional:
    # proc1.construct_qiskit_diagram()

    # ---------- proc2 ----------
    vdata2 = virtualSpace(size=30, label="vdata2")
    vdata2.allocate_range(0, 29)
    vsyn2  = virtualSpace(size=30, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 29)

    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)

    # allocate
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(30)], size=30, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(30)], size=30, processID=2))

    # per-syndrome pattern (different pairing to diversify)
    for i in range(30):
        s  = vsyn2.get_address(i)
        d0 = vdata2.get_address((3*i)   % 30)
        d1 = vdata2.get_address((3*i+7) % 30)
        proc2.add_instruction(Instype.H,       [s])
        proc2.add_instruction(Instype.CNOT,    [d0, s])
        proc2.add_instruction(Instype.CNOT,    [d1, s])
        proc2.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(30)], size=30, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(30)], size=30, processID=2))

    # Optional:
    # proc2.construct_qiskit_diagram()

    # ---------- small hardware ----------
    # 120-node line as a simple default; replace with your real layout if needed.
    N = 156
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]


    kernel_instance = Kernel(config={
        'max_virtual_logical_qubits': 10_000,
        'max_physical_qubits': 100_000,
        'max_syndrome_qubits': 10_000
    })
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=N, error_rate=0.001, edge_list=COUPLING)

    return kernel_instance, virtual_hardware






def generate_example_two_procs_30d_30s_with_syndrome_cnots():
    # ---------- proc1 ----------
    vdata1 = virtualSpace(size=30, label="vdata1")
    vdata1.allocate_range(0, 29)
    vsyn1  = virtualSpace(size=30, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 29)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)

    # allocate
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(30)], size=30, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(30)], size=30, processID=1))

    # per-syndrome pattern
    for i in range(30):
        s  = vsyn1.get_address(i)
        d0 = vdata1.get_address((2*i)   % 30)
        d1 = vdata1.get_address((2*i+1) % 30)

        proc1.add_instruction(Instype.H,       [s])
        proc1.add_instruction(Instype.CNOT,    [d0, s])
        proc1.add_instruction(Instype.CNOT,    [d1, s])

        # NEW: add syndrome-syndrome CNOTs (link to neighbors)
        if i < 29:   # connect to next syndrome
            proc1.add_instruction(Instype.CNOT, [s, vsyn1.get_address(i+1)])
        if i >= 1:   # connect to previous syndrome
            proc1.add_instruction(Instype.CNOT, [s, vsyn1.get_address(i-1)])

        proc1.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(30)], size=30, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(30)], size=30, processID=1))

    # ---------- proc2 ----------
    vdata2 = virtualSpace(size=30, label="vdata2")
    vdata2.allocate_range(0, 29)
    vsyn2  = virtualSpace(size=30, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 29)

    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)

    # allocate
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(30)], size=30, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(30)], size=30, processID=2))

    for i in range(30):
        s  = vsyn2.get_address(i)
        d0 = vdata2.get_address((3*i)   % 30)
        d1 = vdata2.get_address((3*i+7) % 30)

        proc2.add_instruction(Instype.H,       [s])
        proc2.add_instruction(Instype.CNOT,    [d0, s])
        proc2.add_instruction(Instype.CNOT,    [d1, s])

        # NEW: add syndrome-syndrome CNOTs
        if i % 2 == 0 and i < 29:  # connect even syndromes to next
            proc2.add_instruction(Instype.CNOT, [s, vsyn2.get_address(i+1)])
        if i % 3 == 0 and i+2 < 30:  # connect every 3rd to one 2 steps away
            proc2.add_instruction(Instype.CNOT, [s, vsyn2.get_address(i+2)])

        proc2.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(30)], size=30, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(30)], size=30, processID=2))

    # ---------- simple hardware ----------
    N = 156
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]


    kernel_instance = Kernel(config={
        'max_virtual_logical_qubits': 10_000,
        'max_physical_qubits': 100_000,
        'max_syndrome_qubits': 10_000
    })
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=N, error_rate=0.001, edge_list=COUPLING)

    return kernel_instance, virtual_hardware






def generate_example_two_procs_40d_40s_with_syndrome_cnots():
    # ---------- proc1 ----------
    vdata1 = virtualSpace(size=40, label="vdata1")
    vdata1.allocate_range(0, 39)
    vsyn1  = virtualSpace(size=40, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 39)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)

    # allocate
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(40)], size=40, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(40)], size=40, processID=1))

    # per-syndrome pattern + neighbor syndrome CNOTs
    for i in range(40):
        s  = vsyn1.get_address(i)
        d0 = vdata1.get_address((2*i)   % 40)
        d1 = vdata1.get_address((2*i+1) % 40)

        proc1.add_instruction(Instype.H,       [s])
        proc1.add_instruction(Instype.CNOT,    [d0, s])
        proc1.add_instruction(Instype.CNOT,    [d1, s])

        # chain: link to next and previous syndrome before measuring
        if i < 39:
            proc1.add_instruction(Instype.CNOT, [s, vsyn1.get_address(i+1)])
        if i >= 1:
            proc1.add_instruction(Instype.CNOT, [s, vsyn1.get_address(i-1)])

        proc1.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(40)], size=40, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(40)], size=40, processID=1))

    # ---------- proc2 ----------
    vdata2 = virtualSpace(size=40, label="vdata2")
    vdata2.allocate_range(0, 39)
    vsyn2  = virtualSpace(size=40, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 39)

    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)

    # allocate
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(40)], size=40, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(40)], size=40, processID=2))

    for i in range(40):
        s  = vsyn2.get_address(i)
        d0 = vdata2.get_address((3*i)   % 40)
        d1 = vdata2.get_address((3*i+7) % 40)

        proc2.add_instruction(Instype.H,       [s])
        proc2.add_instruction(Instype.CNOT,    [d0, s])
        proc2.add_instruction(Instype.CNOT,    [d1, s])

        # varied topology: even i -> link to i+1; every 3rd -> link to i+2
        if (i % 2 == 0) and (i+1 < 40):
            proc2.add_instruction(Instype.CNOT, [s, vsyn2.get_address(i+1)])
        if (i % 3 == 0) and (i+2 < 40):
            proc2.add_instruction(Instype.CNOT, [s, vsyn2.get_address(i+2)])

        proc2.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(40)], size=40, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(40)], size=40, processID=2))

    # ---------- simple hardware ----------
    N = 156
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]

    kernel_instance = Kernel(config={
        'max_virtual_logical_qubits': 10_000,
        'max_physical_qubits': 100_000,
        'max_syndrome_qubits': 10_000
    })
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=N, error_rate=0.001, edge_list=COUPLING)

    return kernel_instance, virtual_hardware




def generate_example_two_procs_60d_60s_with_syndrome_cnots():
    # ---------- proc1 ----------
    vdata1 = virtualSpace(size=60, label="vdata1")
    vdata1.allocate_range(0, 59)
    vsyn1  = virtualSpace(size=60, label="vsyn1", is_syndrome=True)
    vsyn1.allocate_range(0, 59)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vsyn1)

    # allocate
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(60)], size=60, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(60)], size=60, processID=1))

    # per-syndrome pattern + neighbor syndrome CNOTs
    for i in range(60):
        s  = vsyn1.get_address(i)
        d0 = vdata1.get_address((2*i)   % 60)
        d1 = vdata1.get_address((2*i+1) % 60)

        proc1.add_instruction(Instype.H,       [s])
        proc1.add_instruction(Instype.CNOT,    [d0, s])
        proc1.add_instruction(Instype.CNOT,    [d1, s])

        # chain: link to next and previous syndrome before measuring
        if i < 59:
            proc1.add_instruction(Instype.CNOT, [s, vsyn1.get_address(i+1)])
        if i >= 1:
            proc1.add_instruction(Instype.CNOT, [s, vsyn1.get_address(i-1)])

        proc1.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(60)], size=60, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn1.get_address(i) for i in range(60)], size=60, processID=1))

    # ---------- proc2 ----------
    vdata2 = virtualSpace(size=60, label="vdata2")
    vdata2.allocate_range(0, 59)
    vsyn2  = virtualSpace(size=60, label="vsyn2", is_syndrome=True)
    vsyn2.allocate_range(0, 59)

    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vsyn2)

    # allocate
    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(60)], size=60, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(60)], size=60, processID=2))

    for i in range(60):
        s  = vsyn2.get_address(i)
        d0 = vdata2.get_address((3*i)   % 60)
        d1 = vdata2.get_address((3*i+7) % 60)

        proc2.add_instruction(Instype.H,       [s])
        proc2.add_instruction(Instype.CNOT,    [d0, s])
        proc2.add_instruction(Instype.CNOT,    [d1, s])

        # varied topology: even i -> link to i+1; every 3rd -> link to i+2
        if (i % 2 == 0) and (i+1 < 60):
            proc2.add_instruction(Instype.CNOT, [s, vsyn2.get_address(i+1)])
        if (i % 3 == 0) and (i+2 < 60):
            proc2.add_instruction(Instype.CNOT, [s, vsyn2.get_address(i+2)])

        proc2.add_instruction(Instype.MEASURE, [s])

    # deallocate
    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(60)], size=60, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vsyn2.get_address(i) for i in range(60)], size=60, processID=2))

    # ---------- simple hardware ----------
    # ---------- simple hardware ----------
    N = 156
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]


    kernel_instance = Kernel(config={
        'max_virtual_logical_qubits': 10_000,
        'max_physical_qubits': 100_000,
        'max_syndrome_qubits': 10_000
    })
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)

    virtual_hardware = virtualHardware(qubit_number=N, error_rate=0.001, edge_list=COUPLING)

    return kernel_instance, virtual_hardware





def generate_example_three_procs_40d_40a_with_ancilla_cnots():
    # ---------- proc1 ----------
    vdata1 = virtualSpace(size=40, label="vdata1")
    vdata1.allocate_range(0, 39)
    vanc1  = virtualSpace(size=40, label="vanc1", is_syndrome=True)  # ancilla
    vanc1.allocate_range(0, 39)

    proc1 = process(processID=1, start_time=0, vdataspace=vdata1, vsyndromespace=vanc1)

    # allocate
    proc1.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(40)], size=40, processID=1))
    proc1.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vanc1.get_address(i) for i in range(40)], size=40, processID=1))

    # per-ancilla pattern + neighbor ancilla CNOTs (chain)
    for i in range(40):
        a  = vanc1.get_address(i)
        d0 = vdata1.get_address((2*i)   % 40)
        d1 = vdata1.get_address((2*i+1) % 40)

        proc1.add_instruction(Instype.H,       [a])
        proc1.add_instruction(Instype.CNOT,    [d0, a])
        proc1.add_instruction(Instype.CNOT,    [d1, a])
        if i < 39:
            proc1.add_instruction(Instype.CNOT, [a, vanc1.get_address(i+1)])
        if i >= 1:
            proc1.add_instruction(Instype.CNOT, [a, vanc1.get_address(i-1)])
        proc1.add_instruction(Instype.MEASURE, [a])

    # deallocate
    proc1.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata1.get_address(i) for i in range(40)], size=40, processID=1))
    proc1.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vanc1.get_address(i) for i in range(40)], size=40, processID=1))

    # ---------- proc2 ----------
    vdata2 = virtualSpace(size=40, label="vdata2")
    vdata2.allocate_range(0, 39)
    vanc2  = virtualSpace(size=40, label="vanc2", is_syndrome=True)
    vanc2.allocate_range(0, 39)

    proc2 = process(processID=2, start_time=0, vdataspace=vdata2, vsyndromespace=vanc2)

    proc2.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(40)], size=40, processID=2))
    proc2.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vanc2.get_address(i) for i in range(40)], size=40, processID=2))

    # varied ancilla topology
    for i in range(40):
        a  = vanc2.get_address(i)
        d0 = vdata2.get_address((3*i)   % 40)
        d1 = vdata2.get_address((3*i+7) % 40)

        proc2.add_instruction(Instype.H,       [a])
        proc2.add_instruction(Instype.CNOT,    [d0, a])
        proc2.add_instruction(Instype.CNOT,    [d1, a])
        if (i % 2 == 0) and (i+1 < 40):
            proc2.add_instruction(Instype.CNOT, [a, vanc2.get_address(i+1)])
        if (i % 3 == 0) and (i+2 < 40):
            proc2.add_instruction(Instype.CNOT, [a, vanc2.get_address(i+2)])
        proc2.add_instruction(Instype.MEASURE, [a])

    proc2.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata2.get_address(i) for i in range(40)], size=40, processID=2))
    proc2.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vanc2.get_address(i) for i in range(40)], size=40, processID=2))

    # ---------- proc3 ----------
    vdata3 = virtualSpace(size=40, label="vdata3")
    vdata3.allocate_range(0, 39)
    vanc3  = virtualSpace(size=40, label="vanc3", is_syndrome=True)
    vanc3.allocate_range(0, 39)

    proc3 = process(processID=3, start_time=0, vdataspace=vdata3, vsyndromespace=vanc3)

    proc3.add_syscall(syscallinst=syscall_allocate_data_qubits(
        address=[vdata3.get_address(i) for i in range(40)], size=40, processID=3))
    proc3.add_syscall(syscallinst=syscall_allocate_syndrome_qubits(
        address=[vanc3.get_address(i) for i in range(40)], size=40, processID=3))

    # ring + skip-links (wraparound) on ancilla
    for i in range(40):
        a  = vanc3.get_address(i)
        d0 = vdata3.get_address((5*i)    % 40)
        d1 = vdata3.get_address((5*i+11) % 40)

        proc3.add_instruction(Instype.H,       [a])
        proc3.add_instruction(Instype.CNOT,    [d0, a])
        proc3.add_instruction(Instype.CNOT,    [d1, a])

        nxt = vanc3.get_address((i+1) % 40)   # ring neighbor
        skip = vanc3.get_address((i+5) % 40)  # skip-link
        proc3.add_instruction(Instype.CNOT, [a, nxt])
        proc3.add_instruction(Instype.CNOT, [a, skip])

        proc3.add_instruction(Instype.MEASURE, [a])

    proc3.add_syscall(syscallinst=syscall_deallocate_data_qubits(
        address=[vdata3.get_address(i) for i in range(40)], size=40, processID=3))
    proc3.add_syscall(syscallinst=syscall_deallocate_syndrome_qubits(
        address=[vanc3.get_address(i) for i in range(40)], size=40, processID=3))

    # ---------- simple hardware (adjust as needed) ----------
    N = 156
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], # The first long row
        [3,16], [16,23], [7,17], [17,27], [11,18], [18,31], [15,19], [19,35], # Short row 1
        [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], [33,34], [34,35], # The second long row
        [21,36], [36,41], [25,37], [37,45], [29,38], [38,49], [33,39], [39,53], # Short row 2
        [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], # The third long row
        [43,56], [56,63], [47,57], [57,67], [51,58], [58,71], [55,59], [59,75], # Short row 3
        [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], [71,72], [72,73], [73,74], [74,75], # The forth long row
        [61,76], [76,81], [65,77], [77,85], [69,78], [78,89], [73,79], [79,93],# Short row 4
        [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,91], [91,92], [92,93], [93,94], [94,95], # The fifth long row
        [83,96], [96,103], [87,97], [97,107], [91,98], [98,111], [95,99], [99,115], # Short row 5
        [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], [109,110], [110,111], [111,112], [112,113], [113,114], [114,115], # The sixth long row
        [101,116], [116,121], [105,117], [117,125], [109,118], [118,129], [113,119], [119,133],  # Short row 6
        [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], [128,129], [129,130], [130,131], [131,132], [132,133], [133,134], [134,135], # The seventh long row
        [123,136], [136,143], [127,137], [137,147], [131,138], [138,151], [135,139], [139,155], # Short row 7
        [140,141], [141,142], [142,143], [143,144], [144,145], [145,146], [146,147], [147,148], [148,149], [149,150], [150,151], [151,152], [152,153], [153,154], [154,155] # The eighth long row
    ]


    kernel_instance = Kernel(config={
        'max_virtual_logical_qubits': 10_000,
        'max_physical_qubits': 100_000,
        'max_syndrome_qubits': 10_000
    })
    kernel_instance.add_process(proc1)
    kernel_instance.add_process(proc2)
    kernel_instance.add_process(proc3)

    virtual_hardware = virtualHardware(qubit_number=N, error_rate=0.001, edge_list=COUPLING)

    return kernel_instance, virtual_hardware



def distribution_fidelity(dist1: dict, dist2: dict) -> float:
    """
    Compute fidelity between two distributions based on L1 distance.
    1. Normalize both distributions to get probability distributions.
    2. Compute the L1 distance between the two probability distributions.
    3. Fidelity = 1 - L1/2, which lies in [0,1].
    
    Args:
        dist1 (dict): key=str (event), value=int (count)
        dist2 (dict): key=str (event), value=int (count)
    
    Returns:
        float: fidelity in [0, 1]
    """
    # Step 1: normalize distributions
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    prob1 = {k: v / total1 for k, v in dist1.items()}
    prob2 = {k: v / total2 for k, v in dist2.items()}
    
    # Step 2: union of keys
    all_keys = set(prob1.keys()).union(set(prob2.keys()))
    
    # Step 3: compute L1 distance
    l1_distance = sum(abs(prob1.get(k, 0) - prob2.get(k, 0)) for k in all_keys)
    
    # Step 4: fidelity = 1 - L1/2
    fidelity = 1 - l1_distance / 2
    return fidelity



def build_noise_model(error_rate_1q=0.001, error_rate_2q=0.01, p_reset=0.001, p_meas=0.01):
    custom_noise_model = NoiseModel()
    
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])

    custom_noise_model.add_all_qubit_quantum_error(error_reset,"reset")

    custom_noise_model.add_all_qubit_quantum_error(error_meas,"measure")
    custom_noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate_1q, 1), ['id','rx','rz','sx','x'])

    # Add a depolarizing error to two-qubit gates on specific qubits
    custom_noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate_2q, 2), ['cz','rzz'])


    return custom_noise_model


















# ========================  MAIN  ========================
if __name__ == "__main__":




    kernel_instance, virtual_hardware =generate_simples_example_for_test_2()
    #kernel_instance, virtual_hardware = generate_example_ppt10_on_10_qubit_device()

    schedule_instance = Scheduler(kernel_instance=kernel_instance, hardware_instance=virtual_hardware)



    dis=schedule_instance.calculate_all_pair_distance()


    time1, inst_list1=schedule_instance.dynamic_scheduling()
    #time1, inst_list1=schedule_instance.baseline_scheduling()
    #time1, inst_list1=schedule_instance.dynamic_scheduling_no_consider_connectivity()
    #time1, inst_list1=schedule_instance.scheduling_with_out_sharing_syndrome_qubit()
    schedule_instance.print_dynamic_instruction_list(inst_list1)
    qc=schedule_instance.construct_qiskit_circuit_for_backend(inst_list1)



    # fig_t = qc.draw(output="mpl", fold=-1)
    # fig_t.savefig("before_transpiled.png", dpi=200, bbox_inches="tight")
    # plt.close(fig_t)

    # qc.draw("mpl", fold=-1).show()
    # print(qc.num_qubits)

    # 0) Fake 156-qubit backend (your Pittsburgh layout)
    fake_hard_ware = construct_20_qubit_hardware()


    # 1) Build the abstract (logical) circuit and save as PNG
    # qc = build_dynamic_circuit_15()
    # save_circuit_png(qc, "abstract_circuit.png")  # uses Matplotlib

    # 2) Transpile to hardware; map 15 logical qubits onto a single long row
    #    (contiguous physical qubits minimize SWAPs on your lattice)
    initial_layout = [i for i in range(20)]  # logical i -> physical i



    transpiled = transpile(
        qc,
        backend= fake_hard_ware,
        initial_layout=initial_layout,
        optimization_level=3,
    )
    print("\n=== Transpiled circuit ===")
    print(transpiled)

    # Save the transpiled circuit PNG too
    # import matplotlib.pyplot as plt
    # fig_t = transpiled.draw(output="mpl", fold=-1)
    # fig_t.savefig("transpiled_circuit.png", dpi=200, bbox_inches="tight")
    # plt.close(fig_t)



    process_list = schedule_instance.get_all_processes()
    syndrome_history = schedule_instance.get_syndrome_map_history()
    plot_process_schedule_on_20_qubit_hardware(
        coupling_edges= fake_hard_ware.coupling_map,
        syndrome_qubit_history=syndrome_history,
        process_list=process_list,
        out_png="hardware_processes.png",
    )
    



    # 4) Run on the fake backend (Aer noise if installed; otherwise ideal) and print counts

    job = fake_hard_ware.run(transpiled, shots=5000)
    result = job.result()
    running_time=result.time_taken

   

    sim = AerSimulator(noise_model=build_noise_model(error_rate_1q=0.01, error_rate_2q=0.05, p_reset=0.01, p_meas=0.01))
    tqc = transpile(transpiled, sim)
    result = sim.run(tqc, shots=2000).result()
    counts = result.get_counts(tqc)
    # print("\n=== Counts(Fake hardware) ===")
    print(counts)   



    '''
    Get the ideal result
    '''
    sim = AerSimulator()
    tqc = transpile(qc, sim)

    # Run with 1000 shots
    result = sim.run(tqc, shots=2000).result()
    idcounts = result.get_counts(tqc)
    print("\n=== Counts(Ideal) ===")
    print(idcounts)





    # print(schedule_instance._measure_index_to_process)
    # print(schedule_instance._process_measure_index)


    final_result=schedule_instance.return_measure_states(counts)
    #print(final_result)


    ideal_result=schedule_instance.return_process_ideal_output(shots=2000)
    #print(ideal_result)



    average_fidelity=0
    for pid in final_result.keys():
        print("Ideal result for process ", pid)
        print(ideal_result[pid])
        print("Final result for process ", pid)
        print(final_result[pid])
        fidelity=distribution_fidelity(final_result[pid], ideal_result[pid])
        average_fidelity+=fidelity
        print(f"Fidelity for process {pid}: {fidelity:.4f}")




    print("The TRANSPILED circuit depth is:", transpiled .depth())

    print("\n=== Time taken:===")
    print(running_time)

    average_fidelity/=len(final_result.keys())
    print(f"Average fidelity: {average_fidelity:.4f}")