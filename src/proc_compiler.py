#Compile a process instance written in standard format.


import re
from scheduler import *
from process import process
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_coupling_map
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional


# assuming your ProcessStatus and process classes are already defined in scope
# from <your_module> import process, ProcessStatus

_param_list_re = re.compile(r"\(([^)]*)\)")

def _parse_float_list(s: str) -> List[float]:
    """
    Parse a comma-separated list of floats inside parentheses.
    Accepts whitespace; returns [] if s is empty/only spaces.
    """
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [float(p) for p in parts if p]

def _grab_params(text_after_gate: str) -> Tuple[List[float], str]:
    """
    Given '(<params>) rest', return (params_list, rest_without_paren_prefix).
    If no parentheses, returns ([], original_text).
    """
    m = _param_list_re.match(text_after_gate.strip())
    if not m:
        return [], text_after_gate.strip()
    params = _parse_float_list(m.group(1))
    # remove the '(...)' prefix from the remaining text
    remainder = text_after_gate.strip()[m.end():].strip()
    return params, remainder

def _addr_from_token(tok: str, vdata: virtualSpace, vsyn: virtualSpace) -> virtualAddress:
    """
    Convert 'q7' or 's0' into a virtualAddress from the corresponding virtualSpace.
    """
    tok = tok.strip()
    if not tok:
        raise ValueError("Empty address token.")
    reg = tok[0]
    if reg not in ("q", "s"):
        raise ValueError(f"Unknown register prefix '{reg}' in token '{tok}'. Expected 'q' or 's'.")
    try:
        idx = int(tok[1:])
    except ValueError:
        raise ValueError(f"Bad register index in '{tok}'. Expected e.g. q0, s3.")
    return (vdata if reg == "q" else vsyn).get_address(idx)

def parse_process_program(program_str: str, process_ID: int) -> "process":
    """
    Parse the custom process program DSL into a `process` object.

    Expects header lines:
        q = alloc_data(N)
        s = alloc_helper(M)
        set_shot(S)

    Supports gates like:
        H q0
        CNOT q0, q2
        RX(1.234) q3
        U(theta, phi, lam) q1
        ...
        c0 = MEASURE q2
        deallocate_data(q)
        deallocate_helper(s)
    """
    # normalize and split lines
    raw_lines = [ln.strip() for ln in program_str.strip().splitlines() if ln.strip()]

    # ---- Pass 1: read header (alloc & shots) ----
    data_n = 0
    syn_n = 0
    shots = 1000  # default if not provided

    alloc_data_re = re.compile(r"^q\s*=\s*alloc_data\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    alloc_syn_re  = re.compile(r"^s\s*=\s*alloc_helper\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    set_shot_re   = re.compile(r"^set_shot\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)

    # We’ll keep the non-header lines to parse as instructions
    instr_lines: List[str] = []

    for ln in raw_lines:
        if m := alloc_data_re.match(ln):
            data_n = int(m.group(1))
            continue
        if m := alloc_syn_re.match(ln):
            syn_n = int(m.group(1))
            continue
        if m := set_shot_re.match(ln):
            shots = int(m.group(1))
            continue
        # not a header line → it’s an instruction or dealloc
        instr_lines.append(ln)

    # ---- Build spaces & process and issue allocate syscalls ----
    # Labels are just names for registers; choose something sensible.
    vdata = virtualSpace(size=max(data_n, 0), label="q")                # data qubits labeled 'q'
    vsyn  = virtualSpace(size=max(syn_n, 0), label="s", is_syndrome=True)  # helper/syndrome labeled 's'

    # In your framework, allocations are tracked via syscalls & process counters
    proc = process(processID=process_ID, start_time=0, vdataspace=vdata, vsyndromespace=vsyn, shots=shots)
    if data_n > 0:
        proc.add_syscall(
            syscallinst=syscall_allocate_data_qubits(
                address=[vdata.get_address(0)], size=data_n, processID=process_ID
            )
        )
    if syn_n > 0:
        proc.add_syscall(
            syscallinst=syscall_allocate_syndrome_qubits(
                address=None, size=syn_n, processID=process_ID
            )
        )

    # ---- Parsers for instruction lines ----
    # Simple 1-qubit no-parameter gates
    oneq_no_param = {
        "h": Instype.H,
        "x": Instype.X,
        "y": Instype.Y,
        "z": Instype.Z,
        "t": Instype.T,
        "tdg": Instype.Tdg,
        "s": Instype.S,
        "sdg": Instype.Sdg,
        "sx": Instype.SX,
        "reset": Instype.RESET,
    }

    # Two-qubit gates without params
    twoq_no_param = {
        "cnot": Instype.CNOT,
        "cx": Instype.CNOT,   # alias
        "ch": Instype.CH,
        "swap": Instype.SWAP,
    }

    # Three-qubit no-param
    threeq_no_param = {
        "cswap": Instype.CSWAP,
        "toffoli": Instype.Toffoli,
        "ccx": Instype.Toffoli,  # alias
    }

    # Helpers
    def add_oneq_gate(kind: Instype, addr_tok: str, params: Optional[List[float]] = None):
        addr = _addr_from_token(addr_tok, vdata, vsyn)
        if params:
            proc.add_instruction(kind, [addr], params=params)
        else:
            proc.add_instruction(kind, [addr])

    def add_twoq_gate(kind: Instype, tok_a: str, tok_b: str, params: Optional[List[float]] = None):
        a = _addr_from_token(tok_a, vdata, vsyn)
        b = _addr_from_token(tok_b, vdata, vsyn)
        if params:
            proc.add_instruction(kind, [a, b], params=params)
        else:
            proc.add_instruction(kind, [a, b])

    def add_threeq_gate(kind: Instype, tok_a: str, tok_b: str, tok_c: str):
        a = _addr_from_token(tok_a, vdata, vsyn)
        b = _addr_from_token(tok_b, vdata, vsyn)
        c = _addr_from_token(tok_c, vdata, vsyn)
        proc.add_instruction(kind, [a, b, c])

    # Regexes for instruction shapes
    measure_re = re.compile(r"^c\s*(\d+)\s*=\s*MEASURE\s+([qs]\d+)\s*$", re.IGNORECASE)
    # dealloc lines (we'll synthesize syscalls ourselves, but allow them to appear)
    dealloc_q_re = re.compile(r"^deallocate_data\s*\(\s*q\s*\)\s*$", re.IGNORECASE)
    dealloc_s_re = re.compile(r"^deallocate_helper\s*\(\s*s\s*\)\s*$", re.IGNORECASE)

    for ln in instr_lines:
        # Skip optional trailing dealloc directives; we'll add syscalls at the end.
        if dealloc_q_re.match(ln) or dealloc_s_re.match(ln):
            continue

        # Measurement: "c0 = MEASURE q2"
        m = measure_re.match(ln)
        if m:
            cidx = int(m.group(1))
            qtok = m.group(2)
            qaddr = _addr_from_token(qtok, vdata, vsyn)
            proc.add_instruction(Instype.MEASURE, [qaddr], classical_address=cidx)
            continue

        # Tokenize: first word is gate mnemonic (maybe with params), the rest are args
        # We'll manually pull params when present (RX/RY/RZ/U/CU1).
        # Examples:
        #   "H q0"
        #   "RX(1.23) q1"
        #   "U(θ,φ,λ) q3"
        #   "CNOT q0, q2"
        #   "CU1(π/2) q0, q1"
        parts = ln.split(None, 1)
        if not parts:
            continue
        gate_full = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""

        gate = gate_full.lower()

        # Parameterized single-qubit
        if gate.startswith("rx"):
            params, rest2 = _grab_params(rest if gate == "rx" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RX, target, params=params)
            continue

        if gate.startswith("ry"):
            params, rest2 = _grab_params(rest if gate == "ry" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RY, target, params=params)
            continue

        if gate.startswith("rz"):
            params, rest2 = _grab_params(rest if gate == "rz" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RZ, target, params=params)
            continue

        if gate.startswith("u3"):  # treat like U
            params, rest2 = _grab_params(rest if gate == "u3" else gate_full[2:] + rest)
            target = rest2
            if len(params) != 3:
                raise ValueError(f"U3 expects 3 parameters, got {params}")
            add_oneq_gate(Instype.U3, target, params=params)
            continue

        if gate.startswith("u(") or gate == "u":
            # handle forms like "U( ... ) q0" or weird tokenization
            params, rest2 = _grab_params(ln[1:] if gate_full.lower().startswith("u(") else rest)
            target = rest2
            if len(params) != 3:
                raise ValueError(f"U expects 3 parameters, got {params}")
            add_oneq_gate(Instype.U, target, params=params)
            continue

        # Parameterized two-qubit
        if gate.startswith("cu1") or gate.startswith("cp"):  # accept CP as CU1 alias if desired
            params, rest2 = _grab_params(rest if gate in ("cu1", "cp") else gate_full[3:] + rest)
            if len(params) != 1:
                raise ValueError(f"CU1/CP expects 1 parameter, got {params}")
            # rest2 shape: "q0, q1"
            toks = [t.strip() for t in rest2.split(",")]
            if len(toks) != 2:
                raise ValueError(f"CU1 expects two qubit args, got '{rest2}'")
            add_twoq_gate(Instype.CU1, toks[0], toks[1], params=params)
            continue

        # No-parameter one-qubit?
        if gate in oneq_no_param:
            # rest should be like "q0"
            add_oneq_gate(oneq_no_param[gate], rest)
            continue

        # Two-qubit no-param?
        if gate in twoq_no_param:
            toks = [t.strip() for t in rest.split(",")]
            if len(toks) != 2:
                raise ValueError(f"{gate.upper()} expects two qubit args, got '{rest}'")
            add_twoq_gate(twoq_no_param[gate], toks[0], toks[1])
            continue

        # Three-qubit no-param?
        if gate in threeq_no_param:
            toks = [t.strip() for t in rest.split(",")]
            if len(toks) != 3:
                raise ValueError(f"{gate.upper()} expects three qubit args, got '{rest}'")
            add_threeq_gate(threeq_no_param[gate], toks[0], toks[1], toks[2])
            continue

        raise ValueError(f"Unsupported or malformed instruction line: '{ln}'")

    # ---- Add dealloc syscalls (mirroring your QASM parser) ----
    if data_n > 0:
        proc.add_syscall(
            syscallinst=syscall_deallocate_data_qubits(
                address=[vdata.get_address(0)], size=data_n, processID=process_ID
            )
        )
    if syn_n > 0:
        proc.add_syscall(
            syscallinst=syscall_deallocate_syndrome_qubits(
                address=None, size=syn_n, processID=process_ID
            )
        )

    return proc





#Test code

if __name__ == "__main__":
    path="C:\\Users\\yezhu\\OneDrive\\Documents\\GitHub\\FTQos\\benchmarks\\example1"
    with open(path, 'r') as file:
        test_script = file.read()

    compiled_process = compile_quantum_script(test_script)
    print(compiled_process)