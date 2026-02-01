import os
import subprocess
import inspect
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional
from qiskit import transpile
import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator, Statevector, random_statevector
from qiskit.circuit.library import CXGate, Isometry
from qiskit.synthesis import TwoQubitBasisDecomposer, OneQubitEulerDecomposer

GRIDSYNTH_PATH = "/Users/justinsato/Downloads/gridsynth"
GRIDSYNTH_EFFORT = 5000
ERR_TARGET = 9e-7
SNAP_TOL = 1e-11
THETA_KEY_DIGITS = 15
EPS_KEY_DIGITS = 15
EULER_BASES = ["ZXZ"]
PRESERVE_CX_STRUCTURE = False

def wrap_pi(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi

def nearest_k_pi_over_4(theta: float) -> Tuple[int, float]:
    theta = wrap_pi(theta)
    k = int(np.round(theta / (np.pi / 4))) % 8
    err = wrap_pi(theta - k * (np.pi / 4))
    return k, err

def phase_insensitive_err(U: np.ndarray, V: np.ndarray) -> float:
    u_vec = U.flatten()
    v_vec = V.flatten()
    dot = np.vdot(v_vec, u_vec)
    phase = dot / abs(dot) if abs(dot) > 1e-15 else 1.0
    return np.linalg.norm(u_vec - phase * v_vec)

def count_t_like(qc: QuantumCircuit) -> int:
    ops = qc.count_ops()
    return int(ops.get("t", 0) + ops.get("tdg", 0))

def emit_exact_rz_pi_over_4(qc: QuantumCircuit, q: int, k_mod8: int):
    k = k_mod8 % 8
    if k == 0:
        return
    if k == 1:
        qc.t(q)
    elif k == 2:
        qc.s(q)
    elif k == 3:
        qc.s(q); qc.t(q)
    elif k == 4:
        qc.z(q)
    elif k == 5:
        qc.z(q); qc.t(q)
    elif k == 6:
        qc.z(q); qc.s(q)
    elif k == 7:
        qc.z(q); qc.s(q); qc.t(q)

def needs_gridsynth(theta: float) -> bool:
    theta = wrap_pi(theta)
    k, err = nearest_k_pi_over_4(theta)
    if abs(err) <= SNAP_TOL:
        return False
    if abs(theta) <= 1e-16:
        return False
    return True

@dataclass
class SegOp:
    name: str
    theta: float = 0.0

def simplify_segops(ops: List[SegOp]) -> List[SegOp]:
    out: List[SegOp] = []

    def flush_phase_pow(k_mod8: int):
        if k_mod8 % 8 == 0:
            return
        qc_tmp = QuantumCircuit(1)
        emit_exact_rz_pi_over_4(qc_tmp, 0, k_mod8 % 8)
        for inst in qc_tmp.data:
            nm = inst.operation.name
            out.append(SegOp(nm))

    phase_k = 0
    in_phase_run = False

    def end_phase_run():
        nonlocal phase_k, in_phase_run
        if in_phase_run:
            flush_phase_pow(phase_k)
            phase_k = 0
            in_phase_run = False

    for op in ops:
        if op.name == "rz":
            th = wrap_pi(op.theta)
            if abs(th) <= 1e-16:
                continue
            if out and out[-1].name == "rz":
                out[-1].theta = wrap_pi(out[-1].theta + th)
                if abs(out[-1].theta) <= 1e-16:
                    out.pop()
            else:
                out.append(SegOp("rz", th))
            continue

        if op.name in ("t", "tdg", "s", "sdg", "z"):
            if not in_phase_run:
                in_phase_run = True
                phase_k = 0
            if op.name == "t":
                phase_k = (phase_k + 1) % 8
            elif op.name == "tdg":
                phase_k = (phase_k - 1) % 8
            elif op.name == "s":
                phase_k = (phase_k + 2) % 8
            elif op.name == "sdg":
                phase_k = (phase_k - 2) % 8
            elif op.name == "z":
                phase_k = (phase_k + 4) % 8
            continue

        end_phase_run()

        if op.name == "h":
            if out and out[-1].name == "h":
                out.pop()
            else:
                out.append(op)
        else:
            out.append(op)

    end_phase_run()
    return out

def oneq_circuit_to_segops(qc1: QuantumCircuit) -> List[SegOp]:
    ops: List[SegOp] = []
    for inst in qc1.data:
        nm = inst.operation.name
        if nm == "rz":
            ops.append(SegOp("rz", float(inst.operation.params[0])))
        elif nm == "rx":
            th = float(inst.operation.params[0])
            ops.extend([SegOp("h"), SegOp("rz", th), SegOp("h")])
        elif nm == "ry":
            th = float(inst.operation.params[0])
            ops.extend([SegOp("s"), SegOp("h"), SegOp("rz", th), SegOp("h"), SegOp("sdg")])
        else:
            raise ValueError(f"Unexpected 1q op in Euler decomposition: {nm}")
    return simplify_segops(ops)

def canonicalize_1q_unitary_best(U2: np.ndarray) -> List[SegOp]:
    best_ops: Optional[List[SegOp]] = None
    best_cost = None
    for basis in EULER_BASES:
        decomp = OneQubitEulerDecomposer(basis)
        qc1 = decomp(U2)
        ops = oneq_circuit_to_segops(qc1)
        cost = sum(1 for op in ops if op.name == "rz" and needs_gridsynth(op.theta))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_ops = ops
    assert best_ops is not None
    return best_ops

_GRIDSYNTH_USES_CIRCUIT_ORDER: Optional[bool] = None

def _build_qc_from_tokens(tokens: List[str], circuit_order: bool) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    toks = tokens if circuit_order else list(reversed(tokens))
    for t in toks:
        if t == "H":
            qc.h(0)
        elif t == "T":
            qc.t(0)
        elif t == "S":
            qc.s(0)
        elif t == "X":
            qc.x(0)
        else:
            raise ValueError(f"Unknown gridsynth token: {t}")
    return qc

def _detect_gridsynth_token_order(tokens: List[str], theta: float) -> bool:
    qc_tgt = QuantumCircuit(1)
    qc_tgt.rz(theta, 0)
    U_tgt = Operator(qc_tgt).data
    qc_as_is = _build_qc_from_tokens(tokens, circuit_order=True)
    qc_rev = _build_qc_from_tokens(tokens, circuit_order=False)
    e_as_is = phase_insensitive_err(U_tgt, Operator(qc_as_is).data)
    e_rev = phase_insensitive_err(U_tgt, Operator(qc_rev).data)
    return e_as_is <= e_rev

@lru_cache(maxsize=None)
def rz_via_gridsynth_cached(theta_key: float, eps_key: float) -> QuantumCircuit:
    theta = float(theta_key)
    epsilon = float(eps_key)
    theta_norm = wrap_pi(theta)
    cmd = [
        GRIDSYNTH_PATH,
        "-e", f"{epsilon:.15e}",
        "-f", str(GRIDSYNTH_EFFORT),
        "--phase",
        "--",
        f"{theta_norm:.15f}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    s = "".join(result.stdout.split()).replace("W", "")
    tokens = list(s)
    global _GRIDSYNTH_USES_CIRCUIT_ORDER
    if _GRIDSYNTH_USES_CIRCUIT_ORDER is None:
        _GRIDSYNTH_USES_CIRCUIT_ORDER = _detect_gridsynth_token_order(tokens, theta_norm)
    return _build_qc_from_tokens(tokens, circuit_order=_GRIDSYNTH_USES_CIRCUIT_ORDER)

def rz_via_gridsynth(theta: float, epsilon: float) -> QuantumCircuit:
    theta_norm = wrap_pi(theta)
    k, err = nearest_k_pi_over_4(theta_norm)
    if abs(err) <= SNAP_TOL:
        qc = QuantumCircuit(1)
        emit_exact_rz_pi_over_4(qc, 0, k)
        return qc
    theta_key = float(f"{theta_norm:.{THETA_KEY_DIGITS}f}")
    eps_key = float(f"{epsilon:.{EPS_KEY_DIGITS}e}")
    return rz_via_gridsynth_cached(theta_key, eps_key)

def main():
    psi = random_statevector(4, seed=42)
    base_qc = QuantumCircuit(2)
    base_qc.append(Isometry(psi, 0, 0), [0, 1])
    base_qc = transpile(
        base_qc,
        basis_gates=["cx", "rz", "rx", "ry", "h", "s", "sdg", "x", "z"],
        optimization_level=3
    )
    qc_final, eps_used, err = synthesize_min_t_from_circuit(base_qc, ERR_TARGET)
    raw_qasm = qasm2.dumps(qc_final)
    clean_qasm = enforce_clifford_compliance(raw_qasm)
    with open("7synthesized_state_prep.qasm", "w") as f:
        f.write(clean_qasm)

if __name__ == "__main__":
    main()
