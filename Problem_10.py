import subprocess
import numpy as np
import os
import inspect
import random
import re

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Callable, Any, Optional

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import CXGate
from qiskit.synthesis import TwoQubitBasisDecomposer, OneQubitEulerDecomposer


GRIDSYNTH_PATH = "/Users/justinsato/Downloads/gridsynth"
EFFORT = 100

ERR_TARGET = 7e-5
SNAP_TOL = 1e-11

THETA_KEY_DIGITS = 15
EPS_KEY_DIGITS = 15

FRAME_SAMPLES = 300
TOP_K_FRAMES = 4

EPS_GRID = [3e-9]
USE_RMS_ALLOC = True


def wrap_pi(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def nearest_k_pi_over_4(theta: float) -> Tuple[int, float]:
    theta = wrap_pi(theta)
    k = int(np.round(theta / (np.pi / 4))) % 8
    err = wrap_pi(theta - k * (np.pi / 4))
    return k, err


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


def phase_insensitive_err(U: np.ndarray, V: np.ndarray) -> float:
    u = U.flatten()
    v = V.flatten()
    dot = np.vdot(v, u)
    phase = dot / abs(dot) if abs(dot) > 1e-15 else 1.0
    return np.linalg.norm(u - phase * v)


def tcount(qc: QuantumCircuit) -> int:
    ops = qc.count_ops()
    return int(ops.get("t", 0) + ops.get("tdg", 0))


_SWAP_RE = re.compile(r"^\s*swap\s+([^;]+)\s*;\s*$", re.IGNORECASE)


def expand_swap_in_qasm2(qasm: str) -> str:
    out = []
    for line in qasm.splitlines():
        m = _SWAP_RE.match(line)
        if not m:
            out.append(line)
            continue
        a, b = [p.strip() for p in m.group(1).split(",")]
        out.append(f"cx {a},{b};")
        out.append(f"cx {b},{a};")
        out.append(f"cx {a},{b};")
    return "\n".join(out)


@lru_cache(maxsize=None)
def rz_via_gridsynth_cached(theta_key: float, eps_key: float) -> QuantumCircuit:
    theta = wrap_pi(float(theta_key))
    eps = float(eps_key)

    cmd = [
        GRIDSYNTH_PATH,
        "-e", f"{eps:.15e}",
        "-f", str(EFFORT),
        "--phase",
        "--",
        f"{theta:.15f}",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tokens = "".join(res.stdout.split()).replace("W", "")

    qc = QuantumCircuit(1)
    for t in tokens:
        if t == "H":
            qc.h(0)
        elif t == "T":
            qc.t(0)
        elif t == "S":
            qc.s(0)
        elif t == "X":
            qc.x(0)
        else:
            raise ValueError(t)
    return qc


def rz_via_gridsynth(theta: float, eps: float) -> QuantumCircuit:
    theta = wrap_pi(theta)
    k, err = nearest_k_pi_over_4(theta)
    if abs(err) <= SNAP_TOL:
        qc = QuantumCircuit(1)
        emit_exact_rz_pi_over_4(qc, 0, k)
        return qc

    theta_key = float(f"{theta:.{THETA_KEY_DIGITS}f}")
    eps_key = float(f"{eps:.{EPS_KEY_DIGITS}e}")
    return rz_via_gridsynth_cached(theta_key, eps_key)


@dataclass
class SegOp:
    name: str
    theta: float = 0.0


def canonicalize_1q_unitary(U: np.ndarray) -> List[SegOp]:
    decomp = OneQubitEulerDecomposer("ZXZ")
    qc = decomp(U)
    ops = []
    for inst in qc.data:
        if inst.operation.name == "rz":
            ops.append(SegOp("rz", float(inst.operation.params[0])))
        elif inst.operation.name == "rx":
            ops.append(SegOp("h"))
            ops.append(SegOp("rz", float(inst.operation.params[0])))
            ops.append(SegOp("h"))
        else:
            raise ValueError
    return ops


def plan_rz_angles(qc: QuantumCircuit) -> List[float]:
    acc = [np.eye(2, dtype=complex) for _ in range(qc.num_qubits)]
    angles = []

    def flush(idxs):
        for i in idxs:
            if not np.allclose(acc[i], np.eye(2), atol=1e-12):
                for op in canonicalize_1q_unitary(acc[i]):
                    if op.name == "rz":
                        angles.append(op.theta)
                acc[i] = np.eye(2, dtype=complex)

    for inst in qc.data:
        q = [qc.find_bit(b).index for b in inst.qubits]
        if inst.operation.name == "cx":
            flush(q)
        else:
            acc[q[0]] = Operator(inst.operation).data @ acc[q[0]]

    flush(range(qc.num_qubits))
    return angles


def allocate_eps_rms(angles: List[float], eps_total: float) -> List[float]:
    if not angles:
        return []
    weights = [max(0.08, abs(np.sin(wrap_pi(t) / 2))) for t in angles]
    norm = np.sqrt(sum(w * w for w in weights))
    return [eps_total * w / norm for w in weights]


def synthesize_circuit_from_unitary(U: Operator, eps_total: float) -> QuantumCircuit:
    kak = TwoQubitBasisDecomposer(CXGate())
    base = kak(U)

    angles = plan_rz_angles(base)
    eps_list = allocate_eps_rms(angles, eps_total)
    eps_it = iter(eps_list)

    out = QuantumCircuit(base.num_qubits)
    acc = [np.eye(2, dtype=complex) for _ in range(base.num_qubits)]

    def flush(idxs):
        for i in idxs:
            if not np.allclose(acc[i], np.eye(2), atol=1e-12):
                for op in canonicalize_1q_unitary(acc[i]):
                    if op.name == "h":
                        out.h(i)
                    else:
                        out.compose(rz_via_gridsynth(op.theta, next(eps_it)), [i], inplace=True)
                acc[i] = np.eye(2, dtype=complex)

    for inst in base.data:
        q = [base.find_bit(b).index for b in inst.qubits]
        if inst.operation.name == "cx":
            flush(q)
            out.cx(*q)
        else:
            acc[q[0]] = Operator(inst.operation).data @ acc[q[0]]

    flush(range(base.num_qubits))
    return out


def build_1q_cliffords():
    gens = []
    qcH = QuantumCircuit(1); qcH.h(0)
    qcS = QuantumCircuit(1); qcS.s(0)
    gens = [qcH, qcS]

    reps = [(QuantumCircuit(1), np.eye(2, dtype=complex))]
    seen = [reps[0][1]]

    def is_new(M):
        return not any(np.allclose(M, X, atol=1e-12) for X in seen)

    i = 0
    while i < len(reps):
        qc, M = reps[i]
        i += 1
        for g in gens:
            qn = QuantumCircuit(1)
            qn.compose(qc, inplace=True)
            qn.compose(g, inplace=True)
            Mn = Operator(qn).data
            if is_new(Mn):
                seen.append(Mn)
                reps.append((qn, Mn))
        if len(reps) >= 24:
            break

    out = []
    for qc, M in reps[:24]:
        out.append((qc, M, qc.inverse()))
    return out


def apply_frame(U, L0, L1, R0, R1):
    return np.kron(L0, L1) @ U @ np.kron(R0, R1)


def estimate_frame_cost(U):
    kak = TwoQubitBasisDecomposer(CXGate())
    qc = kak(Operator(U))
    angles = plan_rz_angles(qc)
    cost = 0.0
    for t in angles:
        _, err = nearest_k_pi_over_4(t)
        cost += 1 + 6 * min(1, abs(err) / (np.pi / 8))
        cost += 0.6 * abs(np.sin(t / 2))
    return cost


def pick_top_frames(U, cliffords, n, k, seed=0):
    rng = random.Random(seed)
    mats = [c[1] for c in cliffords]
    best = []
    for _ in range(n):
        idx = [rng.randrange(24) for _ in range(4)]
        U2 = apply_frame(U, *(mats[i] for i in idx))
        best.append((estimate_frame_cost(U2), tuple(idx)))
    best.sort(key=lambda x: x[0])
    return best[:k]


def main():
    U = random_unitary(4, seed=42)
    cliffords = build_1q_cliffords()

    best = None
    for eps in EPS_GRID:
        frames = pick_top_frames(U.data, cliffords, FRAME_SAMPLES, TOP_K_FRAMES)
        for _, (a, b, c, d) in frames:
            L0, L0m, L0d = cliffords[a]
            L1, L1m, L1d = cliffords[b]
            R0, R0m, R0d = cliffords[c]
            R1, R1m, R1d = cliffords[d]

            U2 = apply_frame(U.data, L0m, L1m, R0m, R1m)
            qc = synthesize_circuit_from_unitary(Operator(U2), eps)

            full = QuantumCircuit(2)
            full.compose(L0d, [0], inplace=True)
            full.compose(L1d, [1], inplace=True)
            full.compose(qc, inplace=True)
            full.compose(R0d, [0], inplace=True)
            full.compose(R1d, [1], inplace=True)

            err = phase_insensitive_err(U.data, Operator(full).data)
            t = tcount(full)

            if best is None or (err <= ERR_TARGET and t < best[2]):
                best = (full, err, t)

    qc, err, t = best
    print("T =", t)
    print("err =", err)

    with open("synthesized_minT_frame_search.qasm", "w") as f:
        f.write(qasm2.dumps(qc))


if __name__ == "__main__":
    main()
