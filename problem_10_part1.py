import subprocess
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from functools import lru_cache

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import CXGate
from qiskit.synthesis import TwoQubitBasisDecomposer, OneQubitEulerDecomposer

GRIDSYNTH_PATH = "/Users/justinsato/Downloads/gridsynth" 
EFFORT = 1500         
EPS_TOTAL = 1e-10    

@lru_cache(maxsize=None)
def rz_via_gridsynth(theta: float, epsilon: float) -> QuantumCircuit:
    theta_norm = (theta + np.pi) % (2 * np.pi) - np.pi
    
    cmd = [
        GRIDSYNTH_PATH,
        "-e", f"{epsilon:.15e}",
        "-f", str(EFFORT),
        "--phase",
        "--",
        f"{theta_norm:.15f}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Gridsynth failed!\nCommand: {' '.join(cmd)}\nError: {e.stderr}")
        raise e

    tokens = "".join(out.split()).replace("W", "") 
    
    qc = QuantumCircuit(1)
    for t in tokens:
        if t == "H": qc.h(0)
        elif t == "T": qc.t(0)
        elif t == "S": qc.s(0)
        elif t == "X": qc.x(0)
    return qc

@dataclass
class SegOp:
    name: str   
    theta: float = 0.0

def canonicalize_1q_unitary(U2: np.ndarray) -> List[SegOp]:
    decomp = OneQubitEulerDecomposer("ZXZ")
    qc1 = decomp(U2)
    
    ops = []
    for inst in qc1.data:
        name = inst.operation.name
        if name == "rz":
            ops.append(SegOp("rz", float(inst.operation.params[0])))
        elif name == "rx":
            ops.append(SegOp("h"))
            ops.append(SegOp("rz", float(inst.operation.params[0])))
            ops.append(SegOp("h"))
    return ops

def synthesize_circuit(target_u: Operator):
    kak = TwoQubitBasisDecomposer(CXGate())
    base_qc = kak(target_u)
    
    estimated_rz_count = 25 
    eps_per_gate = EPS_TOTAL / estimated_rz_count

    n = base_qc.num_qubits
    final_qc = QuantumCircuit(n)
    
    acc = [np.eye(2, dtype=complex) for _ in range(n)]

    def flush(indices):
        for i in indices:
            if not np.allclose(acc[i], np.eye(2), atol=1e-12):
                ops = canonicalize_1q_unitary(acc[i])
                for op in ops:
                    if op.name == "h":
                        final_qc.h(i)
                    elif op.name == "rz":
                        approx = rz_via_gridsynth(op.theta, eps_per_gate)
                        final_qc.compose(approx, [i], inplace=True)
                acc[i] = np.eye(2, dtype=complex)

    for circ_inst in base_qc.data:
        q_idx = [base_qc.find_bit(q).index for q in circ_inst.qubits]
        inst = circ_inst.operation

        if inst.name == "cx":
            flush(q_idx)
            final_qc.cx(q_idx[0], q_idx[1])
        else:
            gate_mat = Operator(inst).data
            acc[q_idx[0]] = gate_mat @ acc[q_idx[0]]

    flush(range(n))
    return final_qc

def main():
    U_target = random_unitary(4, seed=42)
    print(f"Targeting total error: {EPS_TOTAL}")
    print("Starting synthesis...")
    
    qc_final = synthesize_circuit(U_target)
    
    U_final = Operator(qc_final).data
    
    def phase_insensitive_err(U, V):
        u_vec = U.flatten()
        v_vec = V.flatten()
        dot = np.vdot(v_vec, u_vec)
        phase = dot / abs(dot) if abs(dot) > 1e-15 else 1.0
        return np.linalg.norm(u_vec - phase * v_vec)

    error = phase_insensitive_err(U_target.data, U_final)
    
    print("\n" + "="*30)
    print(f"Synthesis Complete")
    print(f"T-count: {qc_final.count_ops().get('t', 0)}")
    print(f"H-count: {qc_final.count_ops().get('h', 0)}")
    print(f"Frobenius Error: {error:.2e}")
    print("="*30)

if __name__ == "__main__":
    main()
