"""
QSphera QAOA Server (Qiskit)
============================
Solves a QUBO sent by the app with QAOA on the Qiskit statevector sampler,
and with exact brute force for comparison.

Endpoints:
  GET  /health      server status
  POST /qaoa        QAOA on a QUBO: {"Q": [[...]], "variables": [...], "p": 1, "shots": 2048}
  POST /pipeline    stage 3 only (QAOA), same body inside "prev_output"
  POST /classical   exact brute-force solution of the same QUBO (classical baseline)

QAOA details:
  - QUBO x in {0,1}^n is mapped to an Ising Hamiltonian with x = (1 - z)/2:
      J_ij = Q_ij / 2 (i < j),  h_i = -(Q_ii + sum_{j != i} Q_ij) / 2
  - Circuit: H on all qubits, then p layers of [RZZ(2*gamma*J_ij), RZ(2*gamma*h_i), RX(2*beta)].
  - gamma, beta are tuned by COBYLA to minimize the expected QUBO energy.
  - Qiskit bit order: qubit 0 is the rightmost character of a count key.
    Returned bitstrings are in VARIABLE order (character i = variable i).
Reported quality metrics:
  - probability of measuring the optimal solution (from the final counts),
  - approximation ratio of the expected energy vs the exact optimum,
  - the same probability for uniform random sampling (1 / number of optimal states / 2^n).

Run:
  pip install -r requirements.txt      (qiskit, numpy, scipy, flask, flask-cors)
  python qiskit_qaoa_server.py         (listens on $PORT, default 8765)
Optional: set QSPHERA_API_KEY; requests must then send header X-API-Key.
"""

import json
import os
import itertools

import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.primitives import StatevectorSampler
    from scipy.optimize import minimize
    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    QISKIT_AVAILABLE = False

MAX_QUBITS = 22          # statevector memory and runtime limit for this server
MAX_BRUTE_FORCE = 22

app = Flask(__name__)
CORS(app)
API_KEY = os.environ.get("QSPHERA_API_KEY")


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def json_response(data, status=200):
    return Response(json.dumps(to_native(data)), status=status, mimetype="application/json")


def check_key():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return json_response({"error": "Unauthorized"}, status=401)
    return None


# ---------------------------------------------------------------------------
# QUBO helpers
# ---------------------------------------------------------------------------
def symmetric(Q):
    Q = np.array(Q, dtype=float)
    return (Q + Q.T) / 2.0


def energy(Q, bits):
    x = np.array(bits, dtype=float)
    return float(x @ Q @ x)


def brute_force(Q):
    n = len(Q)
    if n > MAX_BRUTE_FORCE:
        raise ValueError(f"Brute force limited to {MAX_BRUTE_FORCE} variables (got {n}).")
    best_e, best = float("inf"), None
    energies = {}
    for bits in itertools.product([0, 1], repeat=n):
        e = energy(Q, bits)
        energies[bits] = e
        if e < best_e - 1e-12:
            best_e, best = e, list(bits)
    optimal = [list(b) for b, e in energies.items() if abs(e - best_e) <= 1e-9]
    worst_e = max(energies.values())
    mean_e = float(np.mean(list(energies.values())))
    return {"bits": best, "energy": best_e, "optimal_states": optimal, "worst_energy": worst_e, "random_mean_energy": mean_e}


def qubo_to_ising(Q):
    n = len(Q)
    h = np.zeros(n)
    J = {}
    for i in range(n):
        h[i] -= Q[i][i] / 2.0
        for j in range(n):
            if j != i:
                h[i] -= Q[i][j] / 2.0
        for j in range(i + 1, n):
            if Q[i][j] != 0:
                J[(i, j)] = Q[i][j] / 2.0
    return h, J


def bits_from_key(key):
    return [int(c) for c in reversed(key)]   # qubit 0 is the rightmost character


def qaoa_circuit(n, h, J, p):
    g = ParameterVector("g", p)
    b = ParameterVector("b", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for k in range(p):
        for (i, j), Jij in J.items():
            qc.rzz(2 * g[k] * Jij, i, j)
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * g[k] * h[i], i)
        for i in range(n):
            qc.rx(2 * b[k], i)
    qc.measure_all()
    return qc


def run_qaoa(Q, p=1, shots=2048, seed=42, maxiter=100):
    n = len(Q)
    h, J = qubo_to_ising(Q)
    qc = qaoa_circuit(n, h, J, p)
    params = list(qc.parameters)                 # sorted by name: b[0..p-1], g[0..p-1]
    sampler = StatevectorSampler(seed=seed)
    evals = {"n": 0}

    def counts_for(x):
        bind = {params[i]: x[i] for i in range(len(params))}
        return sampler.run([qc.assign_parameters(bind)], shots=shots).result()[0].data.meas.get_counts()

    def expected(x):
        evals["n"] += 1
        c = counts_for(x)
        return sum(energy(Q, bits_from_key(k)) * v for k, v in c.items()) / shots

    x0 = np.full(len(params), 0.5)
    res = minimize(expected, x0, method="COBYLA", options={"maxiter": maxiter})
    counts = counts_for(res.x)
    return counts, res, evals["n"]


def solve(Q_in, variables=None, p=1, shots=2048, seed=42):
    Q = symmetric(Q_in)
    n = len(Q)
    if n == 0:
        return {"error": "Empty QUBO."}
    variables = variables or [f"x{i}" for i in range(n)]
    exact = brute_force(Q) if n <= MAX_BRUTE_FORCE else None

    result = {"n_qubits": n, "p_depth": int(p), "shots": int(shots), "variables": variables, "qiskit_available": QISKIT_AVAILABLE}
    if not QISKIT_AVAILABLE:
        result.update({"method": "brute_force (Qiskit not installed)"})
    elif n > MAX_QUBITS:
        result.update({"method": "not run", "error": f"QAOA limited to {MAX_QUBITS} qubits on this server. Reduce the problem first."})
    else:
        counts, res, n_evals = run_qaoa(Q, p=p, shots=shots, seed=seed)
        total = sum(counts.values())
        best_key = min(counts, key=lambda k: energy(Q, bits_from_key(k)))
        best_bits = bits_from_key(best_key)
        most_key = max(counts, key=counts.get)
        exp_e = sum(energy(Q, bits_from_key(k)) * v for k, v in counts.items()) / total
        result.update({
            "method": "QAOA",
            "bitstring": "".join(str(b) for b in best_bits),
            "bits": best_bits,
            "energy": energy(Q, best_bits),
            "most_probable_bitstring": "".join(str(b) for b in bits_from_key(most_key)),
            "most_probable_share": counts[most_key] / total,
            "expected_energy": exp_e,
            "optimizer": {"name": "COBYLA", "evaluations": n_evals, "angles": list(map(float, res.x))},
            "decisions": [{"variable": v, "value": int(b)} for v, b in zip(variables, best_bits)],
        })
        if exact:
            opt_keys = {"".join(str(b) for b in reversed(o)) for o in exact["optimal_states"]}
            p_opt = sum(v for k, v in counts.items() if k in opt_keys) / total
            span = exact["worst_energy"] - exact["energy"]
            result["quality"] = {
                "probability_optimal": p_opt,
                "random_probability_optimal": len(exact["optimal_states"]) / (2 ** n),
                "approximation_ratio": (exact["worst_energy"] - exp_e) / span if span > 0 else 1.0,
                "random_approximation_ratio": (exact["worst_energy"] - exact["random_mean_energy"]) / span if span > 0 else 1.0,
                "found_optimum": abs(result["energy"] - exact["energy"]) <= 1e-9,
            }
    if exact:
        result["classical"] = {
            "method": "brute_force",
            "bitstring": "".join(str(b) for b in exact["bits"]),
            "energy": exact["energy"],
            "match": ("energy" in result) and abs(result["energy"] - exact["energy"]) <= 1e-9,   # equal energy (ties allowed)
        }
        if result.get("method", "").startswith("brute_force"):
            result["bitstring"] = result["classical"]["bitstring"]
            result["energy"] = exact["energy"]
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return json_response({"status": "ok", "qiskit_available": QISKIT_AVAILABLE, "server": "qsphera_qaoa", "version": "3.0", "max_qubits": MAX_QUBITS})


def read_qubo(data):
    Q = data.get("Q") or data.get("qubo_matrix") or []
    variables = data.get("variables") or data.get("controllable_buses") or data.get("pv_buses")
    if variables and len(variables) != len(Q):
        variables = None
    return Q, variables


@app.route("/qaoa", methods=["POST"])
def qaoa_endpoint():
    denied = check_key()
    if denied:
        return denied
    try:
        data = request.get_json(force=True) or {}
        Q, variables = read_qubo(data)
        if not Q:
            return json_response({"error": "No QUBO matrix provided."}, status=400)
        return json_response(solve(Q, variables, p=int(data.get("p", 1)), shots=int(data.get("shots", 2048)), seed=int(data.get("seed", 42))))
    except Exception as e:
        return json_response({"error": str(e)}, status=500)


@app.route("/pipeline", methods=["POST"])
def pipeline_endpoint():
    denied = check_key()
    if denied:
        return denied
    try:
        data = request.get_json(force=True) or {}
        stage = str(data.get("stage", 3))
        if stage != "3":
            return json_response({"error": "Only stage 3 (QAOA) runs on this server. PCA, ANN and QUBO run in the Base44 backend."}, status=400)
        Q, variables = read_qubo(data.get("prev_output", {}) or data)
        if not Q:
            return json_response({"error": "No QUBO matrix provided from stage 2."}, status=400)
        out = solve(Q, variables, p=int(data.get("p", 1)), shots=int(data.get("shots", 2048)), seed=int(data.get("seed", 42)))
        out["stage"] = 3
        return json_response(out)
    except Exception as e:
        return json_response({"error": str(e)}, status=500)


@app.route("/classical", methods=["POST"])
def classical_endpoint():
    denied = check_key()
    if denied:
        return denied
    try:
        data = request.get_json(force=True) or {}
        Q, variables = read_qubo(data.get("prev_output", {}) or data)
        if not Q:
            return json_response({"error": "No QUBO matrix provided. The classical baseline solves the same QUBO exactly."}, status=400)
        exact = brute_force(symmetric(Q))
        variables = variables or [f"x{i}" for i in range(len(Q))]
        return json_response({
            "method": "brute_force",
            "bitstring": "".join(str(b) for b in exact["bits"]),
            "energy": exact["energy"],
            "decisions": [{"variable": v, "value": int(b)} for v, b in zip(variables, exact["bits"])],
        })
    except Exception as e:
        return json_response({"error": str(e)}, status=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"QSphera QAOA server v3.0 on port {port}. Qiskit available: {QISKIT_AVAILABLE}")
    app.run(host="0.0.0.0", port=port, debug=False)
