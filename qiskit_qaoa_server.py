"""
QSphera QAOA Server (Qiskit)
============================
Solves a QUBO sent by the app with QAOA on the Qiskit Aer statevector simulator
(Qiskit Statevector if qiskit-aer is not installed),
and with exact brute force for comparison.

Endpoints:
  GET  /health      server status
  POST /qaoa        QAOA on a QUBO: {"Q": [[...]], "variables": [...], "p": 1, "warm_start_bits": [...] (optional)}
  POST /pipeline    stage 3 only (QAOA), same body inside "prev_output"
  POST /classical   exact brute-force solution of the same QUBO (classical baseline)
  POST /jobs        start QAOA as a background job (same body as /qaoa); returns
                    {"job_id", "status"} at once. Header Idempotency-Key: the same key
                    returns the same job instead of starting a second one.
  GET  /jobs/<id>   job status: queued, running, completed (with "result", the same
                    object /qaoa returns) or failed (with "result": {"error"}).
                    Jobs are stored as files in QSPHERA_JOBS_DIR (default /tmp/qsphera_jobs),
                    so every server worker on the instance sees them. A restart of the
                    instance loses running jobs; their status then reads 404 "Job not found".
  GET  /source      the code this server is running (shown in the app Library)

QAOA details:
  - QUBO x in {0,1}^n is mapped to an Ising Hamiltonian with x = (1 - z)/2:
      J_ij = Q_ij / 2 (i < j),  h_i = -(Q_ii + sum_{j != i} Q_ij) / 2
  - Circuit: H on all qubits, then p layers of [RZZ(2*gamma*J_ij), RZ(2*gamma*h_i), RX(2*beta)].
  - Warm start (when "warm_start_bits" is sent, one bit per variable): warm-start QAOA
    (Egger, Marecek, Woerner, Quantum 5, 479, 2021). Qubit i starts in RY(theta_i)|0>,
    theta_i = 2*arcsin(sqrt(c_i)), c_i = 1 - WARM_EPS for bit 1 and WARM_EPS for bit 0;
    the mixer on qubit i is RY(theta_i) RZ(-2*beta) RY(-theta_i) instead of RX(2*beta).
    The result reports "warm_start": true.
  - The QUBO is normalized (divided by its largest entry); the optimum is unchanged.
  - Depth is at least 2 layers. gamma, beta are tuned by COBYLA from 4 starting
    angle sets; the objective is CVaR 0.1 (mean energy of the lowest-energy 10%
    of the probability), computed from the exact statevector probabilities.
    The best run is kept and measured with 4096 shots.
  - All 2^n energies are computed once (vectorized) and used for the objective,
    the exact optimum and the reported quality.
  - Qiskit bit order: qubit 0 is the rightmost character of a count key.
    Returned bitstrings are in VARIABLE order (character i = variable i).
Reported quality metrics:
  - probability of measuring the optimal solution (from the final counts),
  - approximation ratio of the expected energy vs the exact optimum,
  - the same probability for uniform random sampling (1 / number of optimal states / 2^n).

Run:
  pip install -r requirements.txt      (qiskit, qiskit-aer, numpy, scipy, flask, flask-cors)
  python qiskit_qaoa_server.py         (listens on $PORT, default 8765)
Optional: set QSPHERA_API_KEY; requests must then send header X-API-Key.
"""

import json
import os
import itertools
import time
import hashlib
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

# One thread per simulation: hosted servers share a few CPUs, and extra threads
# compete for them and slow the run down.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    QISKIT_AVAILABLE = False

try:  # Qiskit Aer: same statevector result, faster (C++). Used when installed.
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    AER = AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1)
except Exception:  # pragma: no cover
    AER = None

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


def all_energies(Q):
    """Energy of every bitstring, indexed like Qiskit: bit i of index k is variable i."""
    n = len(Q)
    k = np.arange(2 ** n, dtype=np.int64)
    X = [((k >> i) & 1).astype(float) for i in range(n)]
    E = np.zeros(2 ** n)
    for i in range(n):
        if Q[i][i] != 0:
            E += Q[i][i] * X[i]
        for j in range(i + 1, n):
            if Q[i][j] != 0:
                E += 2.0 * Q[i][j] * X[i] * X[j]
    return E


def index_bits(k, n):
    return [(int(k) >> i) & 1 for i in range(n)]


def brute_force(Q, E=None):
    n = len(Q)
    if n > MAX_BRUTE_FORCE:
        raise ValueError(f"Brute force limited to {MAX_BRUTE_FORCE} variables (got {n}).")
    if E is None:
        E = all_energies(Q)
    best_e = float(E.min())
    opt_idx = np.flatnonzero(np.abs(E - best_e) <= 1e-9)
    return {"bits": index_bits(opt_idx[0], n), "energy": best_e,
            "optimal_states": [index_bits(k, n) for k in opt_idx],
            "worst_energy": float(E.max()), "random_mean_energy": float(E.mean())}


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


def qaoa_circuit(n, h, J, p, thetas=None):
    """thetas: warm-start angles per qubit (None = standard QAOA)."""
    g = ParameterVector("g", p)
    b = ParameterVector("b", p)
    qc = QuantumCircuit(n)
    if thetas is None:
        qc.h(range(n))
    else:
        for i in range(n):
            qc.ry(thetas[i], i)
    for k in range(p):
        for (i, j), Jij in J.items():
            qc.rzz(2 * g[k] * Jij, i, j)
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * g[k] * h[i], i)
        for i in range(n):
            if thetas is None:
                qc.rx(2 * b[k], i)
            else:
                qc.ry(-thetas[i], i)
                qc.rz(-2 * b[k], i)
                qc.ry(thetas[i], i)
    qc.measure_all()
    return qc


WARM_EPS = 0.25          # warm start: probability kept on the opposite bit value (algorithm setting)


def warm_thetas(bits):
    c = [1 - WARM_EPS if int(x) == 1 else WARM_EPS for x in bits]
    return [2 * float(np.arcsin(np.sqrt(ci))) for ci in c]


MIN_DEPTH = 2            # QAOA layers used at least (depth 1 is the weakest setting)
RESTARTS = 4             # starting angle sets; the best optimized run is kept
CVAR_ALPHA = 0.1         # optimize the mean of the best 10% of the probability (CVaR)
FINAL_SHOTS = 4096       # samples measured with the final angles


def run_qaoa(Q, E, p=1, seed=42, maxiter=150, warm_bits=None):
    """QAOA with a normalized QUBO, depth >= MIN_DEPTH, several starting angles
    and a CVaR objective. The objective is computed from the exact Qiskit
    statevector probabilities (no shot noise). Returns final counts, the best
    optimizer result, the number of circuit evaluations and the settings used."""
    n = len(Q)
    scale = float(np.max(np.abs(Q))) or 1.0
    h, J = qubo_to_ising(Q / scale)              # same optimum, angles on a common scale
    depth = max(int(p), MIN_DEPTH)
    thetas = warm_thetas(warm_bits) if warm_bits is not None else None
    qc = qaoa_circuit(n, h, J, depth, thetas)
    qc.remove_final_measurements()
    params = list(qc.parameters)                 # sorted by name: b[0..p-1], g[0..p-1]
    if AER is not None:
        qc_aer = qc.copy()
        qc_aer.save_statevector()
        qc_aer = transpile(qc_aer, AER)
        params_aer = list(qc_aer.parameters)
    order = np.argsort(E)                        # bitstrings from lowest to highest energy
    E_sorted = E[order] / scale
    evals = {"n": 0}

    def probs_for(x):
        if AER is not None:
            bind = {params_aer[i]: x[i] for i in range(len(params_aer))}
            sv = AER.run(qc_aer.assign_parameters(bind)).result().get_statevector()
            return np.abs(np.asarray(sv)) ** 2
        bind = {params[i]: x[i] for i in range(len(params))}
        return Statevector(qc.assign_parameters(bind)).probabilities()

    def cvar(x):
        evals["n"] += 1
        pr = probs_for(x)[order]
        cum = np.cumsum(pr)
        m = int(np.searchsorted(cum, CVAR_ALPHA)) + 1
        w = pr[:m].copy()
        w[-1] -= cum[m - 1] - CVAR_ALPHA if cum[m - 1] > CVAR_ALPHA else 0.0
        return float(np.dot(w, E_sorted[:m]) / w.sum())

    # Starting angles: an annealing-like ramp (gamma rising, beta falling) at
    # several overall sizes, so different runs start in different regions.
    starts = []
    for size in np.linspace(0.3, 1.2, RESTARTS):
        ramp = (np.arange(depth) + 0.5) / depth
        starts.append(np.concatenate([size * 0.5 * (1 - ramp), size * ramp]))   # b..., g... order
    best = None
    for x0 in starts:
        res = minimize(cvar, x0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    # Final measurement: FINAL_SHOTS samples from the best state.
    pr = probs_for(best.x)
    pr = pr / pr.sum()
    rng = np.random.default_rng(seed)
    idx, cnt = np.unique(rng.choice(len(pr), size=FINAL_SHOTS, p=pr), return_counts=True)
    counts = {format(int(k), f"0{n}b"): int(c) for k, c in zip(idx, cnt)}   # Qiskit key format
    settings = {"simulator": "Qiskit Aer statevector" if AER is not None else "Qiskit Statevector", "depth": depth, "restarts": RESTARTS, "objective": f"CVaR {CVAR_ALPHA}", "qubo_scale": scale, "final_shots": FINAL_SHOTS, "warm_start": thetas is not None}
    if thetas is not None:
        settings["warm_eps"] = WARM_EPS
    return counts, best, evals["n"], settings


def solve(Q_in, variables=None, p=1, shots=2048, seed=42, warm_bits=None):
    t0 = time.time()
    Q = symmetric(Q_in)
    n = len(Q)
    if n == 0:
        return {"error": "Empty QUBO."}
    variables = variables or [f"x{i}" for i in range(n)]
    if warm_bits is not None:
        if len(warm_bits) != n or any(int(x) not in (0, 1) for x in warm_bits):
            return {"error": f"warm_start_bits must have {n} values of 0 or 1 (got {len(warm_bits)})."}
        warm_bits = [int(x) for x in warm_bits]
    E = all_energies(Q)
    exact = brute_force(Q, E) if n <= MAX_BRUTE_FORCE else None

    result = {"n_qubits": n, "p_depth": max(int(p), MIN_DEPTH), "shots": FINAL_SHOTS, "variables": variables, "qiskit_available": QISKIT_AVAILABLE}
    if not QISKIT_AVAILABLE:
        result.update({"method": "brute_force (Qiskit not installed)"})
    elif n > MAX_QUBITS:
        result.update({"method": "not run", "error": f"QAOA limited to {MAX_QUBITS} qubits on this server. Reduce the problem first."})
    else:
        counts, res, n_evals, settings = run_qaoa(Q, E, p=p, seed=seed, warm_bits=warm_bits)
        total = sum(counts.values())
        best_key = min(counts, key=lambda k: E[int(k, 2)])
        best_bits = bits_from_key(best_key)
        most_key = max(counts, key=counts.get)
        exp_e = sum(E[int(k, 2)] * v for k, v in counts.items()) / total
        result.update({
            "method": "QAOA (warm start)" if warm_bits is not None else "QAOA",
            "warm_start": warm_bits is not None,
            "bitstring": "".join(str(b) for b in best_bits),
            "bits": best_bits,
            "energy": energy(Q, best_bits),
            "most_probable_bitstring": "".join(str(b) for b in bits_from_key(most_key)),
            "most_probable_share": counts[most_key] / total,
            "expected_energy": exp_e,
            "optimizer": {"name": "COBYLA", "evaluations": n_evals, "angles": list(map(float, res.x)), **settings},
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
    result["seconds"] = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return json_response({"status": "ok", "qiskit_available": QISKIT_AVAILABLE, "aer_available": AER is not None, "server": "qsphera_qaoa", "version": "3.4", "max_qubits": MAX_QUBITS, "async_jobs": True})


@app.route("/source", methods=["GET"])
def source_endpoint():
    """The code this server is running, for the app's Library."""
    denied = check_key()
    if denied:
        return denied
    path = os.path.abspath(__file__)
    with open(path, encoding="utf-8") as f:
        code = f.read()
    import datetime
    modified = datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).isoformat()
    return json_response({"file": "qiskit_qaoa_server.py", "code": code, "modified": modified})


def read_qubo(data):
    Q = data.get("Q") or data.get("qubo_matrix") or []
    variables = data.get("variables") or data.get("controllable_buses") or data.get("pv_buses")
    if variables and len(variables) != len(Q):
        variables = None
    return Q, variables


def read_warm(data):
    w = data.get("warm_start_bits")
    return list(w) if isinstance(w, (list, tuple)) and len(w) else None


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
        return json_response(solve(Q, variables, p=int(data.get("p", 1)), shots=int(data.get("shots", 2048)), seed=int(data.get("seed", 42)), warm_bits=read_warm(data)))
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
        src = data.get("prev_output", {}) or data
        Q, variables = read_qubo(src)
        if not Q:
            return json_response({"error": "No QUBO matrix provided from stage 2."}, status=400)
        out = solve(Q, variables, p=int(data.get("p", 1)), shots=int(data.get("shots", 2048)), seed=int(data.get("seed", 42)), warm_bits=read_warm(data) or read_warm(src))
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


# ---------------------------------------------------------------------------
# Background jobs (no request time limit on the computation)
# ---------------------------------------------------------------------------
JOBS_DIR = os.environ.get("QSPHERA_JOBS_DIR", "/tmp/qsphera_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
JOB_WORKERS = 1          # QAOA runs at a time per server process; others wait as "queued"
EXECUTOR = ThreadPoolExecutor(max_workers=JOB_WORKERS)
JOB_LOCK = threading.Lock()


def job_path(job_id):
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def write_job(job):
    tmp = job_path(job["job_id"]) + f".{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_native(job), f)
    os.replace(tmp, job_path(job["job_id"]))


def read_job(job_id):
    try:
        with open(job_path(job_id), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def run_job(job_id, Q, variables, p, shots, seed, warm_bits):
    job = read_job(job_id) or {"job_id": job_id}
    job.update({"status": "running", "started": time.time()})
    write_job(job)
    try:
        result = solve(Q, variables, p=p, shots=shots, seed=seed, warm_bits=warm_bits)
        status = "failed" if result.get("error") else "completed"
    except Exception as e:  # the job reports the error; nothing is returned as a result
        result, status = {"error": str(e)}, "failed"
    job.update({"status": status, "finished": time.time(), "result": result})
    write_job(job)


@app.route("/jobs", methods=["POST"])
def jobs_submit():
    denied = check_key()
    if denied:
        return denied
    try:
        data = request.get_json(force=True) or {}
        Q, variables = read_qubo(data)
        if not Q:
            return json_response({"error": "No QUBO matrix provided."}, status=400)
        key = request.headers.get("Idempotency-Key") or uuid.uuid4().hex
        job_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        with JOB_LOCK:
            try:   # create the job file only if it does not exist (same key = same job)
                fd = os.open(job_path(job_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                job = read_job(job_id) or {"job_id": job_id, "status": "queued"}
                return json_response({"job_id": job_id, "status": job.get("status"), "result": job.get("result")})
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"job_id": job_id, "status": "queued", "created": time.time()}, f)
        EXECUTOR.submit(run_job, job_id, Q, variables, int(data.get("p", 1)), int(data.get("shots", 2048)),
                        int(data.get("seed", 42)), read_warm(data))
        return json_response({"job_id": job_id, "status": "queued"}, status=202)
    except Exception as e:
        return json_response({"error": str(e)}, status=500)


@app.route("/jobs/<job_id>", methods=["GET"])
def jobs_status(job_id):
    denied = check_key()
    if denied:
        return denied
    if not all(c in "0123456789abcdef" for c in job_id) or len(job_id) != 64:
        return json_response({"error": "Invalid job identifier."}, status=400)
    job = read_job(job_id)
    if job is None:
        return json_response({"error": "Job not found. The server may have restarted; run the QAOA step again."}, status=404)
    return json_response({"job_id": job_id, "status": job.get("status"), "result": job.get("result")})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"QSphera QAOA server v3.4 on port {port}. Qiskit available: {QISKIT_AVAILABLE}")
    app.run(host="0.0.0.0", port=port, debug=False)
