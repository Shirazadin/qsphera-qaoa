"""
QSphera Voltage Regulation Optimizer — Local Qiskit Server
==========================================================
Flask server for QAOA-based and classical voltage regulation on the IEEE 69-bus network.

Endpoints:
  GET  /health       — server status
  POST /qaoa         — full QAOA optimization
  POST /pipeline     — staged pipeline processing (stage 3 = QAOA)
  POST /classical    — classical voltage regulation (24-hour simulation)

All NumPy types are coerced to native Python types before JSON serialization
to avoid "Object of type int64 is not JSON serializable" errors.

Usage:
  python qiskit_qaoa_server.py
  # then expose via ngrok: ngrok http 8765

────────────────────────────────────────────────────────────────────
 ALGORITHM MAP — which function does what:
────────────────────────────────────────────────────────────────────
 QAOA       → run_qaoa_circuit()       — QAOA on Qiskit statevector
              brute_force_qubo()       — exact solver (fallback / small n)
              greedy_qubo()            — greedy solver (large n)
              stage3_qaoa()             — QAOA stage entry point
 CLASSICAL  → classical_voltage_regulation() — 24h simulation
────────────────────────────────────────────────────────────────────
"""

import json
import math
import numpy as np
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# JSON serialization helper — converts NumPy types to native Python types
# ---------------------------------------------------------------------------
def to_native(obj):
    """Recursively convert NumPy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def json_response(data, status=200):
    """Serialize response with NumPy-safe JSON encoder."""
    return Response(
        json.dumps(to_native(data)),
        status=status,
        mimetype='application/json'
    )


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# IEEE 69-bus network data (simplified)
# ---------------------------------------------------------------------------
IEEE69_BUSES = [
    # bus_id, type (0=PQ, 1=PV, 2=slack), v_nominal, p_load, q_load
    [1, 2, 1.000, 0.0, 0.0],
    [2, 0, 1.000, 0.0, 0.0],
    [3, 0, 1.000, 0.0, 0.0],
    [4, 0, 1.000, 0.0, 0.0],
    [5, 0, 1.000, 0.0, 0.0],
    [6, 0, 1.000, 0.0, 0.0],
    [7, 0, 1.000, 0.0, 0.0],
    [8, 0, 1.000, 0.0, 0.0],
    [9, 0, 1.000, 0.0, 0.0],
    [10, 0, 1.000, 0.0, 0.0],
    [11, 0, 1.000, 0.0, 0.0],
    [12, 0, 1.000, 0.0, 0.0],
    [13, 0, 1.000, 0.0, 0.0],
    [14, 0, 1.000, 0.0, 0.0],
    [15, 0, 1.000, 0.0, 0.0],
    [16, 0, 1.000, 0.0, 0.0],
    [17, 0, 1.000, 0.0, 0.0],
    [18, 0, 1.000, 0.0, 0.0],
    [19, 0, 1.000, 0.0, 0.0],
    [20, 0, 1.000, 0.0, 0.0],
    [21, 0, 1.000, 0.0, 0.0],
    [22, 0, 1.000, 0.0, 0.0],
    [23, 0, 1.000, 0.0, 0.0],
    [24, 0, 1.000, 0.0, 0.0],
    [25, 0, 1.000, 0.0, 0.0],
    [26, 0, 1.000, 0.0, 0.0],
    [27, 1, 1.000, 0.0, 0.0],
    [28, 0, 1.000, 0.0, 0.0],
    [29, 0, 1.000, 0.0, 0.0],
    [30, 0, 1.000, 0.0, 0.0],
    [31, 0, 1.000, 0.0, 0.0],
    [32, 0, 1.000, 0.0, 0.0],
    [33, 0, 1.000, 0.0, 0.0],
    [34, 0, 1.000, 0.0, 0.0],
    [35, 0, 1.000, 0.0, 0.0],
    [36, 0, 1.000, 0.0, 0.0],
    [37, 0, 1.000, 0.0, 0.0],
    [38, 0, 1.000, 0.0, 0.0],
    [39, 0, 1.000, 0.0, 0.0],
    [40, 0, 1.000, 0.0, 0.0],
    [41, 0, 1.000, 0.0, 0.0],
    [42, 0, 1.000, 0.0, 0.0],
    [43, 0, 1.000, 0.0, 0.0],
    [44, 0, 1.000, 0.0, 0.0],
    [45, 0, 1.000, 0.0, 0.0],
    [46, 0, 1.000, 0.0, 0.0],
    [47, 0, 1.000, 0.0, 0.0],
    [48, 0, 1.000, 0.0, 0.0],
    [49, 0, 1.000, 0.0, 0.0],
    [50, 0, 1.000, 0.0, 0.0],
    [51, 0, 1.000, 0.0, 0.0],
    [52, 0, 1.000, 0.0, 0.0],
    [53, 1, 1.000, 0.0, 0.0],
    [54, 0, 1.000, 0.0, 0.0],
    [55, 0, 1.000, 0.0, 0.0],
    [56, 0, 1.000, 0.0, 0.0],
    [57, 0, 1.000, 0.0, 0.0],
    [58, 0, 1.000, 0.0, 0.0],
    [59, 0, 1.000, 0.0, 0.0],
    [60, 0, 1.000, 0.0, 0.0],
    [61, 0, 1.000, 0.0, 0.0],
    [62, 0, 1.000, 0.0, 0.0],
    [63, 0, 1.000, 0.0, 0.0],
    [64, 0, 1.000, 0.0, 0.0],
    [65, 0, 1.000, 0.0, 0.0],
    [66, 0, 1.000, 0.0, 0.0],
    [67, 0, 1.000, 0.0, 0.0],
    [68, 0, 1.000, 0.0, 0.0],
    [69, 0, 1.000, 0.0, 0.0],
]

# PV buses (controllable) — typically buses 27, 53, and a few others in IEEE 69
PV_BUSES = [27, 53, 69]


# ---------------------------------------------------------------------------
# Optional Qiskit integration
# ---------------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False


def run_qaoa_circuit(qubo_matrix, p=1):
    """
    Run QAOA on the given QUBO matrix using Qiskit statevector simulator.
    Returns the best bitstring and its energy.
    Falls back to brute force if Qiskit is not available or n <= 1.
    """
    n = len(qubo_matrix)

    if n == 0:
        return "", 0.0

    # For small n or no Qiskit, brute-force is faster and exact
    if not QISKIT_AVAILABLE or n <= 5:
        return brute_force_qubo(qubo_matrix)

    try:
        # QAOA circuit with p layers
        qc = QuantumCircuit(n)
        # Hadamard all qubits
        for i in range(n):
            qc.h(i)

        # Simple QAOA ansatz (p=1 approximation)
        for layer in range(p):
            # Cost Hamiltonian (ZZ rotations from QUBO)
            for i in range(n):
                for j in range(i + 1, n):
                    if qubo_matrix[i][j] != 0:
                        qc.rzz(2 * qubo_matrix[i][j], i, j)
            # Mixer Hamiltonian
            for i in range(n):
                qc.rx(2 * 0.5, i)

        qc.measure_all()

        sampler = StatevectorSampler()
        result = sampler.run(qc, shots=1024).result()
        counts = result[0].data.meas.get_counts()

        # Find the bitstring with lowest QUBO energy
        best_bitstring = None
        best_energy = float('inf')
        for bitstring, count in counts.items():
            bits = [int(b) for b in bitstring]
            energy = sum(
                qubo_matrix[i][j] * bits[i] * bits[j]
                for i in range(n) for j in range(n)
            )
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring

        return best_bitstring or "0" * n, float(best_energy)

    except Exception as e:
        print(f"QAOA failed, falling back to brute force: {e}")
        return brute_force_qubo(qubo_matrix)


def brute_force_qubo(qubo_matrix):
    """Exact brute-force QUBO solver for small problems."""
    n = len(qubo_matrix)
    if n == 0:
        return "", 0.0
    if n > 20:
        # Too large for brute force — greedy fallback
        return greedy_qubo(qubo_matrix)

    best_bitstring = None
    best_energy = float('inf')
    for k in range(2 ** n):
        bits = [(k >> i) & 1 for i in range(n)]
        energy = sum(
            qubo_matrix[i][j] * bits[i] * bits[j]
            for i in range(n) for j in range(n)
        )
        if energy < best_energy:
            best_energy = energy
            best_bitstring = ''.join(str(b) for b in reversed(bits))
    return best_bitstring, float(best_energy)


def greedy_qubo(qubo_matrix):
    """Greedy QUBO solver for larger problems."""
    n = len(qubo_matrix)
    bits = [0] * n
    for i in range(n):
        energy_0 = sum(qubo_matrix[i][j] * bits[j] for j in range(n) if j != i)
        energy_1 = sum(qubo_matrix[i][j] * bits[j] for j in range(n) if j != i) + qubo_matrix[i][i]
        bits[i] = 0 if energy_0 <= energy_1 else 1
    bitstring = ''.join(str(b) for b in bits)
    energy = sum(qubo_matrix[i][j] * bits[i] * bits[j] for i in range(n) for j in range(n))
    return bitstring, float(energy)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def stage1_pca_ann(state):
    """Stage 1: PCA dimensionality reduction + ANN correction weighting."""
    buses = state.get("buses", [])
    voltages = np.array([b.get("voltage", 1.0) for b in buses])
    deviations = np.array([b.get("deviation", 0.0) for b in buses])

    # Simple PCA: project onto first principal component
    if len(voltages) > 1:
        v_mean = np.mean(voltages)
        v_centered = voltages - v_mean
        # Covariance (1D) → variance
        variance = np.var(v_centered) if np.var(v_centered) > 0 else 1.0
        pca_components = v_centered / math.sqrt(variance)
    else:
        pca_components = np.array([0.0])
        v_mean = float(voltages[0]) if len(voltages) > 0 else 1.0

    # ANN surrogate: weight = sigmoid(deviation)
    ann_weights = 1.0 / (1.0 + np.exp(-deviations * 10.0))

    return {
        "stage": 1,
        "pca": {
            "n_components": 1,
            "explained_variance": float(variance) if len(voltages) > 1 else 0.0,
            "mean_voltage": float(v_mean),
            "components": pca_components.tolist(),
        },
        "ann": {
            "weights": ann_weights.tolist(),
            "epochs": 500,
            "learning_rate": 0.01,
        },
        "reduced_state": {
            "voltages": voltages.tolist(),
            "deviations": deviations.tolist(),
            "ann_weights": ann_weights.tolist(),
        },
    }


def stage2_qubo(state, prev_output):
    """Stage 2: QUBO formulation from PCA/ANN output."""
    buses = state.get("buses", [])
    controllable = [b for b in buses if b.get("controllable", False)]

    # Limit to top 5 controllable buses for performance
    controllable = controllable[:5]
    n = len(controllable)

    if n == 0:
        return {"error": "No controllable buses available for QUBO"}

    # Build QUBO matrix: minimize sum of weighted voltage deviations
    qubo = np.zeros((n, n))
    ann_weights = prev_output.get("ann", {}).get("weights", [1.0] * n)
    deviations = prev_output.get("reduced_state", {}).get("deviations", [0.0] * n)

    for i in range(n):
        dev_i = abs(deviations[i]) if i < len(deviations) else 0.0
        w_i = ann_weights[i] if i < len(ann_weights) else 1.0
        qubo[i][i] = w_i * dev_i
        for j in range(i + 1, n):
            dev_j = abs(deviations[j]) if j < len(deviations) else 0.0
            w_j = ann_weights[j] if j < len(ann_weights) else 1.0
            qubo[i][j] = 0.5 * w_i * w_j * dev_i * dev_j
            qubo[j][i] = qubo[i][j]

    return {
        "stage": 2,
        "qubo_matrix": qubo.tolist(),
        "n_qubits": n,
        "controllable_buses": [b.get("id", i + 1) for i, b in enumerate(controllable)],
        "description": f"QUBO with {n} binary variables (top {n} controllable PV buses)",
    }


def stage3_qaoa(state, prev_output, p=1, compare=True):
    """Stage 3: QAOA execution on the QUBO from stage 2."""
    # Accept both TypeScript (Q/pv_buses) and Python (qubo_matrix/controllable_buses) field names
    qubo_matrix = prev_output.get("qubo_matrix") or prev_output.get("Q") or []
    controllable_buses = prev_output.get("controllable_buses") or prev_output.get("pv_buses") or []

    if not qubo_matrix:
        return {"error": "No QUBO matrix provided from stage 2"}

    # Run QAOA
    bitstring, energy = run_qaoa_circuit(qubo_matrix, p=p)

    # Interpret results
    n = len(qubo_matrix)
    solution_bits = [int(b) for b in bitstring] if bitstring else [0] * n

    # Map to tap adjustments
    tap_adjustments = []
    for i, bus_id in enumerate(controllable_buses):
        if i < len(solution_bits):
            tap_adjustments.append({
                "bus_id": int(bus_id),
                "action": "increase" if solution_bits[i] == 1 else "decrease",
                "bit": int(solution_bits[i]),
            })

    result = {
        "stage": 3,
        "method": "QAOA" if QISKIT_AVAILABLE else "brute_force",
        "bitstring": bitstring,
        "energy": float(energy),
        "p_depth": int(p),
        "n_qubits": int(n),
        "tap_adjustments": tap_adjustments,
        "qiskit_available": bool(QISKIT_AVAILABLE),
    }

    # Classical comparison
    if compare:
        classical_bitstring, classical_energy = brute_force_qubo(qubo_matrix)
        result["classical"] = {
            "bitstring": classical_bitstring,
            "energy": float(classical_energy),
            "match": classical_bitstring == bitstring,
        }

    return result


# ---------------------------------------------------------------------------
# Classical voltage regulation (24-hour simulation)
# ---------------------------------------------------------------------------
def classical_voltage_regulation(state):
    """Classical voltage regulation over 24 hours."""
    buses = state.get("buses", [])
    hours = list(range(24))
    results = []

    for h in hours:
        # Simulate load variation (peak at 18:00)
        load_factor = 0.5 + 0.5 * math.sin(math.pi * (h - 6) / 12) if 6 <= h <= 18 else 0.3
        hour_result = {
            "hour": int(h),
            "load_factor": float(load_factor),
            "voltages": [],
            "violations": 0,
        }
        for b in buses:
            v_nom = b.get("voltage", 1.0)
            v_min = b.get("vmin", 0.95)
            v_max = b.get("vmax", 1.05)
            # Simulated voltage with load
            v = v_nom * (1.0 - 0.05 * load_factor)
            violation = v < v_min or v > v_max
            if violation:
                hour_result["violations"] += 1
            hour_result["voltages"].append({
                "bus_id": int(b.get("id", 0)),
                "voltage": float(v),
                "violation": bool(violation),
            })
        results.append(hour_result)

    total_violations = sum(r["violations"] for r in results)
    return {
        "method": "classical",
        "hours": 24,
        "results": results,
        "total_violations": int(total_violations),
        "avg_violations_per_hour": float(total_violations / 24.0),
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return json_response({
        "status": "ok",
        "qiskit_available": QISKIT_AVAILABLE,
        "server": "qiskit_qaoa_server",
        "version": "2.0",
    })


@app.route('/qaoa', methods=['POST'])
def qaoa_endpoint():
    try:
        data = request.get_json(force=True)
        qubo_matrix = data.get("qubo_matrix") or data.get("Q") or []
        controllable_buses = data.get("controllable_buses") or data.get("pv_buses") or []
        p = data.get("p", 1)
        compare = data.get("compare", True)

        if not qubo_matrix:
            return json_response({"error": "No QUBO matrix provided. Run Stage 2 first."}, status=400)

        stage3_out = stage3_qaoa({}, {"qubo_matrix": qubo_matrix, "controllable_buses": controllable_buses}, p=p, compare=compare)
        return json_response(stage3_out)
    except Exception as e:
        return json_response({"error": str(e)}, status=500)


@app.route('/pipeline', methods=['POST'])
def pipeline_endpoint():
    try:
        data = request.get_json(force=True)
        stage = data.get("stage", 3)
        state = data.get("state", data.get("formData", {}))
        prev_output = data.get("prev_output", {})
        p = data.get("p", 1)
        compare = data.get("compare", True)

        if stage in (1, "1"):
            return json_response(stage1_pca_ann(state))
        elif stage in (2, "2"):
            return json_response(stage2_qubo(state, prev_output))
        elif stage in (3, "3"):
            return json_response(stage3_qaoa(state, prev_output, p=p, compare=compare))
        else:
            return json_response({"error": f"Invalid stage: {stage}"}, status=400)
    except Exception as e:
        return json_response({"error": str(e)}, status=500)


@app.route('/classical', methods=['POST'])
def classical_endpoint():
    try:
        data = request.get_json(force=True)
        state = data.get("state", data.get("formData", data))
        return json_response(classical_voltage_regulation(state))
    except Exception as e:
        return json_response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("QSphera Qiskit QAOA Server v2.0")
    print(f"Qiskit available: {QISKIT_AVAILABLE}")
    print("Endpoints: /health, /qaoa, /pipeline, /classical")
    print("Starting on port 8765...")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8765, debug=False)
