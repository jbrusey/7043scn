from __future__ import annotations
import numpy as np
from typing import Sequence, List

# -----------------------------
# MDP: 4x4 gridworld
# -----------------------------
N = 4
S = N * N
ACTIONS = ["U", "D", "L", "R"]
ARROWS  = ["↑", "↓", "←", "→"]
A = len(ACTIONS)
ABSORB = {0, S - 1}  # top-left, bottom-right

# direction vectors (dx, dy) in cell coordinates
DIRS = {
    0: (0, -1),  # U
    1: (0,  1),  # D
    2: (-1, 0),  # L
    3: (1,  0),  # R
}

def svg_grid_4x4_policy_arrows(Q: np.ndarray,
                               cell: int = 90, pad: int = 16,
                               stroke: str = "#111",
                               bg: str = "white",
                               tol: float = 1e-12) -> str:
    width = pad * 2 + cell * 4
    height = pad * 2 + cell * 4
    parts: List[str] = []

    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}">')

    # Arrow marker
    parts.append("""
    <defs>
      <marker id="arrowhead" markerWidth="6" markerHeight="4"
        refX="5.5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="#111"/>
      </marker>
    </defs>
    """)

    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')

    for s in range(S):
        r, c = divmod(s, N)
        x = pad + c * cell
        y = pad + r * cell
        cx = x + cell / 2
        cy = y + cell / 2

        # cell box
        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
            f'fill="{bg}" stroke="{stroke}" stroke-width="1"/>'
        )

        if s in ABSORB:
            continue

        q = Q[s]
        qmax = q.max()
        actions = [a for a in range(A) if abs(q[a] - qmax) <= tol]

        for a in actions:
            dx, dy = DIRS[a]
            length = 0.35 * cell
            x2 = cx + dx * length
            y2 = cy + dy * length

            parts.append(
                f'<line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" '
                f'stroke="#111" stroke-width="2.5" '
                f'marker-end="url(#arrowhead)"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)

def step(s: int, a: int) -> tuple[int, float]:
    if s in ABSORB:
        return s, 0.0
    r, c = divmod(s, N)
    nr, nc = r, c
    if a == 0:   nr = max(0, r - 1)      # U
    elif a == 1: nr = min(N - 1, r + 1)  # D
    elif a == 2: nc = max(0, c - 1)      # L
    elif a == 3: nc = min(N - 1, c + 1)  # R
    s2 = nr * N + nc
    return s2, -1.0

def build_model() -> tuple[np.ndarray, np.ndarray]:
    P = np.zeros((S, A, S), dtype=float)
    R = np.zeros((S, A), dtype=float)
    for s in range(S):
        for a in range(A):
            s2, r = step(s, a)
            P[s, a, s2] = 1.0
            R[s, a] = r
    return P, R

def greedy_action_mask(Q: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    # mask[s,a] = True if a is greedy (within tol of max)
    m = Q.max(axis=1, keepdims=True)
    return np.abs(Q - m) <= tol

# -----------------------------
# DP policy evaluation (iterative)
# -----------------------------
def policy_evaluation_dp_probs(P: np.ndarray, R: np.ndarray, pi_probs: np.ndarray, gamma: float,
                               V_init: np.ndarray | None = None,
                               max_sweeps: int = 10_000, tol: float = 1e-10) -> np.ndarray:
    V = np.zeros(S) if V_init is None else V_init.copy()

    for _ in range(max_sweeps):
        V_new = V.copy()
        delta = 0.0
        for s in range(S):
            if s in ABSORB:
                V_new[s] = 0.0
                continue

            # Bellman expectation backup under stochastic policy
            v = 0.0
            for a in range(A):
                pa = pi_probs[s, a]
                if pa == 0.0:
                    continue
                v += pa * (R[s, a] + gamma * (P[s, a] @ V))
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < tol:
            break
    return V

# def policy_evaluation_dp(P: np.ndarray, R: np.ndarray, pi: np.ndarray, gamma: float,
#                          max_sweeps: int = 10_000, tol: float = 1e-10) -> np.ndarray:
#     """
#     Iterative (DP) evaluation of V_pi via Bellman expectation backups.
#     Stops when max change across states < tol.
#     """
#     V = np.zeros(S, dtype=float)
#     for _ in range(max_sweeps):
#         delta = 0.0
#         for s in range(S):
#             if s in ABSORB:
#                 continue
#             a = int(pi[s])
#             v_new = R[s, a] + gamma * (P[s, a] @ V)
#             delta = max(delta, abs(v_new - V[s]))
#             V[s] = v_new
#         if delta < tol:
#             break
#     return V

# -----------------------------
# Svg helpers
# -----------------------------
def svg_grid_4x4_numbers(values: Sequence[Sequence[float]],
                         cell: int = 90, pad: int = 16,
                         font_size: int = 20,
                         stroke: str = "#111",
                         text_color: str = "#111",
                         bg: str = "white") -> str:
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise ValueError("values must be 4x4")

    width = pad * 2 + cell * 4
    height = pad * 2 + cell * 4
    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')

    for r in range(4):
        for c in range(4):
            x = pad + c * cell
            y = pad + r * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'fill="{bg}" stroke="{stroke}" stroke-width="1"/>')
            cx = x + cell / 2
            cy = y + cell / 2
            txt = f"{values[r][c]:.1f}"
            parts.append(
                f'<text x="{cx}" y="{cy}" fill="{text_color}" font-size="{font_size}" '
                f'font-family="Helvetica, Arial, sans-serif" text-anchor="middle" '
                f'dominant-baseline="middle">{txt}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)

def svg_grid_4x4_text(values: Sequence[Sequence[str]],
                      cell: int = 90, pad: int = 16,
                      font_size: int = 28,
                      stroke: str = "#111",
                      text_color: str = "#111",
                      bg: str = "white") -> str:
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise ValueError("values must be 4x4")

    width = pad * 2 + cell * 4
    height = pad * 2 + cell * 4
    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')

    for r in range(4):
        for c in range(4):
            x = pad + c * cell
            y = pad + r * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'fill="{bg}" stroke="{stroke}" stroke-width="1"/>')
            cx = x + cell / 2
            cy = y + cell / 2
            txt = values[r][c]
            parts.append(
                f'<text x="{cx}" y="{cy}" fill="{text_color}" font-size="{font_size}" '
                f'font-family="Helvetica, Arial, sans-serif" text-anchor="middle" '
                f'dominant-baseline="middle">{txt}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)

def as_4x4_grid_from_state_vector(v: np.ndarray) -> List[List[float]]:
    return [[float(v[r*N + c]) for c in range(N)] for r in range(N)]

# -----------------------------
# Policy iteration with per-iter SVG dumps
# -----------------------------
def max_action_arrows(Q_row: np.ndarray, tol: float = 1e-12) -> str:
    m = float(np.max(Q_row))
    idxs = [i for i, q in enumerate(Q_row) if abs(float(q) - m) <= tol]
    return "".join(ARROWS[i] for i in idxs)

def dump_iteration_svgs(iter_k: int, V: np.ndarray, Q: np.ndarray, out_dir: str = ".") -> None:
    V_grid = as_4x4_grid_from_state_vector(V)
    with open(f"{out_dir}/iter_{iter_k:02d}_values.svg", "w", encoding="utf-8") as f:
        f.write(svg_grid_4x4_numbers(V_grid))
        
    with open(f"{out_dir}/iter_{iter_k:02d}_policy.svg", "w", encoding="utf-8") as f:
        f.write(svg_grid_4x4_policy_arrows(Q))
        

def policy_iteration_with_svgs(gamma: float = 1.0, max_iters: int = 50, out_dir: str = ".",
                               eval_tol: float = 1e-10, eval_max_sweeps: int = 10_000,
                               tie_tol: float = 1e-12):
    P, R = build_model()

    V = np.zeros(S, dtype=float)
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    pi_probs = np.full((S, A), 1.0 / A)
    for s in ABSORB:
        pi_probs[s, :] = 0.0
        pi_probs[s, 0] = 1.0   # arbitrary

    for k in range(max_iters):
        dump_iteration_svgs(k, V, Q, out_dir=out_dir)
    
        V = policy_evaluation_dp_probs(P, R, pi_probs, gamma, max_sweeps=eval_max_sweeps, tol=eval_tol, V_init=V)

        Q = R + gamma * np.einsum("sat,t->sa", P, V)
        if k == 0:
            print("V:", V.reshape(4,4))
            for s in [1,4]:
                print("state", s, "Q:", Q[s], "best:", np.max(Q[s]))
            for a,name in enumerate(["U","D","L","R"]):
                s2 = int(np.argmax(P[1,a]))
                print("state 1", name, "->", s2, "V[next]=", V[s2])
        mask = greedy_action_mask(Q, tol=tie_tol)
        pi_probs = mask / mask.sum(axis=1, keepdims=True)  # uniform over max actions
        for s in ABSORB:
            pi_probs[s, :] = 0.0
            pi_probs[s, 0] = 1.0

    dump_iteration_svgs(max_iters, V, Q, out_dir=out_dir)
            
    return pi_probs, V

if __name__ == "__main__":
    pi, V = policy_iteration_with_svgs(gamma=1.0, max_iters=10, eval_max_sweeps=10_000, out_dir=".")
    print("Wrote iter_XX_values.svg and iter_XX_policy.svg for each iteration.")
