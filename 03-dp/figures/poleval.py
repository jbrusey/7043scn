from __future__ import annotations
import numpy as np
from typing import Sequence, List

# -----------------------------
# MDP: 4x4 gridworld
# -----------------------------
N = 4
S = N * N
ACTIONS = ["U", "D", "L", "R"]
ARROWS = ["↑", "↓", "←", "→"]
A = len(ACTIONS)
ABSORB = {0, S - 1}  # top-left, bottom-right

# direction vectors (dx, dy) in cell coordinates
DIRS = {
    0: (0, -1),  # U
    1: (0, 1),   # D
    2: (-1, 0),  # L
    3: (1, 0),   # R
}


def svg_grid_4x4_policy_arrows_from_q(Q: np.ndarray,
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
    if a == 0:
        nr = max(0, r - 1)      # U
    elif a == 1:
        nr = min(N - 1, r + 1)  # D
    elif a == 2:
        nc = max(0, c - 1)      # L
    elif a == 3:
        nc = min(N - 1, c + 1)  # R
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


def as_4x4_grid_from_state_vector(v: np.ndarray) -> List[List[float]]:
    return [[float(v[r * N + c]) for c in range(N)] for r in range(N)]


def dump_iteration_svgs(iter_k: int, V: np.ndarray, Q: np.ndarray, out_dir: str = ".") -> None:
    V_grid = as_4x4_grid_from_state_vector(V)
    with open(f"{out_dir}/eval_{iter_k:05d}_values.svg", "w", encoding="utf-8") as f:
        f.write(svg_grid_4x4_numbers(V_grid))

    with open(f"{out_dir}/eval_{iter_k:05d}_policy.svg", "w", encoding="utf-8") as f:
        f.write(svg_grid_4x4_policy_arrows_from_q(Q))


# -----------------------------
# Iterative policy evaluation with per-iter SVG dumps
# -----------------------------

def policy_evaluation_with_svgs(gamma: float = 1.0,
                                max_sweeps: int = 10_000,
                                out_dir: str = ".",
                                dump_iters: Sequence[int] = (0, 1, 2, 3, 10, 10_000)) -> np.ndarray:
    P, R = build_model()

    pi_probs = np.full((S, A), 1.0 / A)
    for s in ABSORB:
        pi_probs[s, :] = 0.0
        pi_probs[s, 0] = 1.0

    V = np.zeros(S, dtype=float)
    dump_set = set(dump_iters)

    for k in range(max_sweeps + 1):
        if k in dump_set:
            Q = R + gamma * np.einsum("sat,t->sa", P, V)
            dump_iteration_svgs(k, V, Q, out_dir=out_dir)

        if k == max_sweeps:
            break

        V_new = V.copy()
        for s in range(S):
            if s in ABSORB:
                V_new[s] = 0.0
                continue
            v = 0.0
            for a in range(A):
                pa = pi_probs[s, a]
                if pa == 0.0:
                    continue
                v += pa * (R[s, a] + gamma * (P[s, a] @ V))
            V_new[s] = v
        V = V_new

    return V


if __name__ == "__main__":
    policy_evaluation_with_svgs(gamma=1.0, max_sweeps=10_000, out_dir=".")
    print("Wrote eval_XXXXX_values.svg and eval_XXXXX_policy.svg at selected iterations.")
