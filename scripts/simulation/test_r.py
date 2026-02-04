from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")


@dataclass
class Config:
    cache_dir: Path = Path("cache/test_r")
    n: int = 200
    slope: float = 0.05
    noise_sd: float = 0.05
    jump_idx: int = 100
    jump_size: float = 1.0
    boxcar_width: int = 100
    seed: int = 0


def compute_corr(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pearson = pearsonr(t, y).statistic
    spearman = spearmanr(t, y).statistic
    return float(pearson), float(spearman)


def simulate_signal(
    t: np.ndarray,
    slope: float,
    noise_sd: float,
    jump_idx: int,
    jump_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base = slope * t + rng.normal(0.0, noise_sd, size=t.size)
    jumped = base.copy()
    jump_sign = 1.0 if slope >= 0 else -1.0
    jumped[jump_idx] += jump_sign * jump_size
    return base, jumped


def apply_boxcar_jump(
    signal: np.ndarray,
    jump_idx: int,
    jump_size: float,
    width: int,
) -> np.ndarray:
    width = max(1, int(width))
    half = width // 2
    start = max(0, int(jump_idx) - half)
    end = min(signal.size, start + width)
    out = signal.copy()
    out[start:end] += jump_size
    return out


def apply_step_jump(signal: np.ndarray, jump_idx: int, jump_size: float) -> np.ndarray:
    out = signal.copy()
    out[int(jump_idx) :] += jump_size
    return out


def main(config: Config) -> None:
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = cache_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n = int(config.n)
    t = np.arange(n, dtype=float)
    jump_idx = int(np.clip(config.jump_idx, 0, n - 1))
    rng = np.random.default_rng(config.seed)

    directions = (("up", config.slope), ("down", -config.slope))
    drift_cases: list[tuple[str, float, np.ndarray, np.ndarray, float, float, float, float]] = []
    for name, slope in directions:
        base, jumped = simulate_signal(
            t,
            slope=slope,
            noise_sd=config.noise_sd,
            jump_idx=jump_idx,
            jump_size=config.jump_size,
            rng=rng,
        )
        p_base, s_base = compute_corr(t, base)
        p_jump, s_jump = compute_corr(t, jumped)
        drift_cases.append((name, float(slope), base, jumped, p_base, s_base, p_jump, s_jump))

    no_drift_base = rng.normal(0.0, config.noise_sd, size=t.size)
    no_drift_boxcar = apply_boxcar_jump(
        no_drift_base, jump_idx=jump_idx, jump_size=config.jump_size, width=config.boxcar_width
    )
    no_drift_step = apply_step_jump(no_drift_base, jump_idx=jump_idx, jump_size=config.jump_size)
    p_boxcar, s_boxcar = compute_corr(t, no_drift_boxcar)
    p_step, s_step = compute_corr(t, no_drift_step)

    save_payload: dict[str, np.ndarray | float | int] = {"t": t, "jump_idx": jump_idx}
    for name, slope, base, jumped, p_base, s_base, p_jump, s_jump in drift_cases:
        save_payload[f"base_{name}"] = base
        save_payload[f"jump_{name}"] = jumped
        save_payload[f"pearson_base_{name}"] = p_base
        save_payload[f"spearman_base_{name}"] = s_base
        save_payload[f"pearson_jump_{name}"] = p_jump
        save_payload[f"spearman_jump_{name}"] = s_jump
    save_payload.update(
        {
            "no_drift_base": no_drift_base,
            "no_drift_boxcar": no_drift_boxcar,
            "no_drift_step": no_drift_step,
            "pearson_no_drift_boxcar": p_boxcar,
            "spearman_no_drift_boxcar": s_boxcar,
            "pearson_no_drift_step": p_step,
            "spearman_no_drift_step": s_step,
        }
    )
    np.savez(cache_dir / "test_r_example.npz", **save_payload)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for ax, (name, slope, base, jumped, p_base, s_base, p_jump, s_jump) in zip(axes, drift_cases):
        ax.plot(t, base, color="C0", label=f"base r_p={p_base:.3f}, r_s={s_base:.3f}")
        ax.plot(t, jumped, color="C1", label=f"jump r_p={p_jump:.3f}, r_s={s_jump:.3f}")
        ax.axvline(jump_idx, color="k", linestyle="--", linewidth=1)
        ax.set_ylabel("signal")
        ax.set_title(f"{name} drift (slope={slope:.3f})")
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("time index")
    fig.savefig(fig_dir / "test_r_example.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout="constrained")
    ax.plot(t, no_drift_base, color="C0", label="no drift (base)")
    ax.plot(
        t,
        no_drift_boxcar,
        color="C2",
        label=f"boxcar r_p={p_boxcar:.3f}, r_s={s_boxcar:.3f}",
    )
    ax.plot(
        t,
        no_drift_step,
        color="C3",
        label=f"step r_p={p_step:.3f}, r_s={s_step:.3f}",
    )
    ax.axvline(jump_idx, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("time index")
    ax.set_ylabel("signal")
    ax.set_title("no drift: boxcar vs step jump")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(fig_dir / "test_r_no_drift.png", dpi=200)
    plt.close(fig)

    for name, _, _, _, p_base, s_base, p_jump, s_jump in drift_cases:
        print(
            f"{name} drift: base (pearson={p_base:.3f}, spearman={s_base:.3f}) "
            f"jump (pearson={p_jump:.3f}, spearman={s_jump:.3f})"
        )
    print(
        f"no drift: boxcar (pearson={p_boxcar:.3f}, spearman={s_boxcar:.3f}) "
        f"step (pearson={p_step:.3f}, spearman={s_step:.3f})"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
