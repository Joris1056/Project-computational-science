#   sensitivity.py – parameter sensitivity analysis for the Parkinson’s CA model
#   
#   This script is used to find out which parameters have the biggest influence on
#   the behaviour of our Parkinson’s disease model.

#   What this script does:
#   - Randomly samples different combinations of key model parameters
#   - Runs the simulation multiple times per parameter setting (to account for randomness)
#   - Measures how fast neurons die and how many neurons die in total
#   - Compares parameters using Spearman rank correlation to see which ones matter most

#   The two main outcomes we look at are:
#   1) Time until 60% of neurons are dead
#   2) Final percentage of dead neurons at the end of the simulation

#   The model itself (Main.py) is not changed by this script.
#   To keep the analysis fast, a smaller grid is used by default.

import os
import numpy as np
import pandas as pd

# Ensure headless plotting (no GUI popups)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Main import ParkinsonSim


def _set_scaled_stage_params(sim: ParkinsonSim, infection_scale: float, degeneration_scale: float, p_cap: float = 0.49):
    """
    Scale the stage-wise infection and degeneration probabilities without editing the model.
    We also clip to keep probabilities valid when multiplied by local_sensitivity.
    """
    # infection
    sim.infection_p_stage1 = float(np.clip(sim.infection_p_stage1 * infection_scale, 0.0, p_cap))
    sim.infection_p_stage2 = float(np.clip(sim.infection_p_stage2 * infection_scale, 0.0, p_cap))
    sim.infection_p_stage3 = float(np.clip(sim.infection_p_stage3 * infection_scale, 0.0, p_cap))
    sim.infection_p_stage4 = float(np.clip(sim.infection_p_stage4 * infection_scale, 0.0, p_cap))
    sim.infection_p_stage5 = float(np.clip(sim.infection_p_stage5 * infection_scale, 0.0, p_cap))

    # degeneration
    sim.degeneration_p_stage1 = float(np.clip(sim.degeneration_p_stage1 * degeneration_scale, 0.0, p_cap))
    sim.degeneration_p_stage2 = float(np.clip(sim.degeneration_p_stage2 * degeneration_scale, 0.0, p_cap))
    sim.degeneration_p_stage3 = float(np.clip(sim.degeneration_p_stage3 * degeneration_scale, 0.0, p_cap))
    sim.degeneration_p_stage4 = float(np.clip(sim.degeneration_p_stage4 * degeneration_scale, 0.0, p_cap))
    sim.degeneration_p_stage5 = float(np.clip(sim.degeneration_p_stage5 * degeneration_scale, 0.0, p_cap))


def run_one(
    *,
    width: int,
    height: int,
    T: int,
    infection_scale: float,
    degeneration_scale: float,
    lateral_ratio_multiplication: float,
    p_spontaneous_degeneration: float,
    seed: int,
) -> dict:
    np.random.seed(seed)

    sim = ParkinsonSim()

    # Keep geometry consistent across runs
    sim.width = width
    sim.height = height

    # Heterogeneity / seeding
    sim.lateral_base_multiplier = 1.0
    sim.lateral_ratio_multiplication = float(lateral_ratio_multiplication)
    sim.p_spontaneous_degeneration = float(p_spontaneous_degeneration)

    # Apply scaling to stage-wise probabilities
    _set_scaled_stage_params(sim, infection_scale, degeneration_scale)

    sim.reset()

    # Run without calling sim.draw() (no visualization)
    for _ in range(T):
        done = sim.step()
        if done:
            break

    dead_series = np.array(sim.neuron_death, dtype=float)  # in percent (0..100)
    if dead_series.size == 0:
        final_dead = 0.0
        t60 = T + 1
    else:
        final_dead = float(dead_series[-1])
        hit = np.where(dead_series >= 60.0)[0]
        t60 = int(sim.time[int(hit[0])]) if hit.size > 0 else (T + 1)

    # Close any figures created in reset() to avoid memory growth
    plt.close("all")

    return {
        "infection_scale": infection_scale,
        "degeneration_scale": degeneration_scale,
        "lateral_ratio_multiplication": lateral_ratio_multiplication,
        "p_spontaneous_degeneration": p_spontaneous_degeneration,
        "seed": seed,
        "T": T,
        "width": width,
        "height": height,
        "time_to_60_dead": t60,
        "final_dead_pct": final_dead,
    }


def screening_experiment(
    *,
    n_sets: int = 30,
    reps: int = 3,
    width: int = 60,
    height: int = 60,
    T: int = 150,
    seed0: int = 1234,
) -> pd.DataFrame:
    """
    Broad screening. Increase n_sets/reps/T/grid later once everything works.
    """
    rows = []
    rng = np.random.default_rng(seed0)

    # Ranges chosen to keep probabilities stable and runs fast
    for i in range(n_sets):
        infection_scale = float(rng.uniform(0.6, 1.4))
        degeneration_scale = float(rng.uniform(0.6, 1.4))
        lateral_ratio = float(rng.uniform(0.0, 1.0))
        p_spont = float(rng.uniform(0.0, 0.02))

        for r in range(reps):
            seed = seed0 + i * 10_000 + r
            rows.append(
                run_one(
                    width=width,
                    height=height,
                    T=T,
                    infection_scale=infection_scale,
                    degeneration_scale=degeneration_scale,
                    lateral_ratio_multiplication=lateral_ratio,
                    p_spontaneous_degeneration=p_spont,
                    seed=seed,
                )
            )

    df = pd.DataFrame(rows)

    # Aggregate replicates per parameter set (mean outcomes)
    group_cols = ["infection_scale", "degeneration_scale", "lateral_ratio_multiplication", "p_spontaneous_degeneration", "T", "width", "height"]
    agg = df.groupby(group_cols, as_index=False).agg(
        time_to_60_dead_mean=("time_to_60_dead", "mean"),
        time_to_60_dead_sd=("time_to_60_dead", "std"),
        final_dead_pct_mean=("final_dead_pct", "mean"),
        final_dead_pct_sd=("final_dead_pct", "std"),
        n_reps=("seed", "count"),
    )
    return agg


def spearman_importance(df: pd.DataFrame) -> pd.DataFrame:
    params = ["infection_scale", "degeneration_scale", "lateral_ratio_multiplication", "p_spontaneous_degeneration"]
    outcomes = ["time_to_60_dead_mean", "final_dead_pct_mean"]

    rows = []
    for out in outcomes:
        for p in params:
            rho = df[[p, out]].corr(method="spearman").iloc[0, 1]
            rows.append({"outcome": out, "parameter": p, "spearman_rho": rho, "abs_rho": abs(rho)})

    return pd.DataFrame(rows)


def plot_importance(imp: pd.DataFrame, outpath: str):
    label_map = {
        "infection_scale": "Infection strength (scale)",
        "degeneration_scale": "Degeneration speed (scale)",
        "p_spontaneous_degeneration": "Spontaneous seeding (p)",
        "lateral_ratio_multiplication": "Heterogeneity (lateral gradient)",
    }
    outcome_map = {
        "time_to_60_dead_mean": "Time to 60% dead (mean)",
        "final_dead_pct_mean": "Final % dead (mean)",
    }

    outcomes = list(imp["outcome"].unique())

    fig, axes = plt.subplots(
        nrows=1, ncols=len(outcomes),
        figsize=(7.5 * len(outcomes), 4.5),
        constrained_layout=True
    )
    if len(outcomes) == 1:
        axes = [axes]

    for ax, out in zip(axes, outcomes):
        sub = imp[imp["outcome"] == out].copy()
        sub["pretty_param"] = sub["parameter"].map(label_map).fillna(sub["parameter"])
        sub = sub.sort_values("abs_rho", ascending=True)  # for horizontal bars 

        y = np.arange(len(sub))
        ax.barh(y, sub["abs_rho"])

        ax.set_yticks(y)
        ax.set_yticklabels(sub["pretty_param"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("|Spearman ρ|")
        ax.set_title(outcome_map.get(out, out))

        # annotate values with sign
        signed = sub["spearman_rho"].values
        for i, (v, s) in enumerate(zip(sub["abs_rho"].values, signed)):
            sign = "+" if s >= 0 else "−"
            ax.text(min(v + 0.02, 0.98), i, f"{sign}{abs(s):.2f}", va="center")

        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Parameter importance (Spearman rank correlation)", fontsize=14)
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)

    df = screening_experiment(
        n_sets=50,
        reps=3,
        width=150,
        height=150,
        T=400,    
        seed0=1234,
    )

    df.to_csv("outputs/screening_results.csv", index=False)

    imp = spearman_importance(df)
    imp.to_csv("outputs/spearman_importance.csv", index=False)

    plot_importance(imp, "outputs/spearman_importance.png")

    print("Wrote:")
    print(" - outputs/screening_results.csv")
    print(" - outputs/spearman_importance.csv")
    print(" - outputs/spearman_importance.png")
    print("\nTop parameters by outcome:")
    for out in imp["outcome"].unique():
        sub = imp[imp["outcome"] == out].sort_values("abs_rho", ascending=False).head(4)
        print("\n", out)
        print(sub[["parameter", "spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
