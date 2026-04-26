from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    acts = data["acts"]
    ep_lengths = data["ep_lengths"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ep_lengths, bins=30)
    axes[0].set_title("Episode Length Distribution")
    axes[0].set_xlabel("length")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.3)

    axes[1].boxplot([acts[:, i] for i in range(acts.shape[1])], showfliers=False)
    axes[1].set_title("Action Distribution by Dimension")
    axes[1].set_xlabel("action dim")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
