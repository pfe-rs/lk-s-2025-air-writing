#!/usr/bin/env python3
"""
Plot accuracy and correction stats vs distance_penalty.
"""

from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Plot metrics for distance_penalty sweep.")
    parser.add_argument("--metrics", default="language_model_metrics.csv", help="Path to metrics CSV.")
    parser.add_argument("--output-prefix", default="distance_penalty", help="Prefix for saved plot files.")
    parser.add_argument("--language", choices=["en", "sr"], default="en", help="Language for labels.")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    sns.set_theme(style="whitegrid", context="talk")

    labels = {
        "en": {
            "penalty": "distance_penalty",
            "corrected_acc": "Corrected accuracy",
            "count": "Count",
            "corrections": "Impact of distance_penalty on correction outcomes",
            "helpful": "Helpful corrections",
            "harmful": "Harmful corrections",
            "neutral": "Unchanged",
            "accuracy": "Effect of distance_penalty on correction accuracy",
            "share_title": "Outcome distribution across penalties",
            "share_y": "Share of cases",
        },
        "sr": {
            "penalty": "Kazneni faktor distance",
            "corrected_acc": "Tačnost posle korekcije",
            "count": "Broj slučajeva",
            "corrections": "Uticaj kaznenog faktora na korekcije",
            "helpful": "Korisne korekcije",
            "harmful": "Štetne korekcije",
            "neutral": "Neizmenjene",
            "accuracy": "Uticaj kaznenog faktora na tačnost",
            "share_title": "Raspodela ishoda po kaznenom faktoru",
            "share_y": "Udeo slučajeva",
        },
    }[args.language]

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["penalty"], df["corrected_accuracy"], marker="o", label=labels["corrected_acc"], color="tab:blue")
    plt.xlabel(labels["penalty"])
    plt.ylabel(labels["corrected_acc"] if args.language == "sr" else "Accuracy")
    plt.title(labels["accuracy"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_{args.language}_accuracy.png", dpi=300)
    plt.close()

    # Helpful vs harmful corrections
    plt.figure(figsize=(8, 5))
    plt.plot(df["penalty"], df["helpful"], marker="o", label=labels["helpful"], color="tab:green")
    plt.plot(df["penalty"], df["harmful"], marker="o", label=labels["harmful"], color="tab:red")
    plt.xlabel(labels["penalty"])
    plt.ylabel(labels["count"])
    plt.title(labels["corrections"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_{args.language}_corrections.png", dpi=300)
    plt.close()

    # Outcome share stacked area
    totals = df["total"]
    shares = pd.DataFrame({
        "penalty": df["penalty"],
        labels["helpful"]: df["helpful"] / totals,
        labels["harmful"]: df["harmful"] / totals,
        labels["neutral"]: df["unchanged"] / totals,
    })
    plt.figure(figsize=(8, 5))
    plt.stackplot(
        shares["penalty"],
        shares[labels["helpful"]],
        shares[labels["harmful"]],
        shares[labels["neutral"]],
        labels=[labels["helpful"], labels["harmful"], labels["neutral"]],
        colors=["#2ca02c", "#d62728", "#7f7f7f"],
        alpha=0.8,
    )
    plt.xlabel(labels["penalty"])
    plt.ylabel(labels["share_y"])
    plt.title(labels["share_title"])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_{args.language}_shares.png", dpi=300)
    plt.close()

    print(
        "Saved plots:",
        f"{args.output_prefix}_{args.language}_accuracy.png",
        f"{args.output_prefix}_{args.language}_corrections.png",
        f"{args.output_prefix}_{args.language}_shares.png",
    )


if __name__ == "__main__":
    main()
