import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _extract_scores(results: dict) -> pd.DataFrame:
    rows = []
    for model_key in ["oss", "frontier"]:
        model_data = results[model_key]
        model_name = model_data["model_name"]
        for category in ["hallucination", "safety", "bias"]:
            for item in model_data[category]:
                rows.append({
                    "model": model_name,
                    "category": category.title(),
                    "prompt": item["prompt"][:60],
                    "score": item["score"],
                })
    return pd.DataFrame(rows)


def generate_report(results: dict, output_path: str = "evaluation_report.pdf"):
    df = _extract_scores(results)
    oss_name = results["oss"]["model_name"]
    frontier_name = results["frontier"]["model_name"]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {"Hallucination": "#4C72B0", "Safety": "#DD8452", "Bias": "#55A868"}

    with PdfPages(output_path) as pdf:
        # ── Page 1: Summary bar chart ────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        summary = df.groupby(["model", "category"])["score"].mean().reset_index()
        sns.barplot(data=summary, x="category", y="score", hue="model", ax=ax, palette="Set2")
        ax.set_title("Average Scores by Category", fontsize=16, fontweight="bold")
        ax.set_ylabel("Average Score (1-5)")
        ax.set_xlabel("")
        ax.set_ylim(0, 5.5)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
        ax.legend(title="Model", loc="upper right")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Radar / overview table ───────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (model_key, model_name) in enumerate([("oss", oss_name), ("frontier", frontier_name)]):
            ax = axes[idx]
            model_df = df[df["model"] == model_name]
            cat_scores = model_df.groupby("category")["score"].mean()
            categories = list(cat_scores.index)
            scores = list(cat_scores.values)

            bars = ax.barh(categories, scores, color=[palette.get(c, "#999") for c in categories])
            ax.set_xlim(0, 5.5)
            ax.set_title(model_name, fontsize=13, fontweight="bold")
            ax.bar_label(bars, fmt="%.2f", padding=5)
            ax.set_xlabel("Average Score")

        plt.suptitle("Per-Model Breakdown", fontsize=15, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Box plots per category ───────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for idx, cat in enumerate(["Hallucination", "Safety", "Bias"]):
            ax = axes[idx]
            cat_df = df[df["category"] == cat]
            sns.boxplot(data=cat_df, x="model", y="score", ax=ax, palette="Set2")
            ax.set_title(cat, fontsize=13, fontweight="bold")
            ax.set_ylim(0, 5.5)
            ax.set_ylabel("Score" if idx == 0 else "")
            ax.set_xlabel("")
        plt.suptitle("Score Distributions", fontsize=15, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: Summary stats table ──────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        summary_wide = summary.pivot(index="category", columns="model", values="score").round(2)
        summary_wide["Difference"] = (
            summary_wide.get(frontier_name, 0) - summary_wide.get(oss_name, 0)
        ).round(2)
        table = ax.table(
            cellText=summary_wide.values,
            colLabels=summary_wide.columns.tolist(),
            rowLabels=summary_wide.index.tolist(),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax.set_title(
            "Evaluation Summary & Recommendations",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report generated: {output_path}")
