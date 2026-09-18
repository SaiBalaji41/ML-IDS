"""
Generate publication-quality Experimental Setup & Pipeline Architecture Diagram.
Saves diagram to results/graphs/experimental_setup/experimental_pipeline.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
out_dir = PROJECT_ROOT / "results" / "graphs" / "experimental_setup"
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
ax.set_facecolor("#f8fafc")
fig.patch.set_facecolor("#ffffff")

# Draw Nodes
def draw_box(ax, x, y, w, h, title, text, bg_color, border_color, title_color="#0f172a"):
    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.1",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x, y + h*0.22, title, ha="center", va="center", fontsize=11, fontweight="bold", color=title_color)
    ax.text(x, y - h*0.18, text, ha="center", va="center", fontsize=8.5, color="#334155")

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.8, mutation_scale=15),
    )
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label, ha="center", va="center", fontsize=8, color="#64748b", fontweight="bold")

# Nodes Layout
# Top row: Data Ingestion & Preprocessing
draw_box(ax, 3, 9, 3.8, 1.2, "1. CICIoT2023 Raw Corpus", "33 Network Traffic CSV Files\n7.84M Total Network Flows", "#e0f2fe", "#0284c7")
draw_box(ax, 7.5, 9, 3.8, 1.2, "2. Data Preprocessing", "Sanitization, Drop Zero-Variance\nStandardScaler (Fitted on Train)", "#e0f2fe", "#0284c7")
draw_box(ax, 12, 9, 3.8, 1.2, "3. Stratified Partitioning", "Train: 5.49M (70%)\nVal: 1.17M (15%) | Test: 1.17M (15%)", "#e0f2fe", "#0284c7")

draw_arrow(ax, 4.9, 9, 5.6, 9)
draw_arrow(ax, 9.4, 9, 10.1, 9)

# Middle row: 5 Evaluated Models
draw_arrow(ax, 12, 8.4, 12, 7.6)

draw_box(ax, 2.5, 6.5, 2.2, 1.4, "Random Forest", "200 Trees, Depth=25\nBalanced Weights\n1,568 MB", "#dcfce7", "#16a34a")
draw_box(ax, 5.0, 6.5, 2.2, 1.4, "XGBoost (Top)", "100 Trees, Depth=6\nHistogram Tree\n7.36 MB", "#fef08a", "#ca8a04")
draw_box(ax, 7.5, 6.5, 2.2, 1.4, "1D-CNN", "Conv1D(64)-Pool\nConv1D(128)-Dense\n46.6K Params", "#ffedd5", "#ea580c")
draw_box(ax, 10.0, 6.5, 2.2, 1.4, "BiLSTM", "BiLSTM(64)-BiLSTM(32)\nDense(64)-Drop(0.3)\n81.3K Params", "#f3e8ff", "#9333ea")
draw_box(ax, 12.5, 6.5, 2.2, 1.4, "CNN + BiLSTM", "Conv1D(64)-Pool\nBiLSTM(64)-Dense\n77.0K Params", "#fce7f3", "#db2777")

# Connect partitioning to models
for mx in [2.5, 5.0, 7.5, 10.0, 12.5]:
    draw_arrow(ax, 12, 7.6, mx, 7.2)

# Bottom row: Evaluation, Diagnostics, XAI
draw_box(ax, 3.5, 3.5, 4.2, 1.3, "4. Evaluation & Benchmarking", "1,176,851 Test Flows (34 Classes)\nAcc, Macro/Weighted F1, Latency", "#ede9fe", "#7c3aed")
draw_box(ax, 8.5, 3.5, 4.2, 1.3, "5. Confusion Matrix & Error Triage", "OvR Breakdown (TP, TN, FP, FN)\nSubclass & DoS/DDoS Confusion", "#fee2e2", "#dc2626")
draw_box(ax, 13.0, 3.5, 3.8, 1.3, "6. Explainable AI (SHAP)", "TreeExplainer & Neural Saliency\nGlobal & Local Feature Attributions", "#ccfbf1", "#0d9488")

for mx in [2.5, 5.0, 7.5, 10.0, 12.5]:
    draw_arrow(ax, mx, 5.8, 3.5, 4.15)

draw_arrow(ax, 5.6, 3.5, 6.4, 3.5)
draw_arrow(ax, 10.6, 3.5, 11.1, 3.5)

# Final Outcome Banner
draw_box(ax, 7.5, 1.2, 12.0, 1.0, "7. Validated Research Paper & SOC Incident Decision Support", "Champion: XGBoost (99.24% Acc, 0.7928 Macro F1, 0.0105 ms Latency) | Full XAI Transparency", "#f1f5f9", "#475569")
draw_arrow(ax, 3.5, 2.85, 5.0, 1.7)
draw_arrow(ax, 8.5, 2.85, 7.5, 1.7)
draw_arrow(ax, 13.0, 2.85, 10.0, 1.7)

ax.set_xlim(0, 15.5)
ax.set_ylim(0, 10.2)
ax.axis("off")
plt.title("ML-Powered Intrusion Detection System (IDS): End-to-End Experimental Methodology & Pipeline", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()

save_path = out_dir / "experimental_pipeline.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Experimental pipeline diagram saved -> {save_path}")
