"""Create the v10.7.1 audit report and publication-ready English figures."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from prepare_sensitivity_v1071 import BASE, CONDITIONS

SIM = Path(__file__).resolve().parent
V107 = SIM / "results/clinical_window_v10_7_confirmation"
REPORT = BASE / "report"
PUBLICATION = BASE / "publication_figures"
ARXIV_FIG = SIM.parents[1] / "arXiv_tech_report" / "figures"
SUMMER_FIG = SIM.parents[1] / "暑研论文" / "figures"


def configure_font() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    chosen = "Times New Roman" if "Times New Roman" in available else "Liberation Serif"
    plt.rcParams.update({
        "font.family": chosen, "font.size": 10.5, "axes.titlesize": 11.5,
        "axes.labelsize": 10.5, "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return chosen


def save(fig, name: str) -> None:
    PUBLICATION.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(PUBLICATION / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(values, seed=202608170704, samples=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def load_replication_rows(controller: str):
    rows = {}
    for path in (V107 / "shards" / "replication" / controller).glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8")); rows[row["scenario_id"]] = row
    return rows


def figure_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (.02, .34, .16, .32, "Planar state\n+ six targets"),
        (.225, .34, .16, .32, "Frozen learned\ntarget ranker"),
        (.43, .34, .18, .32, "Exact full-episode\nsafety shield"),
        (.655, .34, .15, .32, "Deterministic\nlow-level transfer"),
        (.85, .34, .13, .32, "Next cut\ntarget"),
    ]
    colors = ["#e8f1fa", "#dcebdc", "#fce8dd", "#eee7f5", "#e8f1fa"]
    for (x, y, w, h, label), color in zip(boxes, colors):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    facecolor=color, edgecolor="#333333", linewidth=1.0))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center")
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+left[2], .50), (right[0], .50),
                                     arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
                                     color="#333333"))
    ax.text(.52, .78, "Condition-specific constraint", ha="center", va="center", weight="bold")
    ax.text(.52, .91, r"$B_{budget}=B_{S,condition}+16.07$ mL; fixed automatic clamp/unclamp",
            ha="center", va="center")
    ax.add_patch(FancyArrowPatch((.52, .84), (.52, .67), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.0, color="#333333"))
    ax.text(.5, .12, "The learned ranker proposes order; the policy-external shield decides admissibility.",
            ha="center", va="center", style="italic")
    save(fig, "method_overview")


def figure_replication_effects() -> None:
    controllers = ["C2", "C3", "C4", "C5"]
    labels = ["Myopic + shield", "Corrected teacher", "Learned + shield", "Learned, no shield"]
    c0 = load_replication_rows("C0"); ids = sorted(c0)
    time_means=[]; time_ci=[]; blood_means=[]; blood_ci=[]
    for index, controller in enumerate(controllers):
        rows=load_replication_rows(controller)
        dt=np.asarray([rows[s]["elapsed_minutes"]-c0[s]["elapsed_minutes"] for s in ids])
        db=np.asarray([rows[s]["realized_episode_B_ml"]-c0[s]["realized_episode_B_ml"] for s in ids])
        time_means.append(dt.mean()); time_ci.append(bootstrap_ci(dt, 202608170704+index))
        blood_means.append(db.mean()); blood_ci.append(bootstrap_ci(db, 202608170804+index))
    fig, axes=plt.subplots(1,2,figsize=(10.5,3.7)); x=np.arange(len(labels))
    for ax, means, cis, ylabel in ((axes[0],time_means,time_ci,"Paired time difference (min)"),
                                    (axes[1],blood_means,blood_ci,"Paired simulated blood difference (mL)")):
        err=np.asarray([[m-ci[0] for m,ci in zip(means,cis)],[ci[1]-m for m,ci in zip(means,cis)]])
        ax.errorbar(x,means,yerr=err,fmt="o",capsize=3,color="#2f5d8a")
        ax.axhline(0,color="#444444",linewidth=.8,linestyle="--")
        ax.set_xticks(x,labels,rotation=18,ha="right"); ax.set_ylabel(ylabel); ax.grid(axis="y",alpha=.25)
    axes[0].set_title("Time relative to serpentine baseline")
    axes[1].set_title("Simulated blood relative to serpentine baseline")
    fig.tight_layout(); save(fig,"replication_controller_effects")


def figure_replication_paired() -> None:
    c0=load_replication_rows("C0"); c4=load_replication_rows("C4"); ids=sorted(c0)
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8))
    for ax,field,label,unit in ((axes[0],"elapsed_minutes","Procedure time","min"),
                                (axes[1],"realized_episode_B_ml","Simulated blood loss","mL")):
        x=np.asarray([c0[s][field] for s in ids]); y=np.asarray([c4[s][field] for s in ids])
        lo=min(x.min(),y.min()); hi=max(x.max(),y.max())
        ax.scatter(x,y,s=13,alpha=.55,color="#2f5d8a",edgecolors="none")
        ax.plot([lo,hi],[lo,hi],"--",color="#555555",linewidth=.9)
        ax.set_xlabel(f"Serpentine baseline ({unit})"); ax.set_ylabel(f"Learned + shield ({unit})")
        ax.set_title(label); ax.grid(alpha=.2)
    fig.tight_layout(); save(fig,"replication_paired_results")


def figure_shield_ablation() -> None:
    c0=load_replication_rows("C0"); c4=load_replication_rows("C4"); c5=load_replication_rows("C5")
    ids=sorted(c0); margin=16.07054347826075
    d4=np.asarray([c4[s]["realized_episode_B_ml"]-c0[s]["realized_episode_B_ml"] for s in ids])
    d5=np.asarray([c5[s]["realized_episode_B_ml"]-c0[s]["realized_episode_B_ml"] for s in ids])
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.7))
    axes[0].boxplot([d4,d5],labels=["Learned + shield","Learned, no shield"],showfliers=True)
    axes[0].axhline(margin,color="#a33b3b",linestyle="--",linewidth=1,label="Safety margin")
    axes[0].set_ylabel("Paired simulated blood difference (mL)"); axes[0].legend(); axes[0].grid(axis="y",alpha=.2)
    counts=[int((d4>margin+1e-9).sum()),int((d5>margin+1e-9).sum())]
    bars=axes[1].bar(["Learned + shield","Learned, no shield"],counts,color=["#4b8b64","#b65b5b"])
    axes[1].bar_label(bars); axes[1].set_ylabel("Scenes exceeding safety margin")
    axes[1].set_ylim(0,max(counts)+5); axes[1].grid(axis="y",alpha=.2)
    fig.tight_layout(); save(fig,"shield_ablation")


def figure_corrected_sensitivity(stats: dict) -> None:
    conditions=list(CONDITIONS); y=np.arange(len(conditions)); fig,axes=plt.subplots(1,2,figsize=(10.2,4.2),sharey=True)
    for ax,key,title,color in ((axes[0],"C4_minus_C0","Learned + shield vs serpentine","#2f5d8a"),
                               (axes[1],"C4_minus_C2","Learned + shield vs myopic + shield","#8b4a75")):
        means=[]; low=[]; high=[]
        for condition in conditions:
            item=stats["conditions"][condition][key]; mean=item["mean_delta_T_min"]; ci=item["delta_T_95_ci"]
            means.append(mean); low.append(mean-ci[0]); high.append(ci[1]-mean)
        ax.errorbar(means,y,xerr=np.asarray([low,high]),fmt="o",capsize=3,color=color)
        ax.axvline(0,color="#444444",linestyle="--",linewidth=.8)
        ax.set_xlabel("Paired time difference (min)"); ax.set_title(title); ax.grid(axis="x",alpha=.25)
    labels=["S0: 15/5, p=1.00","S1: 12/5, p=1.00","S2: 10/5, p=1.00",
            "S3: 15/5, p=0.50","S4: 15/5, p=0.25"]
    axes[0].set_yticks(y,labels); axes[0].invert_yaxis()
    fig.tight_layout(); save(fig,"corrected_sensitivity")


def main() -> None:
    REPORT.mkdir(parents=True,exist_ok=True)
    chosen_font=configure_font()
    stats=json.loads((BASE/"evaluation/sensitivity_statistics_v1071.json").read_text(encoding="utf-8"))
    figure_method_overview(); figure_replication_effects(); figure_replication_paired(); figure_shield_ablation(); figure_corrected_sensitivity(stats)
    ARXIV_FIG.mkdir(parents=True,exist_ok=True); SUMMER_FIG.mkdir(parents=True,exist_ok=True)
    for path in PUBLICATION.glob("*.pdf"):
        shutil.copy2(path,ARXIV_FIG/path.name); shutil.copy2(path,SUMMER_FIG/path.name)
    for path in PUBLICATION.glob("*.png"):
        shutil.copy2(path,ARXIV_FIG/path.name); shutil.copy2(path,SUMMER_FIG/path.name)
    lines=["# v10.7.1 Condition-specific-baseline Sensitivity Correction","",
           "## Audit conclusion","",
           f"- Robustness classification: **{stats['robustness']['classification']}**.",
           f"- Perturbation conditions passed: **{stats['robustness']['perturbation_passes']}/4**.",
           "- Every condition used its own frozen C0 baseline and the unchanged 16.0705 mL margin.",
           "- C0 paired deltas are exactly zero by construction; C2/C4 used the frozen v10.6 checkpoint.",
           "- No model training, hyperparameter tuning, margin change, or v10.7 main-result recomputation occurred.",
           "- Tail risk uses the upper (worst) 10%, correcting the old lower-tail implementation.",
           f"- Figure font requested: Times New Roman; renderer used **{chosen_font}**.","",
           "## Condition results","",
           "| Condition | C4-C0 time, min (95% CI) | C4-C2 time, min (95% CI) | mean ΔB, mL | max ΔB, mL | Gate |",
           "|---|---:|---:|---:|---:|:---:|"]
    for condition in CONDITIONS:
        item=stats["conditions"][condition]; a=item["C4_minus_C0"]; b=item["C4_minus_C2"]
        lines.append(f"| {condition} | {a['mean_delta_T_min']:.3f} [{a['delta_T_95_ci'][0]:.3f}, {a['delta_T_95_ci'][1]:.3f}] | "
                     f"{b['mean_delta_T_min']:.3f} [{b['delta_T_95_ci'][0]:.3f}, {b['delta_T_95_ci'][1]:.3f}] | "
                     f"{a['mean_delta_B_ml']:.3f} | {a['max_delta_B_ml']:.3f} | {item['gate']['decision']} |")
    lines += ["","## Interpretation","",
              "This supplement replaces the invalid v10.7 sensitivity paragraph only. The independent Replication-256 result remains unchanged. Robustness claims must follow the classification above and must not reuse the original S1-S4 decisions.",""]
    (REPORT/"report_sensitivity_v1071.md").write_text("\n".join(lines),encoding="utf-8")
    (REPORT/"font_audit.json").write_text(json.dumps({"requested":"Times New Roman","used":chosen_font,
        "note":"Liberation Serif is the installed metrically compatible Times substitute" if chosen_font != "Times New Roman" else "exact font available"},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(REPORT/"report_sensitivity_v1071.md"),"font":chosen_font,
                      "figures":len(list(PUBLICATION.glob("*.pdf")))},ensure_ascii=False))


if __name__ == "__main__":
    main()
