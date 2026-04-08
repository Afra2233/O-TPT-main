import matplotlib
matplotlib.use("Agg")

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt



def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    return result


def compute_ece(confidence, correct, num_bins=15):
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(num_bins):
        left = bins[i]
        right = bins[i + 1]

        if i == 0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence > left) & (confidence <= right)

        count = mask.sum()
        if count > 0:
            acc = correct[mask].mean()
            conf = confidence[mask].mean()
        else:
            acc = 0.0
            conf = 0.0

        ece += (count / len(confidence)) * abs(acc - conf)
        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(count)

    return ece, np.array(bin_accs), np.array(bin_confs), np.array(bin_counts), bins


def plot_confidence_histogram(
    confidence_a,
    confidence_b,
    label_a,
    label_b,
    title,
    save_path,
    bins=20,
    density=True
):
    plt.figure(figsize=(7, 5))
    plt.hist(confidence_a, bins=bins, alpha=0.5, label=label_a, density=density)
    plt.hist(confidence_b, bins=bins, alpha=0.5, label=label_b, density=density)
    plt.xlabel("Prediction confidence")
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_single_confidence_histogram(
    confidence,
    correct,
    title,
    save_path,
    bins=20,
    density=True
):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct).astype(bool)

    conf_correct = confidence[correct]
    conf_incorrect = confidence[~correct]

    plt.figure(figsize=(7, 5))
    if len(conf_correct) > 0:
        plt.hist(conf_correct, bins=bins, alpha=0.5, label="Correct", density=density)
    if len(conf_incorrect) > 0:
        plt.hist(conf_incorrect, bins=bins, alpha=0.5, label="Incorrect", density=density)

    plt.xlabel("Prediction confidence")
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_reliability_diagram(confidence, correct, title, save_path, num_bins=15):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct).astype(np.float32)

    ece, bin_accs, bin_confs, bin_counts, bins = compute_ece(confidence, correct, num_bins=num_bins)
    centers = (bins[:-1] + bins[1:]) / 2.0
    widths = bins[1:] - bins[:-1]

    plt.figure(figsize=(6, 6))
    plt.bar(centers, bin_accs, width=widths, alpha=0.7, edgecolor='black', label='Accuracy')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label='Perfect calibration')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title}\nECE={ece * 100:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_reliability_comparison(
    conf_a, corr_a, label_a,
    conf_b, corr_b, label_b,
    title, save_path, num_bins=15
):
    conf_a = np.asarray(conf_a)
    corr_a = np.asarray(corr_a).astype(np.float32)
    conf_b = np.asarray(conf_b)
    corr_b = np.asarray(corr_b).astype(np.float32)

    ece_a, bin_accs_a, _, _, bins = compute_ece(conf_a, corr_a, num_bins=num_bins)
    ece_b, bin_accs_b, _, _, _ = compute_ece(conf_b, corr_b, num_bins=num_bins)

    centers = (bins[:-1] + bins[1:]) / 2.0
    width = (bins[1] - bins[0]) * 0.4

    plt.figure(figsize=(7, 6))
    plt.bar(centers - width / 2, bin_accs_a, width=width, alpha=0.7,
            label=f"{label_a} (ECE={ece_a * 100:.2f})")
    plt.bar(centers + width / 2, bin_accs_b, width=width, alpha=0.7,
            label=f"{label_b} (ECE={ece_b * 100:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label='Perfect calibration')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence bin")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarize(name, confidence, correct):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct).astype(np.float32)

    acc = correct.mean() * 100
    ece, _, _, _, _ = compute_ece(confidence, correct, num_bins=15)

    print(f"===== {name} =====")
    print(f"Accuracy: {acc:.2f}")
    print(f"ECE: {ece * 100:.2f}")
    print(f"Mean confidence: {confidence.mean() * 100:.2f}")
    if (~correct.astype(bool)).sum() > 0:
        print(f"Mean confidence on incorrect predictions: {confidence[~correct.astype(bool)].mean() * 100:.2f}")
    if correct.astype(bool).sum() > 0:
        print(f"Mean confidence on correct predictions: {confidence[correct.astype(bool)].mean() * 100:.2f}")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Plot confidence histograms and reliability diagrams from saved npz files")
    parser.add_argument("--baseline_npz", type=str, required=True, help="path to baseline npz")
    parser.add_argument("--otpt_npz", type=str, required=True, help="path to O-TPT npz")
    parser.add_argument("--output_dir", type=str, required=True, help="directory to save plots")
    parser.add_argument("--mode", type=str, default="robust", choices=["clean", "robust"], help="which split to analyze")
    parser.add_argument("--num_bins", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline = load_npz(args.baseline_npz)
    otpt = load_npz(args.otpt_npz)

    if args.mode == "robust":
        baseline_conf = baseline["robust_confidence"]
        baseline_corr = baseline["robust_correct"]
        otpt_conf = otpt["robust_confidence"]
        otpt_corr = otpt["robust_correct"]
        prefix = "robust"
    else:
        baseline_conf = baseline["clean_confidence"]
        baseline_corr = baseline["clean_correct"]
        otpt_conf = otpt["clean_confidence"]
        otpt_corr = otpt["clean_correct"]
        prefix = "clean"

    summarize(f"Baseline {prefix}", baseline_conf, baseline_corr)
    summarize(f"O-TPT {prefix}", otpt_conf, otpt_corr)

    # 1) overall confidence histogram comparison
    plot_confidence_histogram(
        baseline_conf,
        otpt_conf,
        "Baseline",
        "O-TPT",
        title=f"{prefix.capitalize()} confidence histogram",
        save_path=os.path.join(args.output_dir, f"{prefix}_confidence_hist_baseline_vs_otpt.png"),
        bins=20,
        density=True
    )

    # 2) incorrect-only histogram comparison
    baseline_incorrect_conf = baseline_conf[np.asarray(baseline_corr).astype(bool) == False]
    otpt_incorrect_conf = otpt_conf[np.asarray(otpt_corr).astype(bool) == False]

    plot_confidence_histogram(
        baseline_incorrect_conf,
        otpt_incorrect_conf,
        "Baseline incorrect",
        "O-TPT incorrect",
        title=f"{prefix.capitalize()} incorrect-only confidence histogram",
        save_path=os.path.join(args.output_dir, f"{prefix}_incorrect_confidence_hist_baseline_vs_otpt.png"),
        bins=20,
        density=True
    )

    # 3) per-method correct vs incorrect histogram
    plot_single_confidence_histogram(
        baseline_conf,
        baseline_corr,
        title=f"Baseline {prefix} confidence: correct vs incorrect",
        save_path=os.path.join(args.output_dir, f"{prefix}_baseline_correct_vs_incorrect_hist.png"),
        bins=20,
        density=True
    )

    plot_single_confidence_histogram(
        otpt_conf,
        otpt_corr,
        title=f"O-TPT {prefix} confidence: correct vs incorrect",
        save_path=os.path.join(args.output_dir, f"{prefix}_otpt_correct_vs_incorrect_hist.png"),
        bins=20,
        density=True
    )

    # 4) reliability diagram per method
    plot_reliability_diagram(
        baseline_conf,
        baseline_corr,
        title=f"Baseline {prefix} reliability diagram",
        save_path=os.path.join(args.output_dir, f"{prefix}_baseline_reliability.png"),
        num_bins=args.num_bins
    )

    plot_reliability_diagram(
        otpt_conf,
        otpt_corr,
        title=f"O-TPT {prefix} reliability diagram",
        save_path=os.path.join(args.output_dir, f"{prefix}_otpt_reliability.png"),
        num_bins=args.num_bins
    )

    # 5) side-by-side reliability comparison
    plot_reliability_comparison(
        baseline_conf, baseline_corr, "Baseline",
        otpt_conf, otpt_corr, "O-TPT",
        title=f"{prefix.capitalize()} reliability comparison",
        save_path=os.path.join(args.output_dir, f"{prefix}_reliability_comparison.png"),
        num_bins=args.num_bins
    )

    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()