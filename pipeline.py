"""
Random Data -> Classifier Training Pipeline
-------------------------------------------
Pattern Matching mini project.

Stages
  1. Generate controlled random datasets (class means, covariances, priors,
     separation, and the *distribution family* of each class).
  2. Estimate statistical parameters (MLE of mean, covariance, priors) and
     compare them with the true generating parameters.
  3. Train classifiers (Gaussian Bayes: QDA / LDA / Naive, plus k-NN and
     Logistic Regression as distribution-free / discriminative baselines).
  4. Analyse how violating distributional assumptions affects performance.

Run:  python pipeline.py            (full experiments, ~1-2 minutes)
      python pipeline.py --quick    (fewer repetitions)
Outputs go to ./results (CSV tables + PNG figures).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

OUT = "results"
FAMILIES = ["gaussian", "laplace", "uniform", "student_t", "lognormal", "bimodal"]
MODELS = ["QDA", "LDA", "GaussianNB", "kNN", "LogReg"]


# ----------------------------------------------------------------------------
# 1. Controlled random data generation
# ----------------------------------------------------------------------------
def standard_noise(rng, family, n, d):
    """Zero-mean, unit-variance, independent noise from the chosen family."""
    if family == "gaussian":
        return rng.standard_normal((n, d))
    if family == "laplace":
        return rng.laplace(0, 1 / np.sqrt(2), (n, d))
    if family == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), (n, d))
    if family == "student_t":                      # heavy tails, df = 3
        return rng.standard_t(3, (n, d)) / np.sqrt(3.0)
    if family == "lognormal":                      # strongly skewed
        s = 0.75
        mean = np.exp(s ** 2 / 2)
        var = (np.exp(s ** 2) - 1) * np.exp(s ** 2)
        return (rng.lognormal(0, s, (n, d)) - mean) / np.sqrt(var)
    if family == "bimodal":                        # two-bump mixture
        sign = rng.choice([-1.0, 1.0], (n, d))
        return 0.6 * rng.standard_normal((n, d)) + 0.8 * sign
    raise ValueError(family)


def make_params(sep=2.0, rho=0.0, unequal=False, priors=None):
    """True generating parameters for a 3-class, 2-D problem."""
    means = sep * np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    base = np.array([[1.0, rho], [rho, 1.0]])
    if unequal:   # different size, shape and orientation per class
        covs = [np.array([[0.35, 0.0], [0.0, 0.35]]),
                np.array([[2.5, 1.5], [1.5, 2.5]]),
                np.array([[1.0, -0.8], [-0.8, 1.6]])]
    else:
        covs = [base, base, base]
    priors = np.array([1 / 3, 1 / 3, 1 / 3]) if priors is None else np.asarray(priors)
    return means, covs, priors


def generate(rng, family, params, n):
    means, covs, priors = params
    y = rng.choice(len(priors), size=n, p=priors)
    X = np.empty((n, means.shape[1]))
    for c in range(len(priors)):
        idx = y == c
        L = np.linalg.cholesky(covs[c])
        X[idx] = means[c] + standard_noise(rng, family, idx.sum(), means.shape[1]) @ L.T
    return X, y


# ----------------------------------------------------------------------------
# 2 + 3. Parameter estimation and Gaussian Bayes classifiers (from scratch)
# ----------------------------------------------------------------------------
class GaussianBayes:
    """mode = 'qda' (class covariances), 'lda' (pooled), 'naive' (diagonal)."""

    def __init__(self, mode="qda", ridge=1e-6):
        self.mode, self.ridge = mode, ridge

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        d = X.shape[1]
        self.priors_ = np.array([(y == c).mean() for c in self.classes_])
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])      # MLE
        covs = [np.cov(X[y == c].T, bias=True) + self.ridge * np.eye(d)               # MLE
                for c in self.classes_]
        if self.mode == "lda":
            pooled = sum(p * C for p, C in zip(self.priors_, covs))
            covs = [pooled] * len(covs)
        elif self.mode == "naive":
            covs = [np.diag(np.diag(C)) for C in covs]
        self.covs_ = covs
        return self

    def log_posterior(self, X):
        d = X.shape[1]
        out = []
        for m, C, p in zip(self.means_, self.covs_, self.priors_):
            diff = X - m
            maha = np.einsum("ij,ij->i", diff @ np.linalg.inv(C), diff)
            _, logdet = np.linalg.slogdet(C)
            out.append(-0.5 * (d * np.log(2 * np.pi) + logdet + maha) + np.log(p))
        return np.column_stack(out)

    def predict(self, X):
        return self.classes_[np.argmax(self.log_posterior(X), axis=1)]


def build_model(name):
    return {"QDA": lambda: GaussianBayes("qda"),
            "LDA": lambda: GaussianBayes("lda"),
            "GaussianNB": lambda: GaussianBayes("naive"),
            "kNN": lambda: KNeighborsClassifier(n_neighbors=15),
            "LogReg": lambda: LogisticRegression(max_iter=1000)}[name]()


def estimation_error(Xtr, ytr, params):
    means, covs, priors = params
    g = GaussianBayes("qda").fit(Xtr, ytr)
    return {"mean_err": float(np.mean(np.linalg.norm(g.means_ - means, axis=1))),
            "cov_err": float(np.mean([np.linalg.norm(a - b) for a, b in zip(g.covs_, covs)])),
            "prior_err": float(np.abs(g.priors_ - priors).sum())}


# ----------------------------------------------------------------------------
# 4. Experiment helpers
# ----------------------------------------------------------------------------
def run_trial(seed, family="gaussian", sep=2.0, rho=0.0, unequal=False, priors=None,
              n_train=300, n_test=3000):
    rng = np.random.default_rng(seed)
    params = make_params(sep, rho, unequal, priors)
    Xtr, ytr = generate(rng, family, params, n_train)
    Xte, yte = generate(rng, family, params, n_test)
    res = {}
    for name in MODELS:
        pred = build_model(name).fit(Xtr, ytr).predict(Xte)
        res[name] = (pred == yte).mean()
        res[name + "_bal"] = balanced_accuracy_score(yte, pred)
    # Gaussian "oracle": true parameters plugged into the Gaussian Bayes rule
    oracle = GaussianBayes("qda")
    oracle.classes_ = np.arange(len(params[2]))
    oracle.means_, oracle.covs_, oracle.priors_ = params[0], params[1], params[2]
    res["Oracle"] = (oracle.predict(Xte) == yte).mean()
    return res


def repeat(reps, **kw):
    rows = [run_trial(1000 + r, **kw) for r in range(reps)]
    return pd.DataFrame(rows)


def summarize(df, cols):
    return {c: df[c].mean() for c in cols}, {c: df[c].std() for c in cols}


COLORS = {"QDA": "#d62728", "LDA": "#1f77b4", "GaussianNB": "#2ca02c",
          "kNN": "#9467bd", "LogReg": "#ff7f0e", "Oracle": "k"}


def line_plot(x, table, xlabel, title, fname, logx=False):
    plt.figure(figsize=(6.5, 4.2))
    for m in MODELS + ["Oracle"]:
        mu = np.array([t[0][m] for t in table])
        sd = np.array([t[1][m] for t in table])
        plt.plot(x, mu, marker="o", label=m, color=COLORS[m], ls="--" if m == "Oracle" else "-")
        plt.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.12)
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel); plt.ylabel("Test accuracy"); plt.title(title)
    plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=160); plt.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(quick=False):
    os.makedirs(OUT, exist_ok=True)
    reps = 8 if quick else 30
    cols = MODELS + ["Oracle"]

    # --- Figure 1: what the random data looks like for every family ----------
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for ax, fam in zip(axs.ravel(), FAMILIES):
        X, y = generate(np.random.default_rng(7), fam, make_params(2.2), 600)
        for c, col in zip(range(3), ["#d62728", "#1f77b4", "#2ca02c"]):
            ax.scatter(*X[y == c].T, s=7, alpha=0.6, c=col)
        ax.set_title(fam); ax.set_xlim(-5, 7); ax.set_ylim(-5, 6)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_datasets.png", dpi=160); plt.close()

    # --- Exp A: parameter estimation error vs training size ------------------
    sizes = [30, 60, 120, 250, 500, 1000, 2000, 5000]
    rows = []
    for n in sizes:
        for fam in ["gaussian", "student_t"]:
            errs = []
            for r in range(reps):
                rng = np.random.default_rng(500 + r)
                p = make_params(2.0, 0.3)
                Xtr, ytr = generate(rng, fam, p, n)
                errs.append(estimation_error(Xtr, ytr, p))
            e = pd.DataFrame(errs).mean()
            rows.append({"n": n, "family": fam, **e.to_dict()})
    dfA = pd.DataFrame(rows); dfA.to_csv(f"{OUT}/expA_estimation.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for fam, ls in [("gaussian", "-"), ("student_t", "--")]:
        s = dfA[dfA.family == fam]
        axs[0].loglog(s.n, s.mean_err, ls, marker="o", label=f"mean ({fam})")
        axs[1].loglog(s.n, s.cov_err, ls, marker="o", label=f"covariance ({fam})")
    n_ = np.array(sizes)
    axs[0].loglog(n_, 1.2 / np.sqrt(n_) * 3 ** 0.5, ":", c="gray", label=r"$\propto 1/\sqrt{n}$")
    for a, t in zip(axs, ["Mean estimation error", "Covariance estimation error"]):
        a.set_xlabel("Training samples"); a.set_ylabel("Error (L2 / Frobenius)")
        a.set_title(t); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_estimation.png", dpi=160); plt.close()

    # --- Exp B: accuracy vs class separation (Gaussian) ----------------------
    seps = [0.5, 1, 1.5, 2, 2.5, 3, 4]
    tabB = [summarize(repeat(reps, sep=s), cols) for s in seps]
    pd.DataFrame([t[0] for t in tabB], index=seps).round(4).to_csv(f"{OUT}/expB_separation.csv")
    line_plot(seps, tabB, "Class separation (distance between means)",
              "Accuracy vs separation (Gaussian data)", "fig3_separation.png")

    # --- Exp C: accuracy under each distribution family ----------------------
    tabC, accC, sdC = {}, [], []
    for fam in FAMILIES:
        df = repeat(reps, family=fam, sep=2.0, rho=0.3)
        mu, sd = summarize(df, cols)
        tabC[fam] = mu; accC.append(mu); sdC.append(sd)
    dfC = pd.DataFrame(tabC).T.round(4); dfC.to_csv(f"{OUT}/expC_families.csv")
    dfCs = pd.DataFrame(sdC, index=FAMILIES).round(4); dfCs.to_csv(f"{OUT}/expC_families_std.csv")
    plt.figure(figsize=(9, 4.2))
    w = 0.14
    for i, m in enumerate(MODELS):
        plt.bar(np.arange(len(FAMILIES)) + i * w, dfC[m], w, yerr=dfCs[m], label=m,
                color=COLORS[m], capsize=2)
    plt.xticks(np.arange(len(FAMILIES)) + 2 * w, FAMILIES)
    plt.ylim(0.5, 0.95); plt.ylabel("Test accuracy"); plt.grid(axis="y", alpha=0.3)
    plt.title("Effect of the true data distribution (n_train=300, sep=2)")
    plt.legend(ncol=5, fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_families.png", dpi=160); plt.close()

    # --- Exp D: equal vs unequal covariances --------------------------------
    rowsD = {}
    for label, uneq in [("equal covariances", False), ("unequal covariances", True)]:
        rowsD[label] = summarize(repeat(reps, unequal=uneq, sep=2.0), cols)[0]
    dfD = pd.DataFrame(rowsD).T.round(4); dfD.to_csv(f"{OUT}/expD_covariance.csv")

    # --- Exp E: feature correlation (breaks Naive Bayes independence) --------
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    tabE = [summarize(repeat(reps, rho=r, sep=2.0), cols) for r in rhos]
    pd.DataFrame([t[0] for t in tabE], index=rhos).round(4).to_csv(f"{OUT}/expE_correlation.csv")
    line_plot(rhos, tabE, "Feature correlation ρ",
              "Accuracy vs feature correlation (Gaussian data)", "fig5_correlation.png")

    # --- Exp F: training-set size (data hungry vs parametric) ----------------
    ns = [15, 30, 60, 120, 300, 1000]
    tabF = [summarize(repeat(reps, n_train=n, family="gaussian", sep=2.0, unequal=True), cols)
            for n in ns]
    pd.DataFrame([t[0] for t in tabF], index=ns).round(4).to_csv(f"{OUT}/expF_trainsize.csv")
    line_plot(ns, tabF, "Training samples", "Accuracy vs training-set size (unequal covariances)",
              "fig6_trainsize.png", logx=True)

    # --- Exp G: class imbalance ---------------------------------------------
    rowsG = {}
    for label, pr in [("balanced 1/3 each", None), ("mild 50/30/20", [.5, .3, .2]),
                      ("severe 80/15/5", [.8, .15, .05])]:
        df = repeat(reps, priors=pr, sep=2.0)
        rowsG[label + " (acc)"] = {m: df[m].mean() for m in MODELS}
        rowsG[label + " (balanced acc)"] = {m: df[m + "_bal"].mean() for m in MODELS}
    dfG = pd.DataFrame(rowsG).T.round(4); dfG.to_csv(f"{OUT}/expG_imbalance.csv")

    # --- Exp H: how Gaussian is each family? (normality tests) --------------
    rowsH = []
    for fam in FAMILIES:
        X, y = generate(np.random.default_rng(11), fam, make_params(2.0), 3000)
        Z = X[y == 0] - X[y == 0].mean(axis=0)
        rowsH.append({"family": fam,
                      "skewness": float(np.mean(stats.skew(Z))),
                      "excess_kurtosis": float(np.mean(stats.kurtosis(Z))),
                      "D'Agostino_p": float(stats.normaltest(Z[:, 0]).pvalue)})
    pd.DataFrame(rowsH).round(4).to_csv(f"{OUT}/expH_normality.csv", index=False)

    # --- Decision boundaries + confusion matrix ------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
    xx, yy = np.meshgrid(np.linspace(-5, 7, 250), np.linspace(-5, 6, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    for r, fam in enumerate(["gaussian", "student_t"]):
        rng = np.random.default_rng(3)
        p = make_params(2.0, 0.3)
        Xtr, ytr = generate(rng, fam, p, 300)
        for c, name in enumerate(["QDA", "GaussianNB", "kNN"]):
            m = build_model(name).fit(Xtr, ytr)
            Z = m.predict(grid).reshape(xx.shape)
            ax = axs[r, c]
            ax.contourf(xx, yy, Z, alpha=0.25, levels=[-.5, .5, 1.5, 2.5],
                        colors=["#d62728", "#1f77b4", "#2ca02c"])
            ax.scatter(*Xtr.T, c=ytr, cmap="brg_r", s=6)
            ax.set_title(f"{name} - {fam}")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig7_boundaries.png", dpi=160); plt.close()

    rng = np.random.default_rng(5)
    p = make_params(2.0, 0.3, unequal=True)
    Xtr, ytr = generate(rng, "gaussian", p, 500); Xte, yte = generate(rng, "gaussian", p, 3000)
    cm = confusion_matrix(yte, GaussianBayes("qda").fit(Xtr, ytr).predict(Xte))
    plt.figure(figsize=(4.2, 3.8)); plt.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="w" if cm[i, j] > cm.max() / 2 else "k")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("QDA confusion matrix")
    plt.xticks(range(3)); plt.yticks(range(3)); plt.tight_layout()
    plt.savefig(f"{OUT}/fig8_confusion.png", dpi=160); plt.close()

    # --- console summary -----------------------------------------------------
    pd.set_option("display.width", 140)
    print("\n=== Exp C: accuracy by distribution family ===\n", dfC)
    print("\n=== Exp D: equal vs unequal covariances ===\n", dfD)
    print("\n=== Exp G: class imbalance ===\n", dfG)
    print("\nAll tables and figures saved in ./results")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
