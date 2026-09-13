#!/usr/bin/env python3
"""Task A: MIRACL-only JA--HI controlled-probe Spearman rho."""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260506
N_BOOT = 10_000
SCRIPT_DIR = Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent          # analysis/
REPO_ROOT = SCRIPT_DIR.parent.parent      # repo root
OUT_DIR = SCRIPT_DIR / "task_a_miracl"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(OUT_DIR / "task_a_miracl.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

RESULTS_DIR = REPO_ROOT / "mteb-language-gap" / "results"

EXCLUDE_MODELS = {"intfloat/e5-small-v2", "intfloat/e5-large-v2"}

# Baselines from §4.3 (JA--HI)
FULL_AVG_RHO = 0.829
BELEBELE_RHO = 0.789


def load_task_scores(lang_dir: str, task: str) -> dict:
    """Return {model_id: ndcg_at_10} for all non-excluded models that have both metadata and task file.

    When multiple dirs share the same model_id (e.g. a debug re-run), keep the one with the
    highest total_wall_sec (the most complete production run).
    """
    candidates: dict[str, tuple[float, float]] = {}  # model_id -> (wall_sec, ndcg)
    lb_dir = RESULTS_DIR / lang_dir
    if not lb_dir.exists():
        log.warning(f"Dir not found: {lb_dir}")
        return {}
    for model_dir in sorted(lb_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        meta_f = model_dir / "_metadata.json"
        task_f = model_dir / f"{task}.json"
        if not meta_f.exists() or not task_f.exists():
            continue
        with open(meta_f, encoding="utf-8") as f:
            meta = json.load(f)
        model_id = meta.get("model_id", "")
        if model_id in EXCLUDE_MODELS:
            log.info(f"Excluding {model_id}")
            continue
        wall_sec = float(meta.get("total_wall_sec", 0.0))
        with open(task_f, encoding="utf-8") as f:
            ndcg = json.load(f)["scores"]["test"][0]["ndcg_at_10"]
        if model_id not in candidates or wall_sec > candidates[model_id][0]:
            candidates[model_id] = (wall_sec, ndcg)
            log.info(f"  {lang_dir}/{model_dir.name}: {model_id} wall={wall_sec:.0f}s -> {ndcg:.4f}")
        else:
            log.info(f"  {lang_dir}/{model_dir.name}: {model_id} wall={wall_sec:.0f}s SKIPPED (duplicate, shorter run)")
    return {mid: v[1] for mid, v in candidates.items()}


def bootstrap_spearman_ci(x, y, rng, n_boot=N_BOOT):
    n = len(x)
    mat = np.column_stack([x, y])
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = mat[idx]
        rho = stats.spearmanr(s[:, 0], s[:, 1]).statistic
        boot_rhos.append(rho)
    return np.percentile(boot_rhos, [2.5, 97.5])


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    log.info("Task A: MIRACL-only JA--HI controlled-probe")
    log.info(f"RESULTS_DIR: {RESULTS_DIR}")

    log.info("Loading MIRACLRetrievalHardNegatives for jpn_lb ...")
    ja_miracl = load_task_scores("jpn_lb", "MIRACLRetrievalHardNegatives")
    log.info("Loading MIRACLRetrievalHardNegatives for hin_lb ...")
    hi_miracl = load_task_scores("hin_lb", "MIRACLRetrievalHardNegatives")

    common = sorted(set(ja_miracl) & set(hi_miracl))
    n = len(common)
    log.info(f"Models with both JA+HI MIRACL (after roster filter): {n}")
    for m in common:
        log.info(f"  {m}: JA={ja_miracl[m]:.4f}, HI={hi_miracl[m]:.4f}")

    if n < 3:
        log.error("Too few models for correlation; aborting.")
        return

    descriptive_only = n < 8
    if descriptive_only:
        log.warning(f"n={n} < 8 -- reporting as descriptive only")

    ja_vec = np.array([ja_miracl[m] for m in common])
    hi_vec = np.array([hi_miracl[m] for m in common])

    rho_miracl, pval_miracl = stats.spearmanr(ja_vec, hi_vec)
    ci_lo, ci_hi = bootstrap_spearman_ci(ja_vec, hi_vec, rng)
    log.info(f"MIRACL-only rho={rho_miracl:.4f} p={pval_miracl:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}]")

    log.info("Loading BelebeleRetrieval for jpn_lb and hin_lb (own-run baseline) ...")
    ja_bele = load_task_scores("jpn_lb", "BelebeleRetrieval")
    hi_bele = load_task_scores("hin_lb", "BelebeleRetrieval")
    bele_common = sorted(set(ja_bele) & set(hi_bele))
    n_bele = len(bele_common)
    log.info(f"Belebele common models (own-run): {n_bele}")

    rows = []

    rows.append(dict(
        comparison="full_avg_s43",
        n_models="N/A",
        rho=FULL_AVG_RHO,
        ci_lo="N/A",
        ci_hi="N/A",
        p_value="N/A",
        note="published_baseline",
    ))

    if n_bele >= 3:
        ja_b = np.array([ja_bele[m] for m in bele_common])
        hi_b = np.array([hi_bele[m] for m in bele_common])
        rho_b, pval_b = stats.spearmanr(ja_b, hi_b)
        ci_b_lo, ci_b_hi = bootstrap_spearman_ci(ja_b, hi_b, rng)
        log.info(f"Belebele-only rho={rho_b:.4f} p={pval_b:.4f} CI=[{ci_b_lo:.4f},{ci_b_hi:.4f}]")
        rows.append(dict(
            comparison="belebele_only_ownrun",
            n_models=n_bele,
            rho=round(rho_b, 4),
            ci_lo=round(float(ci_b_lo), 4),
            ci_hi=round(float(ci_b_hi), 4),
            p_value=round(float(pval_b), 4),
            note="own_run_computed",
        ))
    else:
        log.warning(f"Belebele n_bele={n_bele} < 3; using published baseline value")
        rows.append(dict(
            comparison="belebele_only_s43",
            n_models="N/A",
            rho=BELEBELE_RHO,
            ci_lo="N/A",
            ci_hi="N/A",
            p_value="N/A",
            note="published_baseline",
        ))

    rows.append(dict(
        comparison="miracl_only_ownrun",
        n_models=n,
        rho=round(rho_miracl, 4),
        ci_lo=round(float(ci_lo), 4),
        ci_hi=round(float(ci_hi), 4),
        p_value=round(float(pval_miracl), 4),
        note="own_run_computed_new",
    ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "miracl_jahi.csv", index=False)
    log.info(f"Saved miracl_jahi.csv ({len(df)} rows)")

    rho_str = f"{rho_miracl:.3f}"
    if rho_miracl >= 0.75:
        verdict = "confirming"
    elif rho_miracl >= 0.60:
        verdict = "partially confirming"
    else:
        verdict = "not clearly confirming"

    desc_note = " Results are descriptive only (n < 8)." if descriptive_only else ""

    md = (
        "## MIRACL-Only Controlled Probe\n\n"
        f"MIRACLRetrievalHardNegatives-only (n={n}). "
        f"Japanese--Hindi Spearman rho moves from {FULL_AVG_RHO} (full per-language average) "
        f"to {rho_str} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}], p={pval_miracl:.3f}) "
        "under this third task-fixed control, "
        f"{verdict} the Belebele and WikipediaRetrievalMultilingual pattern: "
        "holding the task constant collapses cross-language disagreement on per-model "
        "gaps regardless of which task is held constant."
        f"{desc_note}\n"
    )
    (OUT_DIR / "miracl_jahi.md").write_text(md, encoding="utf-8")
    log.info(f"Saved miracl_jahi.md. Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
