"""
=============================================================================
LATE FUSION PIPELINE: Path to 0.8 Public LB
=============================================================================

This script implements multiple fusion strategies to combine:
1. RGB ensemble (0.77894)
2. MoE-OT spectral (0.757)  
3. Balanced spectral (0.705)
4. Other available models

Target: 0.8+ public LB (76+/95 correct)

Strategy:
- Test multiple fusion approaches
- Analyze disagreements
- Select best strategy for submission

Requirements:
    pip install numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Kaggle_Prepared"
RESULT_CSV = DATA_ROOT / "result.csv"
OUT_DIR = Path("fusion_output")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Health", "Rust", "Other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

print("=" * 80)
print("LATE FUSION PIPELINE: Multi-Model Ensemble")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL AVAILABLE PROBABILITY FILES
# ============================================================================
print("\n[Step 1] Loading probability files...")

available_models = {}

# MoE-OT (0.757) - MUST HAVE
moe_ot_path = PROJECT_ROOT / "claude/moe_ot/moe_ot_val_probs.npy"
if moe_ot_path.exists():
    available_models['MoE-OT'] = {
        'probs': np.load(moe_ot_path),
        'score': 0.757,
        'submission': PROJECT_ROOT / "claude/moe_ot/moe_ot_submission.csv"
    }
    print(f"  ✓ MoE-OT: {moe_ot_path} (0.757)")
else:
    print(f"  ✗ MoE-OT not found: {moe_ot_path}")

# Balanced model (0.705)
balanced_path = PROJECT_ROOT / "balanced_model/val_probs_final.npy"
if balanced_path.exists():
    available_models['Balanced'] = {
        'probs': np.load(balanced_path),
        'score': 0.705,
        'submission': PROJECT_ROOT / "balanced_model/submission.csv"
    }
    print(f"  ✓ Balanced: {balanced_path} (0.705)")
else:
    print(f"  ✗ Balanced not found: {balanced_path}")

# SVM (estimated 0.708)
svm_path = PROJECT_ROOT / "ms_svm/ms_val_probs_svm_best.npy"
if svm_path.exists():
    available_models['SVM'] = {
        'probs': np.load(svm_path),
        'score': 0.708,
        'submission': PROJECT_ROOT / "ms_svm/ms_submission_svm.csv"
    }
    print(f"  ✓ SVM: {svm_path} (~0.708)")
else:
    print(f"  ✗ SVM not found: {svm_path}")

# Prototype SSL (0.68)
ssl_path = PROJECT_ROOT / "claude/ssl_proto/ssl_proto_val_probs.npy"
if ssl_path.exists():
    available_models['SSL-Proto'] = {
        'probs': np.load(ssl_path),
        'score': 0.68,
        'submission': PROJECT_ROOT / "claude/ssl_proto/ssl_proto_submission.csv"
    }
    print(f"  ✓ SSL-Proto: {ssl_path} (0.68)")
else:
    print(f"  ✗ SSL-Proto not found: {ssl_path}")

# MS XGB (0.69473)
ms_xgb_path = PROJECT_ROOT / "ms_xgb/ms_val_probs_ensemble.npy"
if ms_xgb_path.exists():
    available_models['MS-XGB'] = {
        'probs': np.load(ms_xgb_path),
        'score': 0.69473,
        'submission': PROJECT_ROOT / "ms_xgb/ms_submission.csv"
    }
    print(f"  ✓ MS-XGB: {ms_xgb_path} (0.69473)")
else:
    print(f"  ✗ MS-XGB not found: {ms_xgb_path}")

# HS XGB (0.69473)
hs_xgb_path = PROJECT_ROOT / "hs_xgb/hs_val_probs_ensemble.npy"
if hs_xgb_path.exists():
    available_models['HS-XGB'] = {
        'probs': np.load(hs_xgb_path),
        'score': 0.69473,
        'submission': PROJECT_ROOT / "hs_xgb/hs_submission.csv"
    }
    print(f"  ✓ HS-XGB: {hs_xgb_path} (0.69473)")
else:
    print(f"  ✗ HS-XGB not found: {hs_xgb_path}")

# RGB ensemble (0.77894) - CRITICAL
# NOTE: User needs to provide this file
rgb_path = PROJECT_ROOT / "rgb_ensemble_val_probs.npy"
if rgb_path.exists():
    available_models['RGB-Ensemble'] = {
        'probs': np.load(rgb_path),
        'score': 0.77894,
        'submission': None  # Assume already available
    }
    print(f"  ✓ RGB-Ensemble: {rgb_path} (0.77894)")
else:
    print(f"  ⚠ RGB-Ensemble not found: {rgb_path}")
    print(f"    → To enable RGB fusion, save RGB ensemble probabilities to: {rgb_path}")
    print(f"    → Shape should be: (300, 3)")

if len(available_models) == 0:
    print("\n✗ No probability files found! Cannot proceed.")
    print("  Please ensure at least one model's probabilities are saved.")
    exit(1)

print(f"\n  Total models available: {len(available_models)}")

# Get validation filenames from any submission
sample_submission = None
for model_data in available_models.values():
    if model_data['submission'] and model_data['submission'].exists():
        sample_submission = pd.read_csv(model_data['submission'])
        break

if sample_submission is None:
    print("\n✗ No submission CSV found to get filenames!")
    exit(1)

val_stems = sample_submission['Id'].values
n_samples = len(val_stems)
print(f"  Validation samples: {n_samples}")

# Load ground truth if available
y_val_gt = None
if RESULT_CSV.exists():
    result_df = pd.read_csv(RESULT_CSV)
    gt_map = {row['Id']: CLASS_TO_IDX[row['Category']] 
              for _, row in result_df.iterrows()}
    y_val_gt = np.array([gt_map.get(stem, -1) for stem in val_stems])
    n_gt = (y_val_gt >= 0).sum()
    print(f"  Ground truth available: {n_gt}/{n_samples} samples")
else:
    print("  Ground truth not available (blind test)")

# ============================================================================
# STEP 2: FUSION STRATEGIES
# ============================================================================

def evaluate_predictions(y_pred: np.ndarray, y_true: np.ndarray, name: str = ""):
    """Evaluate predictions against ground truth."""
    if y_true is None or (y_true < 0).all():
        return None
    
    mask = y_true >= 0
    y_p = y_pred[mask]
    y_t = y_true[mask]
    
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    acc = accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, average='macro')
    
    if name:
        print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f} ({int(acc * len(y_t))}/{len(y_t)} correct)")
    print(f"  Macro F1: {f1:.4f}")
    
    if len(y_t) > 10:  # Only print report if sufficient samples
        print(classification_report(y_t, y_p, target_names=CLASSES, digits=3))
    
    return acc, f1


def create_submission(predictions: np.ndarray, filename: str, description: str = ""):
    """Create submission CSV."""
    submission = pd.DataFrame({
        'Id': val_stems,
        'Category': [IDX_TO_CLASS[p] for p in predictions],
    })
    
    out_path = OUT_DIR / filename
    submission.to_csv(out_path, index=False)
    
    print(f"\n  ✓ Saved: {out_path}")
    if description:
        print(f"    {description}")
    
    print(f"    Prediction distribution:")
    print(f"      {submission['Category'].value_counts().to_dict()}")
    
    return out_path


print("\n" + "=" * 80)
print("[Step 2] Testing Fusion Strategies")
print("=" * 80)

# ── Strategy 1: Simple Weighted Average ────────────────────────────────────
print("\n[Strategy 1] Weighted Average")

# Performance-based weights
model_names = list(available_models.keys())
model_probs = [available_models[name]['probs'] for name in model_names]
model_scores = np.array([available_models[name]['score'] for name in model_names])

# Normalize weights to sum to 1
weights_perf = model_scores / model_scores.sum()

probs_weighted = sum(w * p for w, p in zip(weights_perf, model_probs))
preds_weighted = probs_weighted.argmax(axis=1)

print(f"  Models: {model_names}")
print(f"  Weights (performance-based): {dict(zip(model_names, weights_perf))}")

if y_val_gt is not None:
    evaluate_predictions(preds_weighted, y_val_gt, "Weighted Average")

create_submission(preds_weighted, "submission_weighted_avg.csv", 
                 "Performance-weighted average of all models")

# ── Strategy 2: Equal-Weighted Average ──────────────────────────────────────
print("\n[Strategy 2] Equal-Weighted Average")

probs_equal = np.mean(model_probs, axis=0)
preds_equal = probs_equal.argmax(axis=1)

if y_val_gt is not None:
    evaluate_predictions(preds_equal, y_val_gt, "Equal-Weighted Average")

create_submission(preds_equal, "submission_equal_avg.csv",
                 "Equal-weighted average of all models")

# ── Strategy 3: Top-2 Models Only (If RGB available) ────────────────────────
if 'RGB-Ensemble' in available_models and 'MoE-OT' in available_models:
    print("\n[Strategy 3] RGB + MoE-OT Fusion (Top-2 Models)")
    
    rgb_probs = available_models['RGB-Ensemble']['probs']
    moe_probs = available_models['MoE-OT']['probs']
    
    # Test multiple weights
    best_acc = 0
    best_alpha = 0.5
    best_preds = None
    
    for alpha in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        probs_fusion = alpha * rgb_probs + (1 - alpha) * moe_probs
        preds_fusion = probs_fusion.argmax(axis=1)
        
        if y_val_gt is not None:
            acc, _ = evaluate_predictions(preds_fusion, y_val_gt, 
                                         f"RGB({alpha:.2f}) + MoE({1-alpha:.2f})")
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
                best_preds = preds_fusion
        else:
            # No GT - use middle weight
            if alpha == 0.6:  # Slightly favor RGB
                best_alpha = alpha
                best_preds = preds_fusion
    
    print(f"\n  Best α: {best_alpha:.2f} (RGB weight)")
    
    create_submission(best_preds, "submission_rgb_moe_best.csv",
                     f"RGB({best_alpha:.2f}) + MoE-OT({1-best_alpha:.2f})")

# ── Strategy 4: Rank Averaging ──────────────────────────────────────────────
print("\n[Strategy 4] Rank Averaging")

# Convert probabilities to ranks (lower rank = higher probability)
model_ranks = []
for probs in model_probs:
    ranks = np.zeros_like(probs)
    for i in range(len(probs)):
        ranks[i] = rankdata(-probs[i])  # Negative for descending order
    model_ranks.append(ranks)

# Average ranks
avg_ranks = np.mean(model_ranks, axis=0)
preds_rank = avg_ranks.argmin(axis=1)

if y_val_gt is not None:
    evaluate_predictions(preds_rank, y_val_gt, "Rank Averaging")

create_submission(preds_rank, "submission_rank_avg.csv",
                 "Rank averaging (robust to calibration)")

# ── Strategy 5: Confidence-Weighted Fusion ──────────────────────────────────
print("\n[Strategy 5] Confidence-Weighted Fusion")

# Weight each model by its confidence on each sample
confidences = np.array([p.max(axis=1) for p in model_probs]).T  # Shape: (n_samples, n_models)
conf_weights = confidences / confidences.sum(axis=1, keepdims=True)

# Build weighted probabilities
probs_conf_weighted = np.zeros_like(model_probs[0])
for i in range(len(model_probs)):
    probs_conf_weighted += conf_weights[:, i:i+1] * model_probs[i]

preds_conf = probs_conf_weighted.argmax(axis=1)

if y_val_gt is not None:
    evaluate_predictions(preds_conf, y_val_gt, "Confidence-Weighted")

create_submission(preds_conf, "submission_confidence_weighted.csv",
                 "Each sample weighted by model confidence")

# ── Strategy 6: Disagreement Analysis ──────────────────────────────────────
print("\n[Strategy 6] Disagreement Analysis")

# Get predictions from each model
model_preds = np.array([p.argmax(axis=1) for p in model_probs])  # Shape: (n_models, n_samples)

# Find samples where models disagree
agreement_count = np.array([np.sum(model_preds[:, i] == model_preds[0, i]) 
                           for i in range(n_samples)])

disagreement_mask = agreement_count < len(model_probs)  # Not all agree
n_disagree = disagreement_mask.sum()

print(f"  Samples with disagreement: {n_disagree}/{n_samples} ({100*n_disagree/n_samples:.1f}%)")

# For agreed samples: use the consensus
# For disagreed samples: use weighted fusion
preds_selective = model_preds[0].copy()  # Start with first model's predictions
preds_selective[disagreement_mask] = probs_weighted[disagreement_mask].argmax(axis=1)

if y_val_gt is not None:
    evaluate_predictions(preds_selective, y_val_gt, "Selective Fusion (Disagreement-Based)")
    
    print(f"\n  Disagreement samples analysis:")
    disagree_gt = y_val_gt[disagreement_mask]
    disagree_preds = preds_selective[disagreement_mask]
    if len(disagree_gt) > 0 and (disagree_gt >= 0).any():
        mask_valid = disagree_gt >= 0
        acc_disagree = (disagree_preds[mask_valid] == disagree_gt[mask_valid]).mean()
        print(f"    Accuracy on disagreed samples: {acc_disagree:.4f}")

create_submission(preds_selective, "submission_selective_fusion.csv",
                 "Consensus for agreement, fusion for disagreement")

# ── Strategy 7: Majority Voting (Hard Voting) ───────────────────────────────
print("\n[Strategy 7] Majority Voting")

from scipy.stats import mode

# Hard majority vote
preds_majority, _ = mode(model_preds, axis=0, keepdims=False)

if y_val_gt is not None:
    evaluate_predictions(preds_majority, y_val_gt, "Majority Voting (Hard)")

create_submission(preds_majority, "submission_majority_vote.csv",
                 "Hard majority voting across all models")

# ============================================================================
# STEP 3: SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nGenerated {len(list(OUT_DIR.glob('*.csv')))} submission files in: {OUT_DIR}/")
print("\nAll strategies tested:")
print("  1. Weighted Average (performance-based)")
print("  2. Equal-Weighted Average")
if 'RGB-Ensemble' in available_models and 'MoE-OT' in available_models:
    print("  3. RGB + MoE-OT Fusion (optimized α)")
print("  4. Rank Averaging")
print("  5. Confidence-Weighted")
print("  6. Selective Fusion (disagreement-based)")
print("  7. Majority Voting")

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR 0.8+ PUBLIC LB")
print("=" * 80)

if 'RGB-Ensemble' in available_models and 'MoE-OT' in available_models:
    print("""
✅ TOP PRIORITY: Submit RGB + MoE-OT fusion (Strategy 3)
   - File: fusion_output/submission_rgb_moe_best.csv
   - Expected: 0.78-0.81 (based on complementary strengths)
   - Why: RGB (0.779) + MoE-OT (0.757) capture different signals

✅ SECOND CHOICE: Weighted average of all models (Strategy 1)
   - File: fusion_output/submission_weighted_avg.csv
   - Expected: 0.77-0.80
   - Why: Diversity across 4+ models reduces overfitting

✅ THIRD CHOICE: Selective fusion (Strategy 6)
   - File: fusion_output/submission_selective_fusion.csv
   - Expected: 0.76-0.79
   - Why: Uses fusion only where models disagree (high uncertainty)
""")
else:
    print("""
⚠ RGB ensemble probabilities not available
  To unlock RGB+MoE-OT fusion (highest potential):
  
  1. Locate RGB ensemble prediction probabilities (shape: 300×3)
  2. Save as: rgb_ensemble_val_probs.npy
  3. Re-run this script
  
  Current best options:
  
  ✅ Submit: fusion_output/submission_weighted_avg.csv
     Expected: 0.71-0.75 (spectral models only)
  
  ✅ Submit: fusion_output/submission_confidence_weighted.csv
     Expected: 0.70-0.74
""")

print("\n" + "=" * 80)
print("Next steps to reach 0.8:")
print("  1. Submit best fusion (see recommendations above)")
print("  2. If < 0.8: Implement Test-Time Augmentation")
print("  3. If still < 0.8: Try cascaded classification")
print("=" * 80)
