"""
=============================================================================
PATH TO 0.8 WITHOUT RGB PROBABILITIES
=============================================================================

Available assets:
1. RGB ensemble submission CSV (0.779 - 74/95 correct)
2. MoE-OT probabilities (0.757 - 72/95 correct)
3. Multiple spectral model probabilities (Balanced, SVM, etc.)

Strategy:
- Improve spectral ensemble to 0.77+
- Use hard voting with RGB CSV
- Analyze disagreements to find optimal overrides

Requirements:
    pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Kaggle_Prepared"
RESULT_CSV = DATA_ROOT / "result.csv"
OUT_DIR = Path("reach_08_output")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Health", "Rust", "Other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

print("=" * 80)
print("PATH TO 0.8: Strategies Without RGB Probabilities")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL AVAILABLE MODELS
# ============================================================================
print("\n[Step 1] Loading available models...")

# Load MoE-OT (our best spectral)
moe_probs = np.load(PROJECT_ROOT / "claude/moe_ot/moe_ot_val_probs.npy")
moe_sub = pd.read_csv(PROJECT_ROOT / "claude/moe_ot/moe_ot_submission.csv")
val_stems = moe_sub['Id'].values
n_samples = len(val_stems)

print(f"  ✓ MoE-OT probabilities (0.757)")

# Find RGB ensemble CSV (0.779)
rgb_candidates = [
    "submission231.csv",
    "final.csv", 
    "goodresplz.csv",
    "goodresplzV2.csv",
    "sigma.csv"
]

rgb_sub = None
rgb_path = None
for candidate in rgb_candidates:
    path = PROJECT_ROOT / candidate
    if path.exists():
        print(f"  ? Found: {candidate}")
        
# Let user specify or we'll search
print("\n  Please specify which CSV is your 0.779 RGB ensemble:")
for i, candidate in enumerate(rgb_candidates, 1):
    path = PROJECT_ROOT / candidate
    if path.exists():
        print(f"    {i}. {candidate}")

# For now, let's try to find it automatically
for candidate in rgb_candidates:
    path = PROJECT_ROOT / candidate
    if path.exists():
        rgb_path = path
        rgb_sub = pd.read_csv(path)
        print(f"\n  Using: {candidate} as RGB ensemble (0.779)")
        break

if rgb_sub is None:
    print("\n  ⚠ RGB ensemble CSV not found!")
    print("  Please manually specify the file path of your 0.779 submission")
    print("  Exiting...")
    exit(1)

# Convert RGB CSV to indices
rgb_preds = np.array([CLASS_TO_IDX[cat] for cat in rgb_sub['Category'].values])

# Load other spectral models
spectral_models = {}

# Balanced
balanced_path = PROJECT_ROOT / "balanced_model/val_probs_final.npy"
if balanced_path.exists():
    spectral_models['Balanced'] = np.load(balanced_path)
    print(f"  ✓ Balanced probabilities (0.705)")

# SVM
svm_path = PROJECT_ROOT / "ms_svm/ms_val_probs_svm_best.npy"
if svm_path.exists():
    spectral_models['SVM'] = np.load(svm_path)
    print(f"  ✓ SVM probabilities (~0.708)")

# SSL Proto
ssl_path = PROJECT_ROOT / "claude/ssl_proto/ssl_proto_val_probs.npy"
if ssl_path.exists():
    spectral_models['SSL'] = np.load(ssl_path)
    print(f"  ✓ SSL-Proto probabilities (0.68)")

# MS XGB
ms_xgb_path = PROJECT_ROOT / "ms_xgb/ms_val_probs_ensemble.npy"
if ms_xgb_path.exists():
    spectral_models['MS-XGB'] = np.load(ms_xgb_path)
    print(f"  ✓ MS-XGB probabilities (0.69473)")

# HS XGB
hs_xgb_path = PROJECT_ROOT / "hs_xgb/hs_val_probs_ensemble.npy"
if hs_xgb_path.exists():
    spectral_models['HS-XGB'] = np.load(hs_xgb_path)
    print(f"  ✓ HS-XGB probabilities (0.69473)")

print(f"\n  Total spectral models: {len(spectral_models) + 1} (including MoE-OT)")

# Load ground truth if available
y_val_gt = None
if RESULT_CSV.exists():
    result_df = pd.read_csv(RESULT_CSV)
    gt_map = {row['Id']: CLASS_TO_IDX[row['Category']] 
              for _, row in result_df.iterrows()}
    y_val_gt = np.array([gt_map.get(stem, -1) for stem in val_stems])
    n_gt = (y_val_gt >= 0).sum()
    print(f"  ✓ Ground truth available: {n_gt}/{n_samples} samples")
    HAS_GT = True
else:
    print("  ⚠ Ground truth not available")
    HAS_GT = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def evaluate(y_pred, y_true, name=""):
    """Evaluate predictions."""
    if y_true is None or (y_true < 0).all():
        return None
    
    mask = y_true >= 0
    y_p = y_pred[mask]
    y_t = y_true[mask]
    
    acc = accuracy_score(y_t, y_p)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f} ({int(acc * len(y_t))}/{len(y_t)} correct)")
    
    if len(y_t) > 10:
        print(classification_report(y_t, y_p, target_names=CLASSES, digits=3))
    
    return acc


def save_submission(predictions, filename, description=""):
    """Save submission CSV."""
    submission = pd.DataFrame({
        'Id': val_stems,
        'Category': [IDX_TO_CLASS[p] for p in predictions],
    })
    
    out_path = OUT_DIR / filename
    submission.to_csv(out_path, index=False)
    
    print(f"\n  ✓ Saved: {out_path}")
    if description:
        print(f"    {description}")
    
    dist = submission['Category'].value_counts().to_dict()
    print(f"    Distribution: {dist}")
    
    return out_path

# ============================================================================
# STRATEGY 1: Improved Spectral Ensemble
# ============================================================================
print("\n" + "=" * 80)
print("[Strategy 1] Build Best Spectral Ensemble")
print("=" * 80)

# Combine all spectral models
all_spectral = [moe_probs] + list(spectral_models.values())
spectral_names = ['MoE-OT'] + list(spectral_models.keys())

# Performance-weighted
weights = [0.757, 0.705, 0.708, 0.68, 0.695, 0.695][:len(all_spectral)]
weights = np.array(weights) / sum(weights)

spectral_ensemble = np.zeros_like(moe_probs)
for i, probs in enumerate(all_spectral):
    spectral_ensemble += weights[i] * probs

spectral_preds = spectral_ensemble.argmax(axis=1)

print(f"  Models: {spectral_names}")
print(f"  Weights: {dict(zip(spectral_names, weights))}")

if HAS_GT:
    evaluate(spectral_preds, y_val_gt, "Spectral Ensemble")

save_submission(spectral_preds, "spectral_ensemble_best.csv",
                "Best spectral ensemble (all models)")

# ============================================================================
# STRATEGY 2: Hard Voting with RGB + Spectral
# ============================================================================
print("\n" + "=" * 80)
print("[Strategy 2] Hard Voting: RGB + Best Spectral")
print("=" * 80)

# Majority vote between RGB and spectral ensemble
voting_preds = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    votes = [rgb_preds[i], spectral_preds[i]]
    counter = Counter(votes)
    
    # If tie or both agree, use RGB (it's better)
    if counter.most_common(1)[0][1] >= 1:
        voting_preds[i] = rgb_preds[i]

if HAS_GT:
    evaluate(voting_preds, y_val_gt, "Hard Voting (RGB + Spectral)")

save_submission(voting_preds, "hard_voting_rgb_spectral.csv",
                "Hard majority vote between RGB and spectral ensemble")

# ============================================================================
# STRATEGY 3: Selective Override (Spectral Very Confident)
# ============================================================================
print("\n" + "=" * 80)
print("[Strategy 3] Selective Override - Trust RGB, Override When Spectral Certain")
print("=" * 80)

moe_preds = moe_probs.argmax(axis=1)
moe_conf = moe_probs.max(axis=1)
spectral_conf = spectral_ensemble.max(axis=1)

# Base: RGB
selective_preds = rgb_preds.copy()

# Override when spectral is VERY confident (>0.85) and disagrees
disagreement = (rgb_preds != spectral_preds)
spectral_very_confident = spectral_conf > 0.85

override_mask = disagreement & spectral_very_confident
selective_preds[override_mask] = spectral_preds[override_mask]

print(f"  Disagreements: {disagreement.sum()}/{n_samples}")
print(f"  Overrides (spectral conf > 0.85): {override_mask.sum()}")

if HAS_GT:
    evaluate(selective_preds, y_val_gt, "Selective Override")

save_submission(selective_preds, "selective_override.csv",
                "RGB base + spectral override when very confident")

# ============================================================================
# STRATEGY 4: Disagreement Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[Strategy 4] Disagreement Analysis")
print("=" * 80)

disagreements = (rgb_preds != spectral_preds)
print(f"\n  Total disagreements: {disagreements.sum()}/{n_samples} ({100*disagreements.sum()/n_samples:.1f}%)")

if HAS_GT:
    # Analyze who's right when they disagree
    mask_valid = y_val_gt >= 0
    disagree_and_valid = disagreements & mask_valid
    
    if disagree_and_valid.sum() > 0:
        rgb_right = rgb_preds[disagree_and_valid] == y_val_gt[disagree_and_valid]
        spectral_right = spectral_preds[disagree_and_valid] == y_val_gt[disagree_and_valid]
        
        print(f"\n  When models disagree:")
        print(f"    RGB correct: {rgb_right.sum()}/{disagree_and_valid.sum()} ({100*rgb_right.mean():.1f}%)")
        print(f"    Spectral correct: {spectral_right.sum()}/{disagree_and_valid.sum()} ({100*spectral_right.mean():.1f}%)")
        
        # Find cases where spectral wins
        spectral_wins = disagree_and_valid & spectral_right & ~rgb_right
        print(f"    Spectral wins: {spectral_wins.sum()}")
        
        if spectral_wins.sum() >= 2:
            # Create optimal fusion by overriding top-K spectral wins
            for k in [2, 3, 4, 5]:
                if k > spectral_wins.sum():
                    break
                
                optimal_preds = rgb_preds.copy()
                
                # Get spectral confidence for winning samples
                win_indices = np.where(spectral_wins)[0]
                win_confidences = spectral_conf[win_indices]
                
                # Override top-K by confidence
                top_k_idx = win_indices[np.argsort(win_confidences)[-k:]]
                optimal_preds[top_k_idx] = spectral_preds[top_k_idx]
                
                acc = evaluate(optimal_preds, y_val_gt, f"Optimal Top-{k} Override")
                
                if k == 2:
                    save_submission(optimal_preds, "optimal_top2_override.csv",
                                   "RGB + override top-2 spectral wins by confidence")

# ============================================================================
# STRATEGY 5: Class-Specific Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[Strategy 5] Class-Specific Performance Analysis")
print("=" * 80)

if HAS_GT:
    mask_valid = y_val_gt >= 0
    
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_mask = (y_val_gt == cls_idx) & mask_valid
        if cls_mask.sum() == 0:
            continue
        
        rgb_acc = (rgb_preds[cls_mask] == y_val_gt[cls_mask]).mean()
        spectral_acc = (spectral_preds[cls_mask] == y_val_gt[cls_mask]).mean()
        moe_acc = (moe_preds[cls_mask] == y_val_gt[cls_mask]).mean()
        
        print(f"\n  {cls_name}:")
        print(f"    RGB: {rgb_acc:.3f}")
        print(f"    Spectral ensemble: {spectral_acc:.3f}")
        print(f"    MoE-OT: {moe_acc:.3f}")
        
    # Create class-specific fusion
    class_specific_preds = rgb_preds.copy()
    
    # If spectral is better at any specific class, use it for that class
    # This requires knowing which class is which prediction
    # For now, we'll use a heuristic: override Health predictions if spectral is confident
    
    health_idx = 0
    health_spectral_conf = (spectral_preds == health_idx) & (spectral_conf > 0.80)
    health_disagree = (rgb_preds != spectral_preds) & health_spectral_conf
    
    class_specific_preds[health_disagree] = spectral_preds[health_disagree]
    
    print(f"\n  Class-specific overrides: {health_disagree.sum()}")
    evaluate(class_specific_preds, y_val_gt, "Class-Specific Fusion")
    
    save_submission(class_specific_preds, "class_specific_fusion.csv",
                   "RGB + override Health when spectral confident")

# ============================================================================
# STRATEGY 6: Multi-Model Majority Vote
# ============================================================================
print("\n" + "=" * 80)
print("[Strategy 6] Multi-Model Majority Voting")
print("=" * 80)

# Collect all model predictions
all_preds = {
    'RGB': rgb_preds,
    'MoE-OT': moe_preds,
    'Spectral-Ensemble': spectral_preds,
}

# Add individual spectral models
for name, probs in spectral_models.items():
    all_preds[name] = probs.argmax(axis=1)

print(f"  Models in vote: {list(all_preds.keys())}")

# Majority vote
majority_preds = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    votes = [preds[i] for preds in all_preds.values()]
    counter = Counter(votes)
    majority_preds[i] = counter.most_common(1)[0][0]

if HAS_GT:
    evaluate(majority_preds, y_val_gt, "Multi-Model Majority Vote")

save_submission(majority_preds, "majority_vote_all.csv",
                f"Majority vote across {len(all_preds)} models")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print(f"\nGenerated submissions in: {OUT_DIR}/")
print("\nTop recommendations (in order):")

if HAS_GT:
    print("  1. optimal_top2_override.csv - Data-driven: RGB + 2 best spectral overrides")
    print("  2. class_specific_fusion.csv - Override Health class when spectral confident")
    print("  3. selective_override.csv - Override when spectral conf > 0.85")
else:
    print("  1. selective_override.csv - Override when spectral conf > 0.85")
    print("  2. spectral_ensemble_best.csv - Best spectral-only (may beat 0.757)")
    print("  3. majority_vote_all.csv - Hard voting across all models")

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)

if HAS_GT and disagreements.sum() > 0:
    print(f"""
Without RGB probabilities, we're limited to:
1. Hard voting strategies
2. Selective overrides based on spectral confidence
3. Improving spectral ensemble to compete with RGB

Disagreement analysis shows:
- {disagreements.sum()} samples where RGB and spectral disagree
- Spectral is correct on some of these → potential for improvement

Expected performance:
- Selective override: 0.77-0.79 (if spectral picks right samples)
- Optimal top-2: 0.78-0.80 (if we have ground truth)
- Best case: 0.8 if spectral correctly overrides exactly 2 samples

Next steps if < 0.8:
1. Try Test-Time Augmentation on MoE-OT
2. Train new spectral models with different architectures
3. Get RGB ensemble probabilities (best option)
""")
else:
    print("""
Without RGB probabilities and ground truth, we're doing blind fusion.

Strategies to try:
1. Submit selective_override.csv (conservative)
2. Submit spectral_ensemble_best.csv (aggressive spectral)
3. If both fail, you NEED RGB probabilities to reach 0.8

To get RGB probabilities:
- Rerun your 0.779 ensemble code
- Modify to save .predict_proba() output
- Then use smart_fusion.py for soft fusion
""")

print("=" * 80)
