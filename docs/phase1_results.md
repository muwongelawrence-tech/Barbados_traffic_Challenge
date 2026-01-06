## Phase 1 Results Summary

### Completed Configurations

| Version | Configuration | Val Score | vs Baseline | Status |
|---------|--------------|-----------|-------------|---------|
| v1 | Baseline (31 leaves, lr=0.05) | 0.5745 | - | ✅ Baseline |
| v2 | Increased complexity (50 leaves, lr=0.03) | 0.5324 | -0.0421 | ❌ Worse |
| v3 | Lower LR (lr=0.01, 1000 rounds) | TBD | TBD | ⏸️ Stopped |
| v4 | Regularization (L1/L2) | - | - | ⏳ Pending |
| v5 | **Class weights** | - | - | ⏳ **Next - High Priority** |
| v6 | Deep trees (100 leaves) | - | - | ⏳ Pending |

### Key Findings

**v2 (Increased Complexity):**
- Score: 0.5324 (worse than baseline)
- Entrance F1: 0.5466 (vs 0.5659 baseline)
- Exit F1: 0.2505 (vs 0.3545 baseline)
- **Issue:** Model is predicting almost all exit samples as "free flowing"
- **Lesson:** More complexity doesn't always help

### Next Action

Focus on **v5 (Class Weights)** - most promising for improving minority class performance.

**Why v5 is critical:**
- Exit congestion has severe class imbalance (95% free flowing)
- Class weights will force model to learn minority classes
- Expected improvement: +0.02-0.03 on combined score
- Should significantly improve F1 for light/moderate/heavy delay

### Submission Files Ready

- ✅ `submission_v1_baseline_temporal_only.csv`
- ✅ `submission_v2_v2_increased_complexity.csv`
