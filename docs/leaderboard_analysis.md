# Leaderboard Analysis

## Current Top 3 (as of 2026-01-06)

| Rank | User | Score | Submissions | Last Submission | Gap from #1 |
|------|------|-------|-------------|-----------------|-------------|
| 1 | kdylky | **0.6532** | 21 | ~1 month ago | - |
| 2 | ct201314ct | 0.6531 | 188 | 28 days ago | -0.0001 |
| 3 | 30NNY | 0.6489 | 166 | 11 days ago | -0.0043 |

---

## Our Position

**Baseline (v1):**
- **Validation Score:** 0.5745
- **Expected Leaderboard:** ~0.55-0.58 (validation often differs from test)
- **Gap from #1:** ~0.10 (10 percentage points)
- **Estimated Rank:** Mid-tier (likely 10-20th place)

---

## Score Timing

**How long for score to appear:**
- Usually: **30 seconds to 2 minutes**
- Sometimes: Up to 5 minutes during high traffic
- If no score after 5 minutes: Check submission status or refresh page

**Possible reasons for delay:**
- ✅ Processing queue (many submissions)
- ✅ Large file size (431KB is fine)
- ❌ Format error (but we validated, so unlikely)
- ❌ Duplicate submission

**What to do:**
1. Refresh the page
2. Check "Submissions" tab for status
3. Look for error messages
4. If still nothing after 5 min, re-upload

---

## Gap Analysis: How to Close the 0.10 Gap

### Current Score Breakdown

**Our baseline (validation):**
- Entrance: F1 = 0.5659, Acc = 0.7307 → Combined = 0.6154
- Exit: F1 = 0.3545, Acc = 0.9515 → Combined = 0.5336
- **Average: 0.5745**

**Top score: 0.6532**
- **Gap: 0.0787** (7.87 percentage points)

### Where We're Losing Points

1. **Exit Congestion Prediction** (Biggest weakness)
   - Our F1: 0.3545
   - Need: ~0.55+ 
   - **Potential gain: +0.10**

2. **Entrance Minority Classes**
   - Light delay F1: 0.36 (need: 0.60+)
   - Moderate delay F1: 0.44 (need: 0.65+)
   - Heavy delay F1: 0.58 (need: 0.70+)
   - **Potential gain: +0.05**

3. **No Video Features**
   - We're using only 17 temporal features
   - Top competitors likely use 50-100+ features
   - **Potential gain: +0.10-0.15**

---

## Improvement Roadmap to Top 3

### Phase 1: Quick Wins (+0.03-0.05)
**Target: 0.60-0.62 (1-2 days)**

- [ ] Hyperparameter tuning
  - Increase `num_leaves` (31 → 50-100)
  - Tune `learning_rate` (0.05 → 0.01-0.03)
  - Adjust `min_child_samples`
- [ ] Class weights for imbalance
  - Weight minority classes higher
  - Use `class_weight='balanced'`
- [ ] Try XGBoost/CatBoost
  - Different algorithms may work better
- [ ] Add more temporal features
  - Lag features (5, 10, 15 minutes back)
  - Rolling statistics (mean, std, min, max)
  - Rate of change

**Expected improvement:** 0.5745 → 0.60-0.62

---

### Phase 2: Video Features (+0.05-0.08)
**Target: 0.65-0.68 (3-5 days)**

- [ ] Classical CV feature extraction
  - Vehicle counts (entrance/exit)
  - Queue length estimation
  - Optical flow magnitude
  - Spatial occupancy
- [ ] Video-based temporal patterns
  - Vehicle entry/exit rates
  - Flow variance over time
  - Congestion buildup speed
- [ ] Camera-specific patterns
  - Per-camera vehicle counts
  - Location-based flow patterns

**Expected improvement:** 0.60-0.62 → 0.65-0.68

---

### Phase 3: Advanced Techniques (+0.02-0.05)
**Target: 0.67-0.70+ (Top 3)**

- [ ] Ensemble methods
  - LightGBM + XGBoost + CatBoost
  - Weighted averaging
  - Stacking
- [ ] Advanced feature engineering
  - Interaction features
  - Polynomial features
  - Feature selection
- [ ] Model-specific optimization
  - Custom loss function (0.7×F1 + 0.3×Acc)
  - Focal loss for hard examples
  - Calibration

**Expected improvement:** 0.65-0.68 → 0.67-0.70+

---

## Realistic Timeline

| Day | Target Score | Actions | Expected Rank |
|-----|--------------|---------|---------------|
| **Today** | 0.57-0.58 | Baseline + quick tuning | 15-20 |
| **Day 2** | 0.60-0.62 | Hyperparams + class weights | 10-15 |
| **Day 3-5** | 0.65-0.68 | Video features | 5-10 |
| **Day 6-10** | 0.67-0.70 | Ensemble + advanced | **Top 3** |

---

## Key Insights from Top Competitors

### kdylky (#1: 0.6532)
- **21 submissions** - Very efficient!
- Last submission ~1 month ago
- Likely found good solution early
- **Strategy:** Quality over quantity

### ct201314ct (#2: 0.6531)
- **188 submissions** - Extensive testing!
- Very close to #1 (0.0001 difference)
- Last submission 28 days ago
- **Strategy:** Exhaustive search

### 30NNY (#3: 0.6489)
- **166 submissions** - Also extensive
- Last submission 11 days ago
- Still active and improving
- **Strategy:** Continuous iteration

### Our Strategy
- Start with baseline (today)
- Rapid iteration with 20 daily submissions
- Focus on video features (biggest gap)
- Ensemble in final days
- **Target:** Top 3 in 7-10 days

---

## Immediate Next Steps

1. **Wait for leaderboard score** (refresh page)
2. **Record actual score** in SUBMISSION_LOG.md
3. **Start v2 improvements:**
   - Tune hyperparameters
   - Add class weights
   - Try different models

**Most important:** Don't be discouraged by the gap! 
- 0.10 is very achievable
- Video features alone can give +0.10-0.15
- We have 20 submissions/day to iterate
- Competition closes in 20 days - plenty of time!

---

## Confidence Level

**Can we reach Top 3?** ✅ **YES!**

**Why:**
- We have the right approach (classical CV + LightGBM)
- Video features are untapped (+0.10-0.15 potential)
- 20 submissions/day × 20 days = 400 attempts
- Top competitors are inactive (last submission weeks ago)
- We're starting from solid baseline

**Timeline:** 7-10 days to Top 3 if we execute well
