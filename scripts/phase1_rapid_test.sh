#!/bin/bash
# Rapid Phase 1 testing script
# Runs all 6 configurations and generates submissions

source barbados/bin/activate

echo "========================================="
echo "PHASE 1: RAPID HYPERPARAMETER TESTING"
echo "========================================="

# Array to store scores
declare -A scores

# v2: Increased complexity
echo -e "\n[1/6] Testing v2: Increased complexity..."
python scripts/train_quick.py --config v2_complexity --output-dir models/v2_complexity
python scripts/predict.py
python scripts/manage_submissions.py create "v2_increased_complexity" submission.csv
scores["v2"]=$(tail -1 models/v2_complexity/score.txt 2>/dev/null || echo "N/A")

# v3: Lower learning rate
echo -e "\n[2/6] Testing v3: Lower learning rate..."
python scripts/train_quick.py --config v3_lower_lr --output-dir models/v3_lower_lr --num-boost-round 1000
python scripts/predict.py
python scripts/manage_submissions.py create "v3_lower_lr_1000rounds" submission.csv
scores["v3"]=$(tail -1 models/v3_lower_lr/score.txt 2>/dev/null || echo "N/A")

# v4: Regularization
echo -e "\n[3/6] Testing v4: Regularization..."
python scripts/train_quick.py --config v4_regularized --output-dir models/v4_regularized
python scripts/predict.py
python scripts/manage_submissions.py create "v4_regularized_l1_l2" submission.csv
scores["v4"]=$(tail -1 models/v4_regularized/score.txt 2>/dev/null || echo "N/A")

# v5: Class weights (BIG WIN expected)
echo -e "\n[4/6] Testing v5: Class weights..."
python scripts/train_quick.py --config v2_complexity --class-weights --output-dir models/v5_class_weights
python scripts/predict.py
python scripts/manage_submissions.py create "v5_class_weights_balanced" submission.csv
scores["v5"]=$(tail -1 models/v5_class_weights/score.txt 2>/dev/null || echo "N/A")

# v6: Deep trees
echo -e "\n[5/6] Testing v6: Deep trees..."
python scripts/train_quick.py --config v6_deep_trees --output-dir models/v6_deep_trees
python scripts/predict.py
python scripts/manage_submissions.py create "v6_deep_trees_100leaves" submission.csv
scores["v6"]=$(tail -1 models/v6_deep_trees/score.txt 2>/dev/null || echo "N/A")

# v7: Best config ensemble
echo -e "\n[6/6] Creating ensemble from best models..."
# This will be implemented after we see which configs work best

echo -e "\n========================================="
echo "PHASE 1 TESTING COMPLETE!"
echo "========================================="
echo "Validation Scores:"
echo "  v2 (Complexity):    ${scores[v2]}"
echo "  v3 (Lower LR):      ${scores[v3]}"
echo "  v4 (Regularized):   ${scores[v4]}"
echo "  v5 (Class Weights): ${scores[v5]}"
echo "  v6 (Deep Trees):    ${scores[v6]}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Upload all 6 submissions to Zindi"
echo "2. Record leaderboard scores"
echo "3. Identify best configuration"
echo "4. Move to Phase 1 Day 2 (advanced features)"
echo "========================================="
