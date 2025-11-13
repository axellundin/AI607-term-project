

#!/bin/bash

# -----------------------------
# Pixie hyperparameter sweep script
# -----------------------------

# 데이터 경로
TRAIN_PATH="../../../data/task2_train.tsv"
VAL_PATH="../../../data/task2_val_queries.tsv"
VAL_ANS_PATH="../../../data/task2_val_answers.tsv"

# 탐색할 파라미터 값들
STEPS_LIST=(10 50 100 200)
ALPHA_LIST=(0.3 0.4 0.5 0.6)
TOPK_LIST=(30 50)

BEST_SCORE=0.0
BEST_CONFIG=""

echo "=========================================="
echo "   Pixie Parameter Search Started"
echo "=========================================="

for steps in "${STEPS_LIST[@]}"; do
  for alpha in "${ALPHA_LIST[@]}"; do
    for topk in "${TOPK_LIST[@]}"; do
      echo "------------------------------------------"
      echo "Running: steps=${steps}, alpha=${alpha}, topk=${topk}"
      echo "------------------------------------------"

      # 실행
      OUTPUT=$(CUDA_VISIBLE_DEVICES=1 python main.py \
        --train-path $TRAIN_PATH \
        --val-path $VAL_PATH \
        --val-answers-path $VAL_ANS_PATH \
        --steps $steps \
        --alpha $alpha \
        --topk $topk \
        2>&1)

      echo "$OUTPUT"

      SCORE=$(echo "$OUTPUT" | grep "Final validation score" | awk '{print $4}')

      if [ ! -z "$SCORE" ]; then
        COMP=$(echo "$SCORE > $BEST_SCORE" | bc -l)
        if [ "$COMP" -eq 1 ]; then
          BEST_SCORE=$SCORE
          BEST_CONFIG="steps=${steps}, alpha=${alpha}, topk=${topk}"
        fi
      fi

      echo ""
    done
  done
done

echo "=========================================="
echo "   Search Completed"
echo "   Best Score: ${BEST_SCORE}"
echo "   Best Config: ${BEST_CONFIG}"
echo "=========================================="