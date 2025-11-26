

#!/bin/bash

# -----------------------------
# Pixie hyperparameter sweep script
# -----------------------------

# 데이터 경로
TRAIN_PATH="../../../data/task2_train.tsv"
VAL_PATH="../../../data/task2_val_queries.tsv"
VAL_ANS_PATH="../../../data/task2_val_answers.tsv"

# 탐색할 파라미터 값들
STEPS_LIST=(2000)
ALPHA_LIST=(0.6)
TOPK_LIST=(50)
LAMBDA_LIST=(1 2 3 4)

BEST_SCORE=0.0
BEST_CONFIG=""

echo "=========================================="
echo "   Pixie Parameter Search Started"
echo "=========================================="

for lambda in "${LAMBDA_LIST[@]}"; do
  for alpha in "${ALPHA_LIST[@]}"; do
    for topk in "${TOPK_LIST[@]}"; do
      echo "------------------------------------------"
      echo "Running: lambdaS=${lambda}, alpha=${alpha}, topk=${topk}"
      echo "------------------------------------------"

      # 실행
      OUTPUT=$(CUDA_VISIBLE_DEVICES=1 python main.py \
        --train-path $TRAIN_PATH \
        --val-path $VAL_PATH \
        --val-answers-path $VAL_ANS_PATH \
        --steps 2000 \
        --alpha $alpha \
        --topk $topk \
        --lambdaS $lambda \
        --lambdaB 1 \
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