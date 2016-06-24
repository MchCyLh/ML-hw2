#!/usr/bin/env sh

ROOT=.
MODEL_FILE=$ROOT/output/model.txt
PREDICT_FILE=$ROOT/data/test.txt
RESULT_FILE=$ROOT/submit/submission.csv

TOOLS=$ROOT/tools

python $TOOLS/predict.py \
	$MODEL_FILE \
	$PREDICT_FILE \
	$RESULT_FILE

echo "Done."
