#!/usr/bin/env sh

ROOT=.
DIMENSION=11392
NUM_PPN=1
SUB_MODEL_BASE_NAME=$ROOT/output/parallel_model

python $ROOT/tools/create_parallel_model.py \
	$DIMENSION \
	$NUM_PPN \
	$SUB_MODEL_BASE_NAME

echo "Done."
