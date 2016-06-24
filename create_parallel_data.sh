#!/usr/bin/env sh

ROOT=.
SUBTRAIN=$ROOT/intermediate/subtrain.txt
NUM_PPN=1
PARALLEL_DATA_BASE_NAME=$ROOT/intermediate/parallel_data

TOOLS=$ROOT/tools

python $TOOLS/create_parallel_data.py \
	$SUBTRAIN \
	$NUM_PPN \
	$PARALLEL_DATA_BASE_NAME

echo "Done."
