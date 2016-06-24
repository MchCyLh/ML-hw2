#!/usr/bin/env sh

ROOT=.
TRAIN=$ROOT/data/train.txt
SUBTRAIN=$ROOT/intermediate/subtrain.txt
SUBTEST=$ROOT/intermediate/subtest.txt
RATE=0.5

TOOLS=$ROOT/tools

python $TOOLS/create_sub_data.py \
	$TRAIN \
	$SUBTRAIN \
	$SUBTEST \
	$RATE

echo "Done."
