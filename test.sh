#!/usr/bin/env sh

ROOT=.
MODEL_FILE=$ROOT/output/model.txt
TEST_FILE=$ROOT/intermediate/subtest.txt
DIMENSION=11392

TOOLS=$ROOT/tools

python $TOOLS/test.py \
	$MODEL_FILE \
	$TEST_FILE \
	$DIMENSION

echo "Done."
