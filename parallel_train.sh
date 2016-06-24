#!/usr/bin/env sh

ROOT=.
PARALLEL_DATA_BASE_NAME=$ROOT/intermediate/parallel_data
PARALLEL_MODEL_BASE_NAME=$ROOT/output/parallel_model
NUM_PPN=1
TOTAL_ITERS=150
ITER_INTERVAL=1
MODEL_FILE=$ROOT/output/model.txt
DIMENSION=11392
BASE_LR=0.01
SUB_TRAIN=$ROOT/intermediate/subtrain.txt
TEST_ITER=10
TOOLS=$ROOT/tools

TOT_PPN=2
# parameter : parallel_data base name	[1]
# parameter : parallel_model base name	[2]
# parameter : # of parallel process node	[3]
# parameter : total iterations	[4]
# parameter : iteration interval	[5]
# parameter : integrated model_file	[6]
# parameter : feature dimension [7]
# parameter : base learning rate [8]
# parameter : sub_train.txt [9]

mpirun -n $TOT_PPN python $TOOLS/parallel_train.py \
        $PARALLEL_DATA_BASE_NAME \
        $PARALLEL_MODEL_BASE_NAME \
        $NUM_PPN \
        $TOTAL_ITERS \
        $ITER_INTERVAL \
        $MODEL_FILE \
        $DIMENSION \
        $BASE_LR \
        $SUB_TRAIN \
        $TEST_ITER

echo "Done."
