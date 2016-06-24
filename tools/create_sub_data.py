import sys
import random
# parameter $1 for rate of subtrain.
if __name__ == '__main__':
    train = sys.argv[1]
    subtrain = sys.argv[2]
    subtest = sys.argv[3]
    rate = float(sys.argv[4])

    fr_train = open(train, 'r')
    fw_subtrain = open(subtrain, 'w')
    fw_subtest = open(subtest, 'w')
    
    for line in fr_train:
        r = random.random()
        if r < rate:
            fw_subtrain.write(line)
        else:
            fw_subtest.write(line)

    fw_subtest.close()
    fw_subtrain.close()
    fr_train.close()
