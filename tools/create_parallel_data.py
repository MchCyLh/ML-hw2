import sys
import random

# param : subtrain.txt, the number of Parallel Process Nodes and the base name of PPN
if __name__ == '__main__':
    subtrain = sys.argv[1]
    # print(subtrain)
    num_ppn = int(sys.argv[2])
    # print(num_ppn)
    parallel_data_base_name = sys.argv[3]

    fr_subtrain = open(subtrain, 'r')
    lines = fr_subtrain.readlines()
    fr_subtrain.close()

    random.shuffle(lines)

    files = []
    for i in range(num_ppn):
        file = open('%s_%d.txt' % (parallel_data_base_name, i+1), 'w')
        files.append(file)

    for i in range(len(lines)):
        files[i % num_ppn].write(lines[i])

    for file in files:
        file.close()
