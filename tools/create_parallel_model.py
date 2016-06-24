import sys


# dimension, num of parallel process nodes and sub model base name
if __name__ == '__main__':
    dimension = int(sys.argv[1])
    num_ppn = int(sys.argv[2])
    sub_model_base_name = sys.argv[3]

    for i in range(num_ppn):
        fw = open('%s_%d.txt' % (sub_model_base_name, i+1), 'w')
        for cnt in range(dimension + 1): # 1 for bias
            fw.write('0 ')
        fw.close()
