import sys
import SGA

if __name__ == '__main__':
    # parameter : model_file
    # parameter : subtest.txt
    # parameter : dimension 
    model_file = sys.argv[1]
    subtest = sys.argv[2]
    dimension = int(sys.argv[3])
    models = SGA.getModelFromFile(model_file, dimension)
    loss_rst = SGA.loss(models, subtest)
    print ('model_file: %s' % model_file)
    print ('test_file: %s' % subtest)
    print ('loss: %f' % loss_rst)
