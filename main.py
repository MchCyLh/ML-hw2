from assist import *
from draw import *
from predict import *
import time
import numpy as np
import pylab as pl


def testMatrix():
    m,n = 10000,1000
    m1 = ones((1,m))
    m2 = ones((m,1))

    sTime = time.time()
    ans = sum(m1 * m2)
    eTime = time.time()
    print ( 'Lib : Cost time %d sec.' % (eTime - sTime) )

    sTime = time.time()
    ans = 0
    for index in range(m):
        ans += m1[0][index]*m2[index][0]
    eTime = time.time()
    print ( 'Normal : Cost time %d sec.' % (eTime - sTime) )

def drawTester():
    xMat, yMat = [],[]
    dimension, sample_rate = 11392, 0.01
    samples = random.sample(range(dimension), int(dimension * sample_rate) )
    for i in samples:
        X, Y = getWeightXY('data/train_weights.txt', i)
        xMat.append( X )
        yMat.append( Y )
        # drawPlot([X],[Y])
    drawPlot( xMat, yMat, x_lim = (0.0, 150.0))

def predictTester():
    index = 200
    testPredict( weight_file='data/predict_weights.txt', test_file='data/test.txt' , rst_file='data/result_train_%d_iteration.csv'%index)

def getWeightTester():
    iteration = 200
    print ( "Program Begin !" )
    weights = stocGradAscent1(  train_file='data/train.txt', weights_file = "data/train_weights_%d_iteration.txt" % iteration, feature_dimension = 11392, numIter = iteration )
    print ( "Program Done !" )

if __name__ == '__main__':
    # predictTester()
    # drawTester()
    # createSubSet()
    # testMatrix()
    getWeightTester()
