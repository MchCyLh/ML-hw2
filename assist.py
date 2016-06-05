from math import exp
from numpy import shape
from numpy import ones
import random
import time

def createSubSet(train_file = 'data/train.txt', sub_train='data/sub_train.txt', sub_test='data/sub_test.txt'):
    fr = open ( train_file, 'r')
    fw_train = open( sub_train, 'w' )
    fw_test = open( sub_test, 'w' )

    for line in fr:
        r = random.random()
        if  r < 0.8:
            fw_train.write(line)
        else:
            fw_test.write(line)

    fr.close()
    fw_train.close()
    fw_test.close()


def loadDataSet( filename ):
    sparseMat,labelMat = [],[]
    fr = open( filename , 'r' )
    for line in fr.readlines():
        lineArr = line.strip().split()

        row = []
        for item in lineArr[ 1 : ] :
            index, value = item.split( ':' )
            row.append( ( int(index), float(value) ) )

        sparseMat.append( row )
        labelMat.append( float( lineArr[0] ) )
    fr.close()
    return sparseMat, labelMat

def sigmoid( inX ):
    return 1.0 / ( 1 + exp( -inX ) )

def shuffleData( train_file ):
    fr = open( train_file , 'r' )
    lines = fr.readlines()
    fr.close()
    random.shuffle(lines)
    fw = open( train_file, 'w' )
    for line in lines:
        fw.write(line)
    fw.close()

def parseDataLine( line ):
    sparseMatrix, classLabel = [], None
    data = line.strip().split()
    if not ':' in data[0]:  # make sure parse test set as well as train set
        classLabel = float( data[0] )
        data.pop( 0 )
    for item in data:
        index, value = item.split(':')
        sparseMatrix.append( ( int(index), float(value) ) )
    return sparseMatrix, classLabel

def sparseSum( sparseMatrix, weights ):
    rst = 1 * weights[0] # bias for sparseMatrix[0] default to 1
    for index, value in sparseMatrix:
        rst += value * weights[index]
    return rst

def updateWeights( weights, sparseMatrix, alpha, error ):
    weights[0] = weights[0] + alpha * error * 1    # update bias weights[0]
    for index, value in sparseMatrix:
        weights[index] = weights[index] + alpha * error * value

def stocGradAscent1( train_file, weights_file = "data/weights.txt", feature_dimension = 11392, numIter = 150 ):
    fw = open(weights_file, 'a')
    fw.write("-----BEGIN A NEW stocGradAscent1-----\n")
    fw.close()

    weights = ones( feature_dimension + 1 ) # bias + feature
    for j in range( numIter ):
        # begin calculating time
        sTime = time.time()

        shuffleData( train_file )
        fr = open ( train_file, 'r' )

        i = 0
        for line in fr:
           sparseMatrix, classLabel = parseDataLine( line )
           alpha = 4 / (1.0 + j + i) + 0.01
           h = sigmoid( sparseSum( sparseMatrix, weights ) )
           error = classLabel - h
           # update weights
           updateWeights( weights, sparseMatrix, alpha, error )

           i += 1
        fr.close()

        eTime = time.time()
        print ("One iteration cost %d sec." % (eTime - sTime))
        # end calculating time

        # record the weight
        fw = open(weights_file, 'a')
        fw.write( ' '.join( [ str(e) for e in weights ] ) )
        fw.write('\n')
        fw.close()

    return weights

def stocGradAscentSample( train_file, weights_file = "data/weights_sample.txt", feature_dimension = 11392, numIter = 150, sample_rate = 0.1):
    fw = open(weights_file, 'a')
    fw.write("-----BEGIN A NEW stocGradAscentSample-----\n")
    fw.close()

    weights = ones( feature_dimension + 1 ) # bias + feature
    for j in range( numIter ):
        # begin calculating time
        sTime = time.time()

        fr = open( train_file, 'r' )
        lines = fr.readlines()
        fr.close()

        samples = random.sample( lines, int(len(lines) * sample_rate))
        del lines

        i = 0
        for line in samples:
           sparseMatrix, classLabel = parseDataLine( line )
           alpha = 4 / (1.0 + j + i) + 0.01
           h = sigmoid( sparseSum( sparseMatrix, weights ) )
           error = classLabel - h
           # update weights
           updateWeights( weights, sparseMatrix, alpha, error )
           i += 1

        eTime = time.time()
        print ("One iteration cost %d sec." % (eTime - sTime))
        # end calculating time

        # record the weight
        fw = open(weights_file, 'a')
        fw.write( ' '.join( [ str(e) for e in weights ] ) )
        fw.write('\n')
        fw.close()

    return weights

