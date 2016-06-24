from math import exp
from numpy import shape
from numpy import ones
import random
import time
import mpi4py.MPI as MPI
import math


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

def getModelFromFile(model_file, dimension):
    fr = open(model_file, 'r')
    weights = [float(x) for x in fr.readline().strip().split()]
    fr.close()
    if len(weights) != dimension + 1:
        print ('Model from \'%s\' is invalid. Model is initialized again.' % model_file)
        weights = [0] * ( dimension + 1 )
        setModelForFile(weights, model_file)
    return weights

def setModelForFile(weights, model_file):
    fw = open(model_file, 'w')
    for weight in weights:
        fw.write('%f ' % weight)
    # fw.write('%s\n' % ' '.join(weights))
    fw.close()

def sendModel(weights, root=0):
    comm = MPI.COMM_WORLD
    comm.reduce(weights, root=0, op = MPI.SUM)

def loss(model, data_file):
    rst = 0.0
    fr = open (data_file, 'r')
    cnt = 0
    for line in fr:
        cnt += 1
        sparseMatrix, classLabel = parseDataLine(line)
        # print (sparseMatrix)
        hi = sigmoid( sparseSum( sparseMatrix, model ) )
        rst += classLabel * (math.log(hi)) + (1-classLabel) * (math.log(1-hi))
    fr.close()
    return rst / -cnt


def stocGradAscent( train_file, model_file, numIter = 150 , send_interval = 5, dimension = 11392, base_lr=0.01):

    weights = getModelFromFile(model_file, dimension) # bias + feature
    for j in range( numIter ):

        shuffleData( train_file )
        fr = open ( train_file, 'r' )

        i = 0
        for line in fr:
           sparseMatrix, classLabel = parseDataLine( line )
           alpha = 4 / (1.0 + j + i) + base_lr
           h = sigmoid( sparseSum( sparseMatrix, weights ) )
           error = classLabel - h
           # update weights
           updateWeights( weights, sparseMatrix, alpha, error )

           i += 1
        fr.close()

        if (j+1) % send_interval == 0:
            setModelForFile(weights, model_file)
            sendModel(weights)
    setModelForFile(weights, model_file)
    sendModel(weights)
