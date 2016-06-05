import numpy as np
import pylab as pl

def getWeightXY(filename = 'data/weight_draw.txt', y = '0'):
    X,Y = [],[]
    fr = open( filename, 'r' )
    index = 0
    for line in fr:
        index += 1
        data = line.strip().split()
        X.append( index )
        Y.append( data[y] )
    fr.close()
    return X,Y

def drawPlot(xMat, yMat, x_label = '# of Iteration', y_label = 'weight', x_lim = (0.0, 30.0), y_lim = (-2.0, 5.0), title = 'Weight Change'):
    for index in range( len( xMat ) ):
        pl.plot( xMat[index], yMat[index] )
    pl.title(title)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.xlim(x_lim)
    pl.ylim(y_lim)
    pl.show()

def drawPlotOne(x, y, x_label = '# of Iteration', y_label = 'weight', x_lim = (0.0, 30.0), y_lim = (-2.0, 5.0), title = 'Weight Change'):
    pl.plot(x, y)
    pl.title(title)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.xlim(x_lim)
    pl.ylim(y_lim)
    pl.show()