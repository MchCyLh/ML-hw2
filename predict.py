from assist import *

def classifyVector( inX, weights ):
    prob = sigmoid( sparseSum( inX, weights ) )
    if prob > 0.5: return 1
    else: return 0

def testPredict( weight_file='data/weight_draw.txt', test_file='data/sub_test.txt' , rst_file='data/result.txt'):
    fwf = open( weight_file, 'r' )
    weights = [ float(x) for x in fwf.readline().strip().split()]
    fwf.close()

    error_count = 0
    tot_count = 0
    ftf = open( test_file, 'r' )
    frf = open( rst_file, 'w')
    frf.write('id,label\n')
    for line in ftf:
        sparseMatrix, classLabel = parseDataLine(line)
        label = classifyVector( sparseMatrix, weights )
        if label != classLabel:
            error_count += 1
        frf.write('%d,%d\n' % (tot_count, label))
        tot_count += 1

    frf.write( 'Error rate: %f' % ( float(error_count) / float(tot_count) ) )
    print ('Error rate: %f' % (float(error_count) / float(tot_count) ))
    ftf.close()
    frf.close()



