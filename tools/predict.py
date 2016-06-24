from SGA import *
import sys

def classifyVector( inX, weights ):
    prob = sigmoid( sparseSum( inX, weights ) )
    if prob > 0.5: return 1
    else: return 0

if __name__ == '__main__':
    # parameter : weight_file
    # parameter : test_file
    # parameter : rst_file
    weight_file= sys.argv[1]
    test_file= sys.argv[2]
    rst_file= sys.argv[3]
    
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

    # frf.write( 'Error rate: %f' % ( float(error_count) / float(tot_count) ) )
    ftf.close()
    frf.close()
