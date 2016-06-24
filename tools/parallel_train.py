import mpi4py.MPI as MPI
import sys
import SGA
import time

# parameter : parallel_data base name	[1]
# parameter : parallel_model base name	[2]
# parameter : # of parallel process node	[3]
# parameter : total iterations	[4]
# parameter : iteration interval	[5]
# parameter : integrated model_file	[6]
# parameter : feature dimension [7]
# parameter : base learning rate [8]
# parameter : sub_train.txt [9]
# parameter : test_iter [10]
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    num_ppn = int(sys.argv[3])
    model_file = sys.argv[6]
    sub_train = sys.argv[9]
    test_iter = int(sys.argv[10])
    
parallel_data_base_name = sys.argv[1]
parallel_model_base_name = sys.argv[2]
total_iters = int(sys.argv[4])
iter_interval = int(sys.argv[5])
dimension = int(sys.argv[7])
base_lr = float(sys.argv[8])

if rank == 0:
    # master process doing 
    loops = int(total_iters / iter_interval) + 1
    for loop in range(loops):
        sub_models = comm.reduce([], root=0, op = MPI.SUM)
        models = [0.0] * (dimension + 1)
        for i in range(len(sub_models)):
            models[i%(dimension+1)] += sub_models[i]
        models = [ weight / num_ppn for weight in models ]
        SGA.setModelForFile(models, model_file)
        # do something
        iteration = min((loop+1) * iter_interval, total_iters)
        if iteration % test_iter == 0:
            lossi = SGA.loss(models, sub_train)
            print ('(Time: %s) Interation %d : loss %f' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), iteration , lossi ) )
        else:
            print ('(Time: %s) Interation %d' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) , iteration ) )

else:
    # follewer process doing
    sub_parallel_data = '%s_%d.txt' % (parallel_data_base_name, rank)
    sub_parallel_model = '%s_%d.txt' % (parallel_model_base_name, rank)
    SGA.stocGradAscent(train_file = sub_parallel_data, model_file = sub_parallel_model, numIter = total_iters, send_interval = iter_interval, dimension=dimension, base_lr=base_lr)
