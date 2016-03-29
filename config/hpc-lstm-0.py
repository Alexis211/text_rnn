import numpy
from numpy.random import RandomState

from blocks.algorithms import AdaDelta, Momentum
from blocks.bricks import Tanh, Rectifier

from model.hpc_lstm import Model

dataset = 'data/logcompil-2016-03-07.txt'

io_dim = 256
repr_dim = 256
embedding_matrix = numpy.eye(io_dim)

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 100
seq_len = 2000
seq_div_size = 100

hidden_dims = [128, 128, 256, 512]
cost_factors = [1., 1., 1., 1.]
hidden_q = [0.1, 0.15, 0.22, 0.33]
activation_function = Tanh()

out_hidden = [512]
out_hidden_act = [Rectifier]

step_rule = AdaDelta()
#step_rule = Momentum(learning_rate=0.0001, momentum=0.99)

# parameter saving freq (number of batches)
monitor_freq = 10
save_freq = 100

# used for sample generation and IRC mode
sample_temperature = 0.7 #0.5

# do we want to generate samples at times during training?
sample_len = 1000
sample_freq = 100
sample_init = '\nalex\ttu crois?\n'

