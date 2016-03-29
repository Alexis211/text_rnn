
import numpy
from numpy.random import RandomState

from blocks.algorithms import AdaDelta, Momentum, RMSProp, CompositeRule, BasicMomentum, Adam
from blocks.bricks import Tanh, Rectifier
from blocks.initialization import IsotropicGaussian, Constant

from model.hpc_lstm import Model

dataset = 'data/logcompil-2016-03-07.txt'

io_dim = 256
repr_dim = 128
embedding_matrix = (RandomState(42).binomial(1, 0.1, ((io_dim, repr_dim)))
                   -RandomState(123).binomial(1, 0.1, ((io_dim, repr_dim))))

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 20
seq_len = 5000
seq_div_size = 50

hidden_dims = [128, 192, 256, 512]
cost_factors = [1., 1., 1., 1.]
hidden_q = [0.5, 0.5, 0.5, 0.5]
activation_function = Tanh()

out_hidden = [512]
out_hidden_act = [Rectifier]

weight_noise = 0.05

step_rule = Adam()
#step_rule = CompositeRule([RMSProp(learning_rate=0.01),
#						   BasicMomentum(momentum=0.9)])
#step_rule = Momentum(learning_rate=.1, momentum=0.9)

weights_init = IsotropicGaussian(0.1)
biases_init = Constant(0.01)

# parameter saving freq (number of batches)
monitor_freq = 500
save_freq = monitor_freq

# used for sample generation and IRC mode
sample_temperature = 0.5 #0.7

# do we want to generate samples at times during training?
sample_len = 1000
sample_freq = monitor_freq
sample_init = '\nalex\ttu crois?\n'

