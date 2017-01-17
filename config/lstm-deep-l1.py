from blocks.algorithms import AdaDelta
from blocks.bricks import Tanh

from model.lstm import Model

dataset = 'data/logcompil.txt'
io_dim = 256

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 100
seq_len = 5000
seq_div_size = 200

layers = [
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
	{'dim':		512},
]
activation_function = Tanh()

i2h_all = True             # input to all hidden layers or only first layer
h2o_all = True             # all hiden layers to output or only last layer

w_noise_std = 0
i_dropout = 0

l1_reg = 0
l1_reg_weight = 0.05

step_rule = AdaDelta()

# parameter saving freq (number of batches)
monitor_freq = 100
save_freq = 100

# used for sample generation and IRC mode
#sample_temperature = 0.7 #0.5
sample_temperature = 0.9 #0.5

# do we want to generate samples at times during training?
sample_len = 1000
sample_freq = 100
sample_init = '\nalex\ttu crois?\n'

