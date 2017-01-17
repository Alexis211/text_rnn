from blocks.algorithms import AdaDelta
from blocks.bricks import Tanh
from blocks.initialization import IsotropicGaussian, Constant

from model.condlstm import Model

dataset = 'data/logcompil.txt'
io_dim = 256

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 50
seq_len = 5000
seq_div_size = 200

layers = [
	# Slowlier
	{'dim':			128,
	 'reset_after': ' \t\n,.:;/!?()[]{}<>\\\'"*+-^_|#~&`@$%',
	},
	{'dim':			128,
	 'run_on':		' \t\n,.:;/!?()[]{}<>\\\'"*+-^_|#~&`@$%',
	 'reset_after': ' \t\n',
	},
	{'dim':			128,
	 'run_on':		' \t\n',
	 'reset_after': '\t\n',
	},
	{'dim':			256,
	 'run_on':		' \t\n',
	 'reset_after': '\n',
	},
	{'dim':			512,
	 'run_on':		'\t\n',
	},
	# Slowest
	{'dim':			512,
	 'run_on':		'\n',
	},
	# Fastlier
	{'dim':			512,
	 'run_on':		'\t\n',
	},
	{'dim':			256,
	 'run_on':		' \t\n',
	 'reset_before':'\n',
	},
	{'dim':			128,
	 'run_on':		' \t\n',
	 'reset_before': '\t\n',
	},
	{'dim':			128,
	 'run_on':		' \t\n,.:;/!?()[]{}<>\\\'"*+-^_|#~&`@$%',
	 'reset_before': ' \t\n',
	},
	{'dim':			128,
	 'reset_before': ' \t\n,.:;/!?()[]{}<>\\\'"*+-^_|#~&`@$%',
	},
]
activation_function = Tanh()

w_noise_std = 0
i_dropout = 0

l1_reg = 0
l1_reg_weight = 0

step_rule = AdaDelta()

weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.)

# parameter saving freq (number of batches)
monitor_freq = 100
save_freq = 100

# used for sample generation and IRC mode
#sample_temperature = 0.7 #0.5
sample_temperature = 0.9 #0.5

# do we want to generate samples at times during training?
sample_len = 500
sample_freq = 100
sample_init = '\nalex\ttu crois?\n'

