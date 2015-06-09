import theano
from theano import tensor

from blocks.algorithms import Momentum, AdaDelta
from blocks.bricks import Tanh, Softmax, Linear, MLP
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise

chars_per_seq = 100
seqs_per_epoch = 1

io_dim = 256

hidden_dims = [200, 500]
activation_function = Tanh()

w_noise_std = 0.01

step_rule = AdaDelta()

pt_freq = 1

param_desc = '' # todo

class Model():
    def __init__(self):
        inp = tensor.lvector('bytes')

        in_onehot = tensor.eq(tensor.arange(io_dim, dtype='int16').reshape((1, io_dim)),
                              inp[:, None])

        dims = [io_dim] + hidden_dims
        prev = in_onehot[None, :, :]
        bricks = []
        for i in xrange(1, len(dims)):
            linear = Linear(input_dim=dims[i-1], output_dim=4*dims[i],
                            name="lstm_in_%d"%i)
            lstm = LSTM(dim=dims[i], activation=activation_function,
                        name="lstm_rec_%d"%i)
            prev = lstm.apply(linear.apply(prev))[0]
            bricks = bricks + [linear, lstm]

        top_linear = MLP(dims=[hidden_dims[-1], io_dim],
                         activations=[Softmax()],
                         name="pred_mlp")
        bricks.append(top_linear)

        out = top_linear.apply(prev.reshape((inp.shape[0], hidden_dims[-1])))

        pred = out.argmax(axis=1)

        cost = Softmax().categorical_cross_entropy(inp[:-1], out[1:])
        error_rate = tensor.neq(inp[:-1], pred[1:]).mean()

        # Initialize
        for brick in bricks:
            brick.weights_init = IsotropicGaussian(0.1)
            brick.biases_init = Constant(0.)
            brick.initialize()

        # apply noise
        cg = ComputationGraph([cost, error_rate])
        noise_vars = VariableFilter(roles=[WEIGHT])(cg)
        cg = apply_noise(cg, noise_vars, w_noise_std)
        [cost_reg, error_rate_reg] = cg.outputs

        self.cost = cost
        self.error_rate = error_rate
        self.cost_reg = cost_reg
        self.error_rate_reg = error_rate_reg
        self.pred = pred

