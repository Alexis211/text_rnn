import theano
from theano import tensor
import numpy

from theano.tensor.shared_randomstreams import RandomStreams

from blocks.algorithms import Momentum, AdaDelta, RMSProp
from blocks.bricks import Tanh, Softmax, Linear, MLP, Initializable
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM, BaseRecurrent, recurrent
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

rng = RandomStreams()

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 50
seq_len = 2000
seq_div_size = 100

io_dim = 256

# Model structure
hidden_dims = [512, 512, 512, 512, 512]
activation_function = Tanh()

cond_cert = [0.5, 0.5, 0.5, 0.5]
block_prob = [0.1, 0.1, 0.1, 0.1]

# Regularization
w_noise_std = 0.02

# Step rule
step_rule = 'adadelta'
learning_rate = 0.1
momentum = 0.9


param_desc = '%s(x%sp%s)-n%s-%dx%d(%d)-%s' % (
                 repr(hidden_dims), repr(cond_cert), repr(block_prob),
                 repr(w_noise_std),
                 num_seqs, seq_len, seq_div_size,
                 step_rule
                ) 

save_freq = 5
on_irc = False

# parameters for sample generation
sample_len = 200
sample_temperature = 0.7 #0.5
sample_freq = 1

if step_rule == 'rmsprop':
    step_rule = RMSProp()
elif step_rule == 'adadelta':
    step_rule = AdaDelta()
elif step_rule == 'momentum':
    step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
else:
    assert(False)

class CCHLSTM(BaseRecurrent, Initializable):
    def __init__(self, io_dim, hidden_dims, cond_cert, activation=None, **kwargs):
        super(CCHLSTM, self).__init__(**kwargs)

        self.cond_cert = cond_cert

        self.io_dim = io_dim
        self.hidden_dims = hidden_dims

        self.children = []
        self.layers = []

        self.softmax = Softmax()
        self.children.append(self.softmax)

        for i, d in enumerate(hidden_dims):
            i0 = LookupTable(length=io_dim,
                             dim=4*d,
                             name='i0-%d'%i)
            self.children.append(i0)

            if i > 0:
                i1 = Linear(input_dim=hidden_dims[i-1],
                            output_dim=4*d,
                            name='i1-%d'%i)
                self.children.append(i1)
            else:
                i1 = None

            lstm = LSTM(dim=d, activation=activation,
                        name='LSTM-%d'%i)
            self.children.append(lstm)

            o = Linear(input_dim=d,
                       output_dim=io_dim,
                       name='o-%d'%i)
            self.children.append(o)

            self.layers.append((i0, i1, lstm, o))


    @recurrent(contexts=[])
    def apply(self, inputs, **kwargs):

        l0i, _, l0l, l0o = self.layers[0]
        l0iv = l0i.apply(inputs)
        new_states0, new_cells0 = l0l.apply(states=kwargs['states0'],
                                            cells=kwargs['cells0'],
                                            inputs=l0iv,
                                            iterate=False)
        l0ov = l0o.apply(new_states0)

        pos = l0ov
        ps = new_states0

        passnext = tensor.ones((inputs.shape[0],))
        out_sc = [new_states0, new_cells0, passnext]

        for i, (cch, (i0, i1, l, o)) in enumerate(zip(self.cond_cert, self.layers[1:])):
            pop = self.softmax.apply(pos)
            best = pop.max(axis=1)
            passnext = passnext * tensor.le(best, cch) * kwargs['pass%d'%i]

            i0v = i0.apply(inputs)
            i1v = i1.apply(ps)

            prev_states = kwargs['states%d'%i]
            prev_cells = kwargs['cells%d'%i]
            new_states, new_cells = l.apply(inputs=i0v + i1v,
                                            states=prev_states,
                                            cells=prev_cells,
                                            iterate=False)
            new_states = tensor.switch(passnext[:, None], new_states, prev_states)
            new_cells = tensor.switch(passnext[:, None], new_cells, prev_cells)
            out_sc += [new_states, new_cells, passnext]

            ov = o.apply(new_states)
            pos = tensor.switch(passnext[:, None], pos + ov, pos)
            ps = new_states

        return [pos] + out_sc

    def get_dim(self, name):
        dims = {'pred': self.io_dim}
        for i, d in enumerate(self.hidden_dims):
            dims['states%d'%i] = dims['cells%d'%i] = d
        if name in dims:
            return dims[name]
        return super(CCHLSTM, self).get_dim(name)

    @apply.property('sequences')
    def apply_sequences(self):
        return ['inputs'] + ['pass%d'%i for i in range(len(self.hidden_dims)-1)]

    @apply.property('states')
    def apply_states(self):
        ret = []
        for i in range(len(self.hidden_dims)):
            ret += ['states%d'%i, 'cells%d'%i]
        return ret

    @apply.property('outputs')
    def apply_outputs(self):
        ret = ['pred']
        for i in range(len(self.hidden_dims)):
            ret += ['states%d'%i, 'cells%d'%i, 'active%d'%i]
        return ret


class Model():
    def __init__(self):
        inp = tensor.lmatrix('bytes')

        # Make state vars
        state_vars = {}
        for i, d in enumerate(hidden_dims):
            state_vars['states%d'%i] = theano.shared(numpy.zeros((num_seqs, d))
                                                        .astype(theano.config.floatX),
                                                     name='states%d'%i)
            state_vars['cells%d'%i] = theano.shared(numpy.zeros((num_seqs, d))
                                                        .astype(theano.config.floatX),
                                                    name='cells%d'%i)
        # Construct brick
        cchlstm = CCHLSTM(io_dim=io_dim,
                          hidden_dims=hidden_dims,
                          cond_cert=cond_cert,
                          activation=activation_function)

        # Random pass
        passdict = {}
        for i, p in enumerate(block_prob):
            passdict['pass%d'%i] = rng.binomial(size=(inp.shape[1], inp.shape[0]), p=1-p)

        # Apply it
        outs = cchlstm.apply(inputs=inp.dimshuffle(1, 0),
                             **dict(state_vars.items() + passdict.items()))
        states = []
        active_prop = []
        for i in range(len(hidden_dims)):
            states.append((state_vars['states%d'%i], outs[3*i+1][-1, :, :]))
            states.append((state_vars['cells%d'%i], outs[3*i+2][-1, :, :]))
            active_prop.append(outs[3*i+3].mean())
            active_prop[-1].name = 'active_prop_%d'%i

        out = outs[0].dimshuffle(1, 0, 2)

        # Do prediction and calculate cost
        pred = out.argmax(axis=2)

        cost = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           io_dim)))
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).mean()

        # Initialize all bricks
        for brick in [cchlstm]:
            brick.weights_init = IsotropicGaussian(0.1)
            brick.biases_init = Constant(0.)
            brick.initialize()

        # Apply noise and dropoutvars
        cg = ComputationGraph([cost, error_rate])
        if w_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, w_noise_std)
        [cost_reg, error_rate_reg] = cg.outputs

        self.sgd_cost = cost_reg
        self.monitor_vars = [[cost, cost_reg],
                             [error_rate, error_rate_reg],
                             active_prop]

        cost.name = 'cost'
        cost_reg.name = 'cost_reg'
        error_rate.name = 'error_rate'
        error_rate_reg.name = 'error_rate_reg'

        self.out = out
        self.pred = pred

        self.states = states

