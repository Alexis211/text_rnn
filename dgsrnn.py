import theano
from theano import tensor
import numpy

from theano.tensor.shared_randomstreams import RandomStreams

from blocks.algorithms import Momentum, AdaDelta, RMSProp, Adam
from blocks.bricks import Activation, Tanh, Logistic, Softmax, Rectifier, Linear, MLP, Initializable, Identity
from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import IsotropicGaussian, Constant
from blocks.utils import shared_floatx_zeros

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, INITIAL_STATE, add_role
from blocks.graph import ComputationGraph, apply_noise, apply_dropout

rng = RandomStreams()

class TRectifier(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.switch(input_ > 1, input_, 0)

# An epoch will be composed of 'num_seqs' sequences of len 'seq_len'
# divided in chunks of lengh 'seq_div_size'
num_seqs = 10
seq_len = 1000
seq_div_size = 5

io_dim = 256

state_dim = 1024
activation = Tanh()
transition_hidden = [1024, 1024]
transition_hidden_activations = [Rectifier(), Rectifier()]

output_hidden = []
output_hidden_activations = []

weight_noise_std = 0.05

output_h_dropout = 0.0
drop_update = 0.0

l1_state = 0.00
l1_reset = 0.1

step_rule = 'momentum'
learning_rate = 0.001
momentum = 0.99


param_desc = '%s,t%s,o%s-n%s-d%s,%s-L1:%s,%s-%s' % (
                 repr(state_dim), repr(transition_hidden), repr(output_hidden),
                 repr(weight_noise_std),
                 repr(output_h_dropout), repr(drop_update),
                 repr(l1_state), repr(l1_reset),
                 step_rule
                ) 

save_freq = 5
on_irc = False

# parameters for sample generation
sample_len = 100
sample_temperature = 0.7 #0.5
sample_freq = 1

if step_rule == 'rmsprop':
    step_rule = RMSProp()
elif step_rule == 'adadelta':
    step_rule = AdaDelta()
elif step_rule == 'momentum':
    step_rule = Momentum(learning_rate=learning_rate, momentum=momentum)
elif step_rule == 'adam':
    step_rule = Adam()
else:
    assert(False)



class DGSRNN(BaseRecurrent, Initializable):
    def __init__(self, input_dim, state_dim, act, transition_h, tr_h_activations, **kwargs):
        super(DGSRNN, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.state_dim = state_dim

        logistic = Logistic()

        self.inter = MLP(dims=[input_dim + state_dim] + transition_h,
                         activations=tr_h_activations,
                         name='inter')
        self.reset = MLP(dims=[transition_h[-1], state_dim],
                         activations=[logistic],
                         name='reset')
        self.update = MLP(dims=[transition_h[-1], state_dim],
                          activations=[act],
                          name='update')

        self.children = [self.inter, self.reset, self.update, logistic, act] + tr_h_activations

        # init state
        self.params = [shared_floatx_zeros((state_dim,), name='init_state')]
        add_role(self.params[0], INITIAL_STATE)

    def get_dim(self, name):
        if name == 'state':
            return self.state_dim
        return super(GFGRU, self).get_dim(name)

    @recurrent(sequences=['inputs', 'drop_updates_mask'], states=['state'],
               outputs=['state', 'reset'], contexts=[])
    def apply(self, inputs=None, drop_updates_mask=None, state=None):
        inter_v = self.inter.apply(tensor.concatenate([inputs, state], axis=1))
        reset_v = self.reset.apply(inter_v)
        update_v = self.update.apply(inter_v)

        reset_v = reset_v * drop_updates_mask

        new_state = state * (1 - reset_v) + reset_v * update_v

        return new_state, reset_v

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return tensor.repeat(self.params[0][None, :],
                             repeats=batch_size,
                             axis=0)
            

class Model():
    def __init__(self):
        inp = tensor.lmatrix('bytes')

        in_onehot = tensor.eq(tensor.arange(io_dim, dtype='int16').reshape((1, 1, io_dim)),
                              inp[:, :, None])
        in_onehot.name = 'in_onehot'

        dgsrnn = DGSRNN(input_dim=io_dim,
                        state_dim=state_dim,
                        act=activation,
                        transition_h=transition_hidden,
                        tr_h_activations=transition_hidden_activations,
                        name='dgsrnn')

        prev_state = theano.shared(numpy.zeros((num_seqs, state_dim)).astype(theano.config.floatX),
                                   name='state')

        states, resets = dgsrnn.apply(inputs=in_onehot.dimshuffle(1, 0, 2),
                                      drop_updates_mask=rng.binomial(size=(inp.shape[1], inp.shape[0], state_dim),
                                                                     p=1-drop_update,
                                                                     dtype=theano.config.floatX),
                                      state=prev_state)
        states = states.dimshuffle(1, 0, 2)
        resets = resets.dimshuffle(1, 0, 2)

        self.states = [(prev_state, states[:, -1, :])]

        out_mlp = MLP(dims=[state_dim] + output_hidden + [io_dim],
                      activations=output_hidden_activations + [None],
                      name='output_mlp')
        states_sh = states.reshape((inp.shape[0]*inp.shape[1], state_dim))
        out = out_mlp.apply(states_sh).reshape((inp.shape[0], inp.shape[1], io_dim))


        # Do prediction and calculate cost
        pred = out.argmax(axis=2)

        cost = Softmax().categorical_cross_entropy(inp[:, 1:].flatten(),
                                                   out[:, :-1, :].reshape((inp.shape[0]*(inp.shape[1]-1),
                                                                           io_dim)))
        error_rate = tensor.neq(inp[:, 1:].flatten(), pred[:, :-1].flatten()).mean()

        # Initialize all bricks
        for brick in [dgsrnn, out_mlp]:
            brick.weights_init = IsotropicGaussian(0.001)
            brick.biases_init = Constant(0.0)
            brick.initialize()

        # Apply noise and dropout
        cg = ComputationGraph([cost, error_rate, states, resets])
        if weight_noise_std > 0:
            noise_vars = VariableFilter(roles=[WEIGHT])(cg)
            cg = apply_noise(cg, noise_vars, weight_noise_std)
        if output_h_dropout > 0:
            dv = VariableFilter(name='input_', bricks=out_mlp.linear_transformations)(cg)
            print "Output H dropout on", len(dv), "vars"
            cg = apply_dropout(cg, dv, output_h_dropout)
        [cost_reg, error_rate_reg, states, resets] = cg.outputs

        if l1_state > 0:
            cost_reg = cost_reg + l1_state * abs(states).mean()
        if l1_reset > 0:
            cost_reg = cost_reg + l1_reset * abs(resets).mean()

        self.cost = cost
        self.error_rate = error_rate
        self.cost_reg = cost_reg
        self.error_rate_reg = error_rate_reg
        self.out = out
        self.pred = pred


