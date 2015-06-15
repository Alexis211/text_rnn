import sys

import numpy

import theano
from theano import tensor

from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph

class GenText(SimpleExtension):
    def __init__(self, model, init_text, max_bytes, sample_temperature, **kwargs):
        super(GenText, self).__init__(**kwargs)
        
        self.init_text = init_text
        self.max_bytes = max_bytes

        out = model.out[:, -1, :] / numpy.float32(sample_temperature)
        prob = tensor.nnet.softmax(out)

        cg = ComputationGraph([prob])
        assert(len(cg.inputs) == 1)
        assert(cg.inputs[0].name == 'bytes')

        state_vars = [theano.shared(v[0:1, :].zeros_like().eval(), v.name+'-gen')
                                for v, _ in model.states]
        givens = [(v, x) for (v, _), x in zip(model.states, state_vars)]
        updates= [(x, upd) for x, (_, upd) in zip(state_vars, model.states)] 

        self.f = theano.function(inputs=cg.inputs, outputs=[prob],
                                 givens=givens, updates=updates)
        self.reset_states = theano.function(inputs=[], outputs=[],
                                            updates=[(v, v.zeros_like()) for v in state_vars])

    def do(self, which_callback, *args):

        print "Sample:"
        print "-------"

        self.reset_states()

        v = numpy.array([ord(i) for i in self.init_text],
                        dtype='int16')[None, :]
        prob, = self.f(v)

        sys.stdout.write(self.init_text)
        while v.shape[1] < self.max_bytes:
            prob = prob / 1.00001
            pred = numpy.random.multinomial(1, prob[0, :]).nonzero()[0][0]

            v = numpy.concatenate([v, pred[None, None]], axis=1)
            sys.stdout.write(chr(int(pred)))
            sys.stdout.flush()

            prob, = self.f(pred[None, None])
        print
        print "-------"
        print



