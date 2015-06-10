#!/usr/bin/env python

import logging
import numpy
import sys
import importlib

from contextlib import closing

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.serialization import load_parameter_values, secure_dump, BRICK_DELIMITER
from blocks.extensions import Printing, SimpleExtension
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extras.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint, Load
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

import datastream
# from apply_model import Apply

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('%s' % model_name)


class GenText(SimpleExtension):
    def __init__(self, model, init_text, max_bytes, **kwargs):
        self.init_text = init_text
        self.max_bytes = max_bytes


        out = model.out[:, -1, :] / numpy.float32(config.sample_temperature)
        prob = tensor.nnet.softmax(out)

        cg = ComputationGraph([prob])
        assert(len(cg.inputs) == 1)
        assert(cg.inputs[0].name == 'bytes')
        self.f = theano.function(inputs=cg.inputs, outputs=[prob])

        super(GenText, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        v = numpy.array([ord(i) for i in self.init_text],
                        dtype='int16')[None, :].repeat(axis=0, repeats=config.num_seqs)

        while v.shape[1] < self.max_bytes:
            prob, = self.f(v)
            prob = prob / 1.00001
            pred = numpy.zeros((prob.shape[0],), dtype='int16')
            for i in range(prob.shape[0]):
                pred[i] = numpy.random.multinomial(1, prob[i, :]).nonzero()[0][0]
            v = numpy.concatenate([v, pred[:, None]], axis=1)

        for i in range(v.shape[0]):
            print "Sample:", ''.join([chr(int(v[i, j])) for j in range(v.shape[1])])

def train_model(m, train_stream, dump_path=None):

    # Define the model
    model = Model(m.cost)

    cg = ComputationGraph(m.cost_reg)
    algorithm = GradientDescent(cost=m.cost_reg,
                                step_rule=config.step_rule,
                                params=cg.parameters)

    algorithm.add_updates(m.updates)

    # Load the parameters from a dumped model
    if dump_path is not None:
        try:
            logger.info('Loading parameters...')
            with closing(numpy.load(dump_path)) as source:
                param_values = {'/' + name.replace(BRICK_DELIMITER, '/'): source[name]
                                    for name in source.keys()
                                    if name != 'pkl' and not 'None' in name}
            model.set_param_values(param_values)
        except IOError:
            pass

    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            Checkpoint(path=dump_path,
                       after_epoch=False, every_n_epochs=config.save_freq),

            TrainingDataMonitoring(
                [m.cost_reg, m.error_rate_reg, m.cost, m.error_rate],
                prefix='train', every_n_epochs=1),
            Printing(every_n_epochs=1, after_epoch=False),
            Plot(document='tr_'+model_name+'_'+config.param_desc,
                 channels=[['train_cost', 'train_cost_reg'],
                           ['train_error_rate', 'train_error_rate_reg']],
                 server_url='http://eos21:4201/',
                 every_n_epochs=1, after_epoch=False),

            GenText(m, ' ', config.sample_len, every_n_epochs=1, after_epoch=False)
        ]
    )
    main_loop.run()


if __name__ == "__main__":
    # Build datastream
    train_stream = datastream.setup_datastream('data/logcompil.txt',
                                               config.num_seqs,
                                               config.seq_len,
                                               config.seq_div_size)

    # Build model
    m = config.Model()
    m.cost.name = 'cost'
    m.cost_reg.name = 'cost_reg'
    m.error_rate.name = 'error_rate'
    m.error_rate_reg.name = 'error_rate_reg'
    m.pred.name = 'pred'

    # Train the model
    saveloc = 'model_data/%s-%s' % (model_name, config.param_desc)
    train_model(m, train_stream, dump_path=saveloc)

