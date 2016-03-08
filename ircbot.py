#!/usr/bin/env python2

import logging
import sys
import importlib

from irc.client import SimpleIRCClient

import numpy
import theano
from theano import tensor

from blocks.model import Model

import datastream
from paramsaveload import SaveLoadParams
from blocks.graph import ComputationGraph

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class IRCClient(SimpleIRCClient):
    def __init__(self, model, sample_temperature, server, port, nick, channels, saveload):
        super(IRCClient, self).__init__()

        out = model.out[:, -1, :] / numpy.float32(sample_temperature)
        prob = tensor.nnet.softmax(out)

        cg = ComputationGraph([prob])
        assert(len(cg.inputs) == 1)
        assert(cg.inputs[0].name == 'bytes')

        # channel functions & state
        chfun = {}
        for ch in channels + ['']:
            logger.info("Building theano function for channel '%s'"%ch)
            state_vars = [theano.shared(v[0:1, :].zeros_like().eval(), v.name+'-'+ch)
                                    for v, _ in model.states]
            givens = [(v, x) for (v, _), x in zip(model.states, state_vars)]
            updates= [(x, upd) for x, (_, upd) in zip(state_vars, model.states)] 

            pred = theano.function(inputs=cg.inputs, outputs=[prob],
                                   givens=givens, updates=updates)
            reset_states = theano.function(inputs=[], outputs=[],
                                           updates=[(v, v.zeros_like()) for v in state_vars])
            chfun[ch] = (pred, reset_states)

        self.saveload = saveload

        self.chfuns = chfun

        self.chans = chans
        self.nick = nick
        self.server = None


    def on_welcome(self, server, ev):
        logger.info("Welcomed to " + repr(server))
        for ch in self.chans:
            if ch != '' and ch[0] == '#':
                server.join(ch)

    def on_join(self, server, ev):
        self.server = server

    def str2data(self, s):
        return numpy.array([ord(x) for x in s], dtype='int16')[None, :]

    def pred_until(self, pred_f, prob, delim='\n'):
        s = ''
        while True:
            prob = prob / 1.00001
            pred = numpy.random.multinomial(1, prob[0, :]).nonzero()[0][0].astype('int16')

            s = s + chr(int(pred))

            prob, = pred_f(pred[None, None])

            if s[-1] == delim:
                break
        return s[:-1]

    def privmsg(self, chan, msg):
        if len(msg) > 500:
            msg = 'blip bloup'
        logger.info("%s >> %s" % (chan, msg))
        self.server.privmsg(chan, msg.decode('utf-8', 'ignore'))

    def on_pubmsg(self, server, ev):
        chan = ev.target.encode('utf-8')
        nick = ev.source.split('!')[0].encode('utf-8')
        msg = ev.arguments[0].encode('utf-8')

        logger.info("%s <%s> %s" % (chan, nick, msg))

        s0 = nick+'\t'+msg

        rep = None

        if chan in self.chfuns:
            pred_f, _ = self.chfuns[chan]
            if s0[-2:] == '^I':
                prob, = pred_f(self.str2data(s0[:-2]))
                rep = s0[:-2] + self.pred_until(pred_f, prob)
                rep = rep.split('\t', 1)[-1]
            else:
                # feed phrase to bot
                prob, = pred_f(self.str2data(s0+'\n'))
                if self.nick in msg:
                    self.pred_until(pred_f, prob, '\t') 
                    prob, = pred_f(self.str2data(nick+': '))
                    rep = nick + ': ' + self.pred_until(pred_f, prob)
        else:
            logger.warn('Recieved message on unknown channel: %s'%chan)
        
        if rep != None:
            self.privmsg(chan, rep)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s [options] config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[-1]
    config = importlib.import_module('.%s' % model_name, 'config')

    # Build model
    logger.info('Building model...')
    m = config.Model(config)

    # Define the computation graph && load parameters
    logger.info('Building computation graph...')
    dump_path = 'params/%s-use_on_irc.pkl' % model_name
    saveload = SaveLoadParams(path=dump_path,
                              model=Model(m.sgd_cost))
    saveload.do_load()

    # Build IRC client
    server = 'clipper.ens.fr'
    port = 6667
    nick = 'frigo'
    chans = ['#frigotest', '#courssysteme']

    irc = IRCClient(model=m,
                    sample_temperature=config.sample_temperature,
                    server=server,
                    port=port,
                    nick=nick,
                    channels=chans,
                    saveload=saveload)
    irc.connect(server, port, nick)
    irc.reactor.process_forever()


#  vim: set sts=4 ts=4 sw=4 tw=0 et :
