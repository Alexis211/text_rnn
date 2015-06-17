from irc.client import SimpleIRCClient

import logging

import numpy

import theano
from theano import tensor

from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph

logging.basicConfig(level='INFO')
logger = logging.getLogger('irc_ext')

class IRCClient(SimpleIRCClient):
    def __init__(self, chans, nick):
        super(IRCClient, self).__init__()

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
            pred = numpy.random.multinomial(1, prob[0, :]).nonzero()[0][0]

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

        if chan in self.chans:
            pred_f, _ = self.chans[chan]
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
            pass

        if rep != None:
            self.privmsg(chan, rep)

class IRCClientExt(SimpleExtension):
    def __init__(self, model, sample_temperature, server, port, nick, channels, **kwargs):
        super(IRCClientExt, self).__init__(**kwargs)

        # model output
        out = model.out[:, -1, :] / numpy.float32(sample_temperature)
        prob = tensor.nnet.softmax(out)

        cg = ComputationGraph([prob])
        assert(len(cg.inputs) == 1)
        assert(cg.inputs[0].name == 'bytes')

        # channel functions & state
        chfun = {}
        for ch in channels + ['']:
            state_vars = [theano.shared(v[0:1, :].zeros_like().eval(), v.name+'-'+ch)
                                    for v, _ in model.states]
            givens = [(v, x) for (v, _), x in zip(model.states, state_vars)]
            updates= [(x, upd) for x, (_, upd) in zip(state_vars, model.states)] 

            pred = theano.function(inputs=cg.inputs, outputs=[prob],
                                   givens=givens, updates=updates)
            reset_states = theano.function(inputs=[], outputs=[],
                                           updates=[(v, v.zeros_like()) for v in state_vars])
            chfun[ch] = (pred, reset_states)

        self.irc = IRCClient(chfun, nick)
        self.irc.connect(server, port, nick)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['irc']
        return state

    def __setstate__(self, state):
        irc = self.irc
        self.__dict__.update(state)
        self.irc = irc

    def do(self, which_callback, *args):
        logger.info('Polling...')
        self.irc.reactor.process_once()


