import theano
import numpy as np
import theano.tensor as T
from collections import OrderedDict

"""
Binary Classification based on RNN
x = Input [[id,id,id,id],[id,id,id,id],[id,id,id,id]]
y = Output [[p1,p2],[p1,p2],[p1,p2],[p1,p2]]
"""

VOCAB_SIZE = 10
HIDDEN_SIZE = 10
NUM_CLASS = 2

# initialize Theano shared variables according to the initial parameters
def init_theano_params(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')


def NP_id2vec(id,VOCAB_SIZE):
    vec = np.zeros((VOCAB_SIZE))
    vec[id] = 1.0
    return vec

def NP_lookup_layer_param(param):
    NW = norm_weight(VOCAB_SIZE,HIDDEN_SIZE)
    param["Wemb"] = NW
    return param

def NP_ff_layer_param(param):
    NW = norm_weight(HIDEN_SIZE,NUM_CLASS)

def Layer_Wemb(TparamD,ids):
    TW = TparamD["Wemb"]
    nsamples = ids.shape[0]
    ntimesteps = ids.shape[1]

    print nsamples
    print ntimesteps

    emb = TW[ids.flatten()]
    emb = emb.reshape((nsamples,ntimesteps,-1)) #This is not normal case for RNN, for RNN timestep should on dim 0.
    return emb

def build_model():
    paramD = OrderedDict()
    paramD = NP_lookup_layer_param(paramD)
    TparamD = init_theano_params(paramD)

    x = T.imatrix("int32")
    Wembs = Layer_Wemb(TparamD,x)
    out = T.sum(Wembs,axis=1)
    f = theano.function([x],out)

    #np.array([seq0,seq1])
    y = f(np.array([[0,1,2],[0,1,2]],dtype="int32"))

    print y


if __name__ == "__main__":
    print NP_id2vec(4,10)
    build_model()
