import theano
import numpy as np
import theano.tensor as T
from collections import OrderedDict

"""
Batch First Dimension is timestep.
x = Input Batch
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
    NW = [NP_id2vec(i,10) for i in range(10)]
    param["Wemb"] = np.array(NW)
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

def NP_layer_scan_param(param):
    Nh = np.zeros(VOCAB_SIZE,dtype="float32")
    param["scan_H"] = Nh
    return param

def Layer_ScanSum(TparamD,Temb_batch):

    TscanH = TparamD["scan_H"]
    Tinitial_h = T.zeros_like(Temb_batch[0])

    def _step(x,h):
        return x+h

    output_sum , _ = theano.scan(_step,
                        sequences=Temb_batch,
                        outputs_info=[Tinitial_h]
                        )

    return output_sum

def Reshape_Batch(NPx):
    NPx = NPx.transpose()
    return NPx

def TReshape_InputBatch(Tx):
    Tx = Tx.transpose()
    return Tx

def build_model():
    paramD = OrderedDict()
    paramD = NP_lookup_layer_param(paramD)
    paramD = NP_layer_scan_param(paramD)

    TparamD = init_theano_params(paramD)

    x = T.imatrix("int32")
    Wembs = Layer_Wemb(TparamD,x.transpose())

    out1 = Wembs
    out2 = Layer_ScanSum(TparamD,Wembs)

    f1 = theano.function([x],out1)
    f2 = theano.function([x],out2)

    #np.array([seq0,seq1])
    y = f1(np.array([[0,1,2,3],[0,1,2,3]],dtype="int32"))
    z = f2(np.array([[0,1,2,3],[0,1,2,3]],dtype="int32"))
    print "Y"
    print y
    print "Z"
    print z


if __name__ == "__main__":
    x = [[0,1,2],[0,1,2]]
    build_model()
