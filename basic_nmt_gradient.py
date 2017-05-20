import theano
from theano import config
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
    return u.astype(config.floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(config.floatX)

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
    Nh = np.zeros(VOCAB_SIZE,dtype=config.floatX)
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

    return output_sum[-1] #output only the last state

def NP_layer_output_param(param):
    NWout = norm_weight(VOCAB_SIZE,1)
    param["Wout"] = NWout
    return param

def Layer_Output(TparamD,x_batch):
    TWout = TparamD["Wout"]
    Tout = T.nnet.sigmoid(T.dot(x_batch,TWout))
    return Tout

def Reshape_Batch(NPx):
    NPx = NPx.transpose()
    return NPx

def TReshape_InputBatch(Tx):
    Tx = Tx.transpose()
    return Tx

def Layer_Loss(Tparam,output_batch,desire_batch):
    nsamples = output_batch.shape[0]
    out = output_batch.flatten()
    out = out.reshape((nsamples,))
    loss = (desire_batch - out) ** 2
    return T.sum(loss)

#This function defines sgd with some design pattern techniques.
def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer without mask -- adapted from deeplearning.net

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def build_model():
    #Init Parameters

    print config.floatX

    lrate = 0.01
    paramD = OrderedDict()
    paramD = NP_lookup_layer_param(paramD)
    paramD = NP_layer_scan_param(paramD)
    paramD = NP_layer_output_param(paramD)

    #Convert Parameters to Tensor
    TparamD = init_theano_params(paramD)

    #Create Computation Graph
    x = T.imatrix("int32")

    Wembs = Layer_Wemb(TparamD,x.transpose())

    out1 = Wembs
    out2 = Layer_ScanSum(TparamD,Wembs)

    #Define predict function
    f1 = theano.function([x],out1)
    f2 = theano.function([x],out2)

    #np.array([seq0,seq1])
    y = f1(np.array([[0,1,2,3],[0,1,2,3]],dtype="int32"))
    z = f2(np.array([[0,1,2,3],[0,1,2,3]],dtype="int32"))

    out3 = Layer_Output(TparamD,out2)
    f3 = theano.function([x],out3)
    out = f3(np.array([[0,1,2,3],[0,1,2,3]],dtype="int32"))

    #Defind Training Process
    desire = T.dvector(config.floatX)
    cost = Layer_Loss(TparamD,out3,desire) #loss is cost

    f_cost = theano.function([x,desire],cost)

    UpdateParams = {"Wemb" : TparamD["Wemb"],
                    "Wout" : TparamD["Wout"]}

    grads = T.grad(cost, wrt=list(UpdateParams.values()))
    f_grad = theano.function([x, desire], grads, name='f_grad')

    lr = T.scalar(name='lr')
    optimizer = adadelta
    f_grad_shared, f_update = optimizer(lr, UpdateParams, grads,
                                        x, desire, cost)

    nepoch = 1000
    for i in range(0,nepoch):
        print i
        loss = f_grad_shared(np.array([[0,1,2,3],[0,1,2,3]],dtype="int32"),np.array([1.,1.],dtype=config.floatX))
        print loss
        f_update(lrate)


if __name__ == "__main__":
    x = [[0,1,2],[0,1,2]]
    build_model()
