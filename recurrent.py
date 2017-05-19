import theano
import numpy as np
import theano.tensor as T

x = np.asarray([[0.1,0.2,0.4],[0.3,0.2,0.1],[0.2,0.2,0.2],[0.8,0.8,0.8]])
initial_h = theano.shared(np.asarray([0.0,0.0,0.0]),"init h")
h = theano.shared(np.asarray([0.0,0.0,0.0]),"h")
w = theano.shared(np.asarray([0.1,0.1,0.1]),"W")
w2 = theano.shared(np.asarray([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]]),"W2")
const = theano.shared(np.asarray([0.1,0.1,0.1]),"const")


inp = T.scalar("input")


update_h = { h : h + const }

in_vector = T.vector()


def h_next(inp,previous_h,w,w2):
    h = previous_h + T.dot(T.dot(inp,w),w2)
    y = h
    return h,y

[h_new,y], _ = theano.scan(h_next,
        sequences = in_vector,
        outputs_info = [initial_h, None], # first argument is to tell that the output will feed back to the next step, None is no feedback
        non_sequences = [w,w2])

out = y

calcuate = theano.function(inputs=[in_vector],outputs=out,updates=update_h)

zz = calcuate(x[0])
print zz

print h.eval()
