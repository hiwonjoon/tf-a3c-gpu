import numpy as np
from collections import deque

class HistoryBuffer():
    def __init__(self,preprocess_fn,image_shape,frames_for_state) :
        self.buf = deque(maxlen=frames_for_state)
        self.preprocess_fn = preprocess_fn
        self.image_shape = image_shape
        self.clear()

    def clear(self) :
        for i in range(self.buf.maxlen):
            self.buf.append(np.zeros(self.image_shape,np.float32))

    def add(self,o) :
        #assert( list(o.shape) == self.image_shape ),'%s, %s'%(o.shape,self.image_shape)
        self.buf.append(self.preprocess_fn(o))
        state = np.concatenate([img for img in self.buf], axis=2)
        return state
