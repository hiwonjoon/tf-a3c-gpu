import tensorflow as tf

class Conv2d(object) :
    def __init__(self,name,input_dim,output_dim,k_h=4,k_w=4,d_h=2,d_w=2,
                 stddev=0.02, data_format='NCHW') :
        with tf.variable_scope(name) :
            assert(data_format == 'NCHW' or data_format == 'NHWC')
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            if( data_format == 'NCHW' ) :
                self.strides = [1, 1, d_h, d_w]
            else :
                self.strides = [1, d_h, d_w, 1]
            self.data_format = data_format
    def __call__(self,input_var,name=None) :
        if( self.data_format =='NCHW' ) :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, self.w,
                                    use_cudnn_on_gpu=True,data_format='NCHW',
                                    strides=self.strides, padding='SAME'),
                        self.b,data_format='NCHW',name=name)
        else :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, self.w,data_format='NHWC',
                                    strides=self.strides, padding='SAME'),
                        self.b,data_format='NHWC',name=name)

class DilatedConv3D(object) :
    def __init__(self,name,input_dim,output_dim,k_t=2,k_h=3,k_w=3,d_t=2,d_h=1,d_w=1,
                 stddev=0.02, data_format='NDHWC') :
        with tf.variable_scope(name) :
            assert(data_format == 'NDHWC')
            self.w = tf.get_variable('w', [k_t, k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            self.strides = [1,1,1]
            self.dilates = [d_t, d_h, d_w]
    def __call__(self,input_var,name=None) :
        k_t,k_h,k_w,_,_ = self.w.get_shape().as_list()
        _t = tf.pad(input_var, [[0,0],[0,0],[k_h//2,k_h//2],[k_w//2,k_w//2],[0,0]], "SYMMETRIC")
        return tf.nn.bias_add(
                    tf.nn.convolution(_t, self.w,
                                      strides=self.strides, dilation_rate=self.dilates,
                                      padding='VALID'),
                    self.b,name=name)

class Linear(object) :
    def __init__(self,name,input_dim,output_dim,stddev=0.02) :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w',[input_dim, output_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim],
                                initializer=tf.constant_initializer(0.0))

    def __call__(self,input_var,name=None) :
        if( input_var.get_shape().dims > 2 ) :
            return tf.matmul(tf.reshape(input_var,[tf.shape(input_var)[0],-1]),self.w) + self.b
        else :
            return tf.matmul(input_var,self.w)+self.b

class SymPadConv2d(object): #Resize and Convolution(upsacle by 2)
    def __init__(self,name,input_dim,output_dim,
                 k_h=4,k_w=4,d_h=1,d_w=1,stddev=0.02) :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
        self.strides = [1, 1, d_h, d_w]
        self.padding = [ [0,0],[k_h//2,k_h//2],[k_w//2,k_w//2],[0,0] ]

    def __call__(self,input_var,name=None):
        #_t = tf.image.resize_nearest_neighbor(input_var, [self.output_shape[2], self.output_shape[3]])
        _t = tf.pad(input_var,self.padding, mode='SYMMETRIC')
        return tf.nn.bias_add(
                    tf.nn.conv2d(_t, self.w,
                                 data_format='NHWC', #we can't use cudnn due to resize method...
                                 strides=self.strides, padding="VALID"),
                    self.b,data_format='NHWC',name=name)

class TransposedConv2d(object):
    def __init__(self,name,input_dim,output_shape,
                 k_h=4,k_w=4,d_h=2,d_w=2,stddev=0.02) :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w', [k_h, k_w, output_shape[1], input_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_shape[1]], initializer=tf.constant_initializer(0.0))
        self.output_shape = output_shape
        self.strides = [1, 1, d_h, d_w]

    def __call__(self,input_var,name=None):
        return tf.nn.bias_add(
            tf.nn.conv2d_transpose(input_var,self.w,output_shape=self.output_shape,
                                   data_format='NCHW',
                                   strides=self.strides,padding='SAME'),
            self.b,data_format='NCHW',name=name)

class InstanceNorm():
    def __init__(self,name,format='NCHW',epsilon=1e-5) :
        assert(format=='NCHW' or format=='NHWC')
        self.axis = [2,3] if format == 'NCHW' else [1,2]

        self.epsilon = epsilon
        self.name = name

    def __call__(self,input_var) :
        mean, var = tf.nn.moments(input_var, self.axis, keep_dims=True)
        return (input_var - mean) / tf.sqrt(var+self.epsilon)

class BatchNorm(object):
    def __init__(self,name,scope,fused,epsilon=1e-5,momentum=0.9) :
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name
        self.fused = fused

    def __call__(self,input_var,is_training=True,reuse=False) :
        return tf.contrib.layers.batch_norm(input_var,
                                            data_format='NCHW',
                                            fused=self.fused,
                                            decay=self.momentum,
                                            updates_collections=None, #You should add operations for batch norm in control dependencies.
                                            epsilon=self.epsilon,
                                            scale=False,
                                            center=False,
                                            is_training=is_training,
                                            reuse=reuse,
                                            scope=self.name)

class Lrelu(object):
    def __init__(self,leak=0.2,name='lrelu') :
        self.leak = leak
        self.name = name
    def __call__(self, x) :
        return tf.maximum(x, self.leak*x, name=self.name)

class ResidualBlock() :
    def __init__(self,name,filters,filter_size=3,non_linearity=Lrelu,normal_method=InstanceNorm) :
        self.conv_1 = Conv2d(name+'_1',filters,filters,filter_size,filter_size,1,1)
        self.normal = normal_method(name+'_norm')
        self.nl = non_linearity()
        self.conv_2 = Conv2d(name+'_2',filters,filters,filter_size,filter_size,1,1)
    def __call__(self,input_var) :
        _t = self.conv_1(input_var)
        _t = self.normal(_t)
        _t = self.nl(_t)
        _t = self.conv_2(_t)
        return input_var + _t

