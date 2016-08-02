import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy as np
import functools

rng = np.random.RandomState(10)

# num_maps: (# layers)
num_maps = (3, 5, 5, 3)

# input_img: (mini-batch size, # input maps, height, width)
input_img = T.tensor4(name='input')

# W: (# output maps, # input maps, kernel width, kernel height)
w_shp = (num_maps[1], num_maps[0], 9, 9)
w_bound = np.sqrt(w_shp[1] * w_shp[2] * w_shp[3])
W = theano.shared( np.asarray( rng.normal(
                loc = 0.0,
                scale = 1.0 / w_bound,
                size = w_shp),
            dtype = input_img.dtype),
        name = 'W')

# b: (# output maps)
b_shp = (num_maps[1],)
b = theano.shared( np.asarray( rng.normal(
                loc = 0.0,
                scale = 1.0,
                size = b_shp),
            dtype = input_img.dtype),
        name = 'b')

input_shp = T.shape(input_img)

# padded_img: (mini-batch size, # input maps,
#               height + kernel height, width + kernel width)
padded_img = T.zeros((input_shp[0], input_shp[1],
                     input_shp[2] + w_shp[2] - 1,
                     input_shp[3] + w_shp[3] - 1))
padded_img = T.set_subtensor(padded_img[:,:,
                    (w_shp[2] // 2):(input_shp[2] + w_shp[2] // 2),
                    (w_shp[3] // 2):(input_shp[3] + w_shp[3] // 2)],
                input_img)

# (mini-batch size, #output maps, height, width)
conv_out = conv2d(padded_img, W)

# (1, #output maps, 1, 1)
reshaped_bias = b.dimshuffle('x', 0, 'x', 'x')

# (mini-batch size, #output maps, height, width)
output_img = T.nnet.relu(conv_out + reshaped_bias, 0.25)

# (mini-batch size, #input maps, height, width) ->
# (mini-batch size, #output maps, height, width)
f = theano.function([input_img], output_img)
