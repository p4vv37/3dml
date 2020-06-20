import numpy as np
from keras import backend
from keras.layers import Add, Conv2D
from keras.layers import Layer

import tensorflow as tf


# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs, **kwargs):
        # calculate square pixel values
        values = inputs ** 2.0
        # calculate the mean pixel values
        mean_values = backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = backend.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape


# mini-batch standard deviation layer
class MinibatchStdev(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs, **kwargs):
        # calculate the mean value for each pixel across channels
        mean = backend.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = backend.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = backend.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = backend.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)


class AddBiasLayer(Layer):
    def __init__(self, **kwargs):
        super(AddBiasLayer, self).__init__(**kwargs)
        self.bias = None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
        super(AddBiasLayer, self).build(input_shape)

    def call(self, input, **kwargs):
        if self.bias is not None:
            input = backend.bias_add(input, self.bias)
        return input

    def compute_output_shape(self, input_shape):
        return input_shape


# weighted sum output
class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        self.he_constant = None
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        # input_1 = Conv2D(inputs[0].shape[-1], (1, 1), padding='same', kernel_initializer=Ones())(inputs[1])
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class EqualizedConv2D(Conv2D):
    def __init__(self, filters, kernel, *args, **kwargs):
        self.scale = 1.0
        super(EqualizedConv2D, self).__init__(filters, kernel, *args, **kwargs)

    def build(self, input_shape):
        fan_in = np.prod(input_shape[1:])
        self.scale = np.sqrt(2/fan_in)
        return super(EqualizedConv2D, self).build(input_shape)

    def call(self, inputs):
        return super(EqualizedConv2D, self).call(inputs)
        # --- disabled rest equalized learning rate for now, does not work as expected.
        outputs = backend.conv2d(
            inputs,
            self.kernel*self.scale,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        outputs = backend.bias_add(
            outputs,
            self.bias,
            data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
