# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library
# (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
import numpy as np


class StackedHourglassModel(Network):

    def setup(self, is_training, n_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
        '''
        # 384x384-->192x192
        (self.feed('data')
             .pad([[0,0], [2,2], [2,2], [0,0]], name='pad_1')
             .conv(7, 7, 64, 2, 2, biased=True, relu=False, name='conv1_')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='conv1__')
             .relu(name='conv1')
         # .max_pool(3, 3, 2, 2, name='pool1')
         # .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='res2a_branch1')
         # .tf_batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1')
         )

        (self.feed('conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res1_batch1')
             .relu(name='res1_relu1')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res1_batch2')
             .relu(name='res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 64, 1, 1, biased=True, relu=False, name='res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res1_batch3')
             .relu(name='res1_relu3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res1_conv3'))

        (self.feed('conv1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res1_skip'))
        # 192x192-->96x96
        (self.feed('res1_conv3',
                   'res1_skip')
         .add(name='Res1')
         .max_pool(2, 2, 2, 2, name='Res1_pool1'))


# id:004
        (self.feed('Res1_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res2_batch1')
             .relu(name='res2_relu1')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res2_batch2')
             .relu(name='res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 64, 1, 1, biased=True, relu=False, name='res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res2_batch3')
             .relu(name='res2_relu3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res2_conv3'))

        (self.feed('Res1_pool1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res2_skip'))

        (self.feed('res2_conv3',
                   'res2_skip')
         .add(name='Res2'))
# id:005
        (self.feed('Res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res3_batch1')
             .relu(name='res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res3_batch2')
             .relu(name='res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='res3_batch3')
             .relu(name='res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='res3_conv3'))

        (self.feed('Res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='res3_skip'))

        (self.feed('res3_conv3',
                   'res3_skip')
         .add(name='Res3'))


#######################################  Hourglass1  #####################
# res1
        (self.feed('Res3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res1_batch1')
             .relu(name='HG1_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res1_batch2')
             .relu(name='HG1_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res1_batch3')
             .relu(name='HG1_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res1_conv3'))

        (self.feed('Res3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res1_skip'))

        (self.feed('HG1_res1_conv3',
                   'HG1_res1_skip')
         .add(name='HG1_res1'))

#   pool1
        (self.feed('Res3')
             .max_pool(2, 2, 2, 2, name='HG1_pool1'))


# res2
        (self.feed('HG1_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res2_batch1')
             .relu(name='HG1_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res2_batch2')
             .relu(name='HG1_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res2_batch3')
             .relu(name='HG1_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res2_conv3'))

        (self.feed('HG1_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res2_skip'))

        (self.feed('HG1_res2_conv3',
                   'HG1_res2_skip')
         .add(name='HG1_res2'))
# id:009 max-pooling
        # (self.feed('HG1_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG1_pool2'))


# res3
        (self.feed('HG1_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res3_batch1')
             .relu(name='HG1_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res3_batch2')
             .relu(name='HG1_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res3_batch3')
             .relu(name='HG1_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res3_conv3'))

        (self.feed('HG1_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res3_skip'))

        (self.feed('HG1_res3_conv3',
                   'HG1_res3_skip')
         .add(name='HG1_res3'))


# pool2
        (self.feed('HG1_res2')
             .max_pool(2, 2, 2, 2, name='HG1_pool2'))


# res4
        (self.feed('HG1_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res4_batch1')
             .relu(name='HG1_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res4_batch2')
             .relu(name='HG1_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res4_batch3')
             .relu(name='HG1_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res4_conv3'))

        (self.feed('HG1_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res4_skip'))

        (self.feed('HG1_res4_conv3',
                   'HG1_res4_skip')
         .add(name='HG1_res4'))
# id:013 max-pooling
        # (self.feed('HG1_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG1_pool4'))


# res5
        (self.feed('HG1_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res5_batch1')
             .relu(name='HG1_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res5_batch2')
             .relu(name='HG1_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res5_batch3')
             .relu(name='HG1_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res5_conv3'))

        (self.feed('HG1_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res5_skip'))

        (self.feed('HG1_res5_conv3',
                   'HG1_res5_skip')
         .add(name='HG1_res5'))


# pool3
        (self.feed('HG1_res4')
             .max_pool(2, 2, 2, 2, name='HG1_pool3'))


# res6
        (self.feed('HG1_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res6_batch1')
             .relu(name='HG1_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res6_batch2')
             .relu(name='HG1_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res6_batch3')
             .relu(name='HG1_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res6_conv3'))

        (self.feed('HG1_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res6_skip'))

        (self.feed('HG1_res6_conv3',
                   'HG1_res6_skip')
         .add(name='HG1_res6'))

# res7
        (self.feed('HG1_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res7_batch1')
             .relu(name='HG1_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res7_batch2')
             .relu(name='HG1_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res7_batch3')
             .relu(name='HG1_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res7_conv3'))

        (self.feed('HG1_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res7_skip'))

        (self.feed('HG1_res7_conv3',
                   'HG1_res7_skip')
         .add(name='HG1_res7'))


# pool4
        (self.feed('HG1_res6')
             .max_pool(2, 2, 2, 2, name='HG1_pool4'))

# res8
        (self.feed('HG1_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res8_batch1')
             .relu(name='HG1_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res8_batch2')
             .relu(name='HG1_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res8_batch3')
             .relu(name='HG1_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res8_conv3'))

        (self.feed('HG1_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res8_skip'))

        (self.feed('HG1_res8_conv3',
                   'HG1_res8_skip')
         .add(name='HG1_res8'))

# res9
        (self.feed('HG1_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res9_batch1')
             .relu(name='HG1_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res9_batch2')
             .relu(name='HG1_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res9_batch3')
             .relu(name='HG1_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res9_conv3'))

        (self.feed('HG1_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res9_skip'))

        (self.feed('HG1_res9_conv3',
                   'HG1_res9_skip')
         .add(name='HG1_res9'))

# res10
        (self.feed('HG1_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res10_batch1')
             .relu(name='HG1_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res10_batch2')
             .relu(name='HG1_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res10_batch3')
             .relu(name='HG1_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res10_conv3'))

        (self.feed('HG1_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res10_skip'))

        (self.feed('HG1_res10_conv3',
                   'HG1_res10_skip')
         .add(name='HG1_res10'))


# upsample1
        (self.feed('HG1_res10')
             .upsample(8, 8, name='HG1_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG1_upSample1',
                   'HG1_res7')
         .add(name='HG1_add1'))


# res11
        (self.feed('HG1_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res11_batch1')
             .relu(name='HG1_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res11_batch2')
             .relu(name='HG1_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res11_batch3')
             .relu(name='HG1_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res11_conv3'))

        (self.feed('HG1_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res11_skip'))

        (self.feed('HG1_res11_conv3',
                   'HG1_res11_skip')
         .add(name='HG1_res11'))


# upsample2
        (self.feed('HG1_res11')
             .upsample(16, 16, name='HG1_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG1_upSample2',
                   'HG1_res5')
         .add(name='HG1_add2'))


# res12
        (self.feed('HG1_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res12_batch1')
             .relu(name='HG1_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res12_batch2')
             .relu(name='HG1_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res12_batch3')
             .relu(name='HG1_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res12_conv3'))

        (self.feed('HG1_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res12_skip'))

        (self.feed('HG1_res12_conv3',
                   'HG1_res12_skip')
         .add(name='HG1_res12'))


# upsample3
        (self.feed('HG1_res12')
             .upsample(32, 32, name='HG1_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG1_upSample3',
                   'HG1_res3')
         .add(name='HG1_add3'))


# res13
        (self.feed('HG1_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res13_batch1')
             .relu(name='HG1_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res13_batch2')
             .relu(name='HG1_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res13_batch3')
             .relu(name='HG1_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res13_conv3'))

        (self.feed('HG1_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res13_skip'))

        (self.feed('HG1_res13_conv3',
                   'HG1_res13_skip')
         .add(name='HG1_res13'))


# upsample4
        (self.feed('HG1_res13')
             .upsample(64, 64, name='HG1_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG1_upSample4',
                   'HG1_res1')
         .add(name='HG1_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass1  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass1 postprocess #################

# id:025  Res14
        (self.feed('HG1_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res11_batch1')
             .relu(name='HG1_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG1_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res14_batch2')
             .relu(name='HG1_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG1_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_res14_batch3')
             .relu(name='HG1_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res14_conv3')

         )

        (self.feed('HG1_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_res14_skip'))

        (self.feed('HG1_res14_conv3',
                   'HG1_res14_skip')
         .add(name='HG1_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG1_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG1_linearfunc_batch1')
             .relu(name='HG1_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG1_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG1_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass1 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Hourglass ##########

# ll_
        (self.feed('HG1_linearfunc_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_ll_', padding='SAME'))
# tmpOut_
        (self.feed('HG1_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG1_tmpOut_', padding='SAME'))
# inter
        (self.feed('Res3',
                   'HG1_ll_',
                   'HG1_tmpOut_')
         .add(name='HG2_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


#######################################  Hourglass2  #####################
# res1
        (self.feed('HG2_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch1')
             .relu(name='HG2_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch2')
             .relu(name='HG2_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res1_batch3')
             .relu(name='HG2_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res1_conv3'))

        (self.feed('HG2_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res1_skip'))

        (self.feed('HG2_res1_conv3',
                   'HG2_res1_skip')
         .add(name='HG2_res1'))

#   pool1
        (self.feed('HG2_input')
             .max_pool(2, 2, 2, 2, name='HG2_pool1'))


# res2
        (self.feed('HG2_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch1')
             .relu(name='HG2_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch2')
             .relu(name='HG2_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res2_batch3')
             .relu(name='HG2_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res2_conv3'))

        (self.feed('HG2_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res2_skip'))

        (self.feed('HG2_res2_conv3',
                   'HG2_res2_skip')
         .add(name='HG2_res2'))
# id:009 max-pooling
        # (self.feed('HG2_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG2_pool2'))


# res3
        (self.feed('HG2_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch1')
             .relu(name='HG2_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch2')
             .relu(name='HG2_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res3_batch3')
             .relu(name='HG2_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res3_conv3'))

        (self.feed('HG2_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res3_skip'))

        (self.feed('HG2_res3_conv3',
                   'HG2_res3_skip')
         .add(name='HG2_res3'))


# pool2
        (self.feed('HG2_res2')
             .max_pool(2, 2, 2, 2, name='HG2_pool2'))


# res4
        (self.feed('HG2_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch1')
             .relu(name='HG2_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch2')
             .relu(name='HG2_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res4_batch3')
             .relu(name='HG2_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res4_conv3'))

        (self.feed('HG2_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res4_skip'))

        (self.feed('HG2_res4_conv3',
                   'HG2_res4_skip')
         .add(name='HG2_res4'))
# id:013 max-pooling
        # (self.feed('HG2_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG2_pool4'))


# res5
        (self.feed('HG2_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch1')
             .relu(name='HG2_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch2')
             .relu(name='HG2_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res5_batch3')
             .relu(name='HG2_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res5_conv3'))

        (self.feed('HG2_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res5_skip'))

        (self.feed('HG2_res5_conv3',
                   'HG2_res5_skip')
         .add(name='HG2_res5'))


# pool3
        (self.feed('HG2_res4')
             .max_pool(2, 2, 2, 2, name='HG2_pool3'))


# res6
        (self.feed('HG2_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch1')
             .relu(name='HG2_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch2')
             .relu(name='HG2_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res6_batch3')
             .relu(name='HG2_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res6_conv3'))

        (self.feed('HG2_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res6_skip'))

        (self.feed('HG2_res6_conv3',
                   'HG2_res6_skip')
         .add(name='HG2_res6'))

# res7
        (self.feed('HG2_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch1')
             .relu(name='HG2_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch2')
             .relu(name='HG2_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res7_batch3')
             .relu(name='HG2_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res7_conv3'))

        (self.feed('HG2_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res7_skip'))

        (self.feed('HG2_res7_conv3',
                   'HG2_res7_skip')
         .add(name='HG2_res7'))


# pool4
        (self.feed('HG2_res6')
             .max_pool(2, 2, 2, 2, name='HG2_pool4'))

# res8
        (self.feed('HG2_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch1')
             .relu(name='HG2_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch2')
             .relu(name='HG2_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res8_batch3')
             .relu(name='HG2_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res8_conv3'))

        (self.feed('HG2_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res8_skip'))

        (self.feed('HG2_res8_conv3',
                   'HG2_res8_skip')
         .add(name='HG2_res8'))

# res9
        (self.feed('HG2_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch1')
             .relu(name='HG2_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch2')
             .relu(name='HG2_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res9_batch3')
             .relu(name='HG2_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res9_conv3'))

        (self.feed('HG2_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res9_skip'))

        (self.feed('HG2_res9_conv3',
                   'HG2_res9_skip')
         .add(name='HG2_res9'))

# res10
        (self.feed('HG2_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch1')
             .relu(name='HG2_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch2')
             .relu(name='HG2_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res10_batch3')
             .relu(name='HG2_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res10_conv3'))

        (self.feed('HG2_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res10_skip'))

        (self.feed('HG2_res10_conv3',
                   'HG2_res10_skip')
         .add(name='HG2_res10'))


# upsample1
        (self.feed('HG2_res10')
             .upsample(8, 8, name='HG2_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG2_upSample1',
                   'HG2_res7')
         .add(name='HG2_add1'))


# res11
        (self.feed('HG2_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch1')
             .relu(name='HG2_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch2')
             .relu(name='HG2_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res11_batch3')
             .relu(name='HG2_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res11_conv3'))

        (self.feed('HG2_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res11_skip'))

        (self.feed('HG2_res11_conv3',
                   'HG2_res11_skip')
         .add(name='HG2_res11'))


# upsample2
        (self.feed('HG2_res11')
             .upsample(16, 16, name='HG2_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG2_upSample2',
                   'HG2_res5')
         .add(name='HG2_add2'))


# res12
        (self.feed('HG2_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch1')
             .relu(name='HG2_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch2')
             .relu(name='HG2_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res12_batch3')
             .relu(name='HG2_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res12_conv3'))

        (self.feed('HG2_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res12_skip'))

        (self.feed('HG2_res12_conv3',
                   'HG2_res12_skip')
         .add(name='HG2_res12'))


# upsample3
        (self.feed('HG2_res12')
             .upsample(32, 32, name='HG2_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG2_upSample3',
                   'HG2_res3')
         .add(name='HG2_add3'))


# res13
        (self.feed('HG2_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch1')
             .relu(name='HG2_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch2')
             .relu(name='HG2_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res13_batch3')
             .relu(name='HG2_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res13_conv3'))

        (self.feed('HG2_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res13_skip'))

        (self.feed('HG2_res13_conv3',
                   'HG2_res13_skip')
         .add(name='HG2_res13'))


# upsample4
        (self.feed('HG2_res13')
             .upsample(64, 64, name='HG2_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG2_upSample4',
                   'HG2_res1')
         .add(name='HG2_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass2  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass2 postprocess #################

# id:025  Res14
        (self.feed('HG2_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res14_batch1')
             .relu(name='HG2_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG2_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res14_batch2')
             .relu(name='HG2_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG2_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_res14_batch3')
             .relu(name='HG2_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res14_conv3')

         )

        (self.feed('HG2_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_res14_skip'))

        (self.feed('HG2_res14_conv3',
                   'HG2_res14_skip')
         .add(name='HG2_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG2_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG2_linearfunc_batch1')
             .relu(name='HG2_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG2_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG2_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass2 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Hourglass ##########

# ll_
        (self.feed('HG2_linearfunc_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_ll_', padding='SAME'))
# tmpOut_
        (self.feed('HG2_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG2_tmpOut_', padding='SAME'))
# inter
        (self.feed('HG2_input',
                   'HG2_ll_',
                   'HG2_tmpOut_')
         .add(name='HG3_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


#######################################  Hourglass3  #####################
# res1
        (self.feed('HG3_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch1')
             .relu(name='HG3_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch2')
             .relu(name='HG3_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res1_batch3')
             .relu(name='HG3_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res1_conv3'))

        (self.feed('HG3_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res1_skip'))

        (self.feed('HG3_res1_conv3',
                   'HG3_res1_skip')
         .add(name='HG3_res1'))

#   pool1
        (self.feed('HG3_input')
             .max_pool(2, 2, 2, 2, name='HG3_pool1'))


# res2
        (self.feed('HG3_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch1')
             .relu(name='HG3_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch2')
             .relu(name='HG3_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res2_batch3')
             .relu(name='HG3_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res2_conv3'))

        (self.feed('HG3_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res2_skip'))

        (self.feed('HG3_res2_conv3',
                   'HG3_res2_skip')
         .add(name='HG3_res2'))
# id:009 max-pooling
        # (self.feed('HG3_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG3_pool2'))


# res3
        (self.feed('HG3_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch1')
             .relu(name='HG3_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch2')
             .relu(name='HG3_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res3_batch3')
             .relu(name='HG3_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res3_conv3'))

        (self.feed('HG3_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res3_skip'))

        (self.feed('HG3_res3_conv3',
                   'HG3_res3_skip')
         .add(name='HG3_res3'))


# pool2
        (self.feed('HG3_res2')
             .max_pool(2, 2, 2, 2, name='HG3_pool2'))


# res4
        (self.feed('HG3_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch1')
             .relu(name='HG3_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch2')
             .relu(name='HG3_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res4_batch3')
             .relu(name='HG3_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res4_conv3'))

        (self.feed('HG3_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res4_skip'))

        (self.feed('HG3_res4_conv3',
                   'HG3_res4_skip')
         .add(name='HG3_res4'))
# id:013 max-pooling
        # (self.feed('HG3_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG3_pool4'))


# res5
        (self.feed('HG3_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch1')
             .relu(name='HG3_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch2')
             .relu(name='HG3_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res5_batch3')
             .relu(name='HG3_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res5_conv3'))

        (self.feed('HG3_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res5_skip'))

        (self.feed('HG3_res5_conv3',
                   'HG3_res5_skip')
         .add(name='HG3_res5'))


# pool3
        (self.feed('HG3_res4')
             .max_pool(2, 2, 2, 2, name='HG3_pool3'))


# res6
        (self.feed('HG3_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch1')
             .relu(name='HG3_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch2')
             .relu(name='HG3_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res6_batch3')
             .relu(name='HG3_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res6_conv3'))

        (self.feed('HG3_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res6_skip'))

        (self.feed('HG3_res6_conv3',
                   'HG3_res6_skip')
         .add(name='HG3_res6'))

# res7
        (self.feed('HG3_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch1')
             .relu(name='HG3_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch2')
             .relu(name='HG3_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res7_batch3')
             .relu(name='HG3_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res7_conv3'))

        (self.feed('HG3_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res7_skip'))

        (self.feed('HG3_res7_conv3',
                   'HG3_res7_skip')
         .add(name='HG3_res7'))


# pool4
        (self.feed('HG3_res6')
             .max_pool(2, 2, 2, 2, name='HG3_pool4'))

# res8
        (self.feed('HG3_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch1')
             .relu(name='HG3_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch2')
             .relu(name='HG3_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res8_batch3')
             .relu(name='HG3_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res8_conv3'))

        (self.feed('HG3_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res8_skip'))

        (self.feed('HG3_res8_conv3',
                   'HG3_res8_skip')
         .add(name='HG3_res8'))

# res9
        (self.feed('HG3_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch1')
             .relu(name='HG3_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch2')
             .relu(name='HG3_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res9_batch3')
             .relu(name='HG3_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res9_conv3'))

        (self.feed('HG3_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res9_skip'))

        (self.feed('HG3_res9_conv3',
                   'HG3_res9_skip')
         .add(name='HG3_res9'))

# res10
        (self.feed('HG3_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch1')
             .relu(name='HG3_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch2')
             .relu(name='HG3_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res10_batch3')
             .relu(name='HG3_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res10_conv3'))

        (self.feed('HG3_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res10_skip'))

        (self.feed('HG3_res10_conv3',
                   'HG3_res10_skip')
         .add(name='HG3_res10'))


# upsample1
        (self.feed('HG3_res10')
             .upsample(8, 8, name='HG3_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG3_upSample1',
                   'HG3_res7')
         .add(name='HG3_add1'))


# res11
        (self.feed('HG3_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch1')
             .relu(name='HG3_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch2')
             .relu(name='HG3_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res11_batch3')
             .relu(name='HG3_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res11_conv3'))

        (self.feed('HG3_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res11_skip'))

        (self.feed('HG3_res11_conv3',
                   'HG3_res11_skip')
         .add(name='HG3_res11'))


# upsample2
        (self.feed('HG3_res11')
             .upsample(16, 16, name='HG3_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG3_upSample2',
                   'HG3_res5')
         .add(name='HG3_add2'))


# res12
        (self.feed('HG3_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch1')
             .relu(name='HG3_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch2')
             .relu(name='HG3_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res12_batch3')
             .relu(name='HG3_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res12_conv3'))

        (self.feed('HG3_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res12_skip'))

        (self.feed('HG3_res12_conv3',
                   'HG3_res12_skip')
         .add(name='HG3_res12'))


# upsample3
        (self.feed('HG3_res12')
             .upsample(32, 32, name='HG3_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG3_upSample3',
                   'HG3_res3')
         .add(name='HG3_add3'))


# res13
        (self.feed('HG3_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch1')
             .relu(name='HG3_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch2')
             .relu(name='HG3_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res13_batch3')
             .relu(name='HG3_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res13_conv3'))

        (self.feed('HG3_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res13_skip'))

        (self.feed('HG3_res13_conv3',
                   'HG3_res13_skip')
         .add(name='HG3_res13'))


# upsample4
        (self.feed('HG3_res13')
             .upsample(64, 64, name='HG3_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG3_upSample4',
                   'HG3_res1')
         .add(name='HG3_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass3  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass3 postprocess #################

# id:025  Res14
        (self.feed('HG3_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res14_batch1')
             .relu(name='HG3_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG3_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res14_batch2')
             .relu(name='HG3_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG3_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_res14_batch3')
             .relu(name='HG3_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res14_conv3')

         )

        (self.feed('HG3_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_res14_skip'))

        (self.feed('HG3_res14_conv3',
                   'HG3_res14_skip')
         .add(name='HG3_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG3_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG3_linearfunc_batch1')
             .relu(name='HG3_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG3_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG3_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass3 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Hourglass ##########

# ll_
        (self.feed('HG3_linearfunc_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_ll_', padding='SAME'))
# tmpOut_
        (self.feed('HG3_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG3_tmpOut_', padding='SAME'))
# inter
        (self.feed('HG3_input',
                   'HG3_ll_',
                   'HG3_tmpOut_')
         .add(name='HG4_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###


#######################################  Hourglass4  #####################
# res1
        (self.feed('HG4_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch1')
             .relu(name='HG4_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch2')
             .relu(name='HG4_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res1_batch3')
             .relu(name='HG4_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res1_conv3'))

        (self.feed('HG4_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res1_skip'))

        (self.feed('HG4_res1_conv3',
                   'HG4_res1_skip')
         .add(name='HG4_res1'))

#   pool1
        (self.feed('HG4_input')
             .max_pool(2, 2, 2, 2, name='HG4_pool1'))


# res2
        (self.feed('HG4_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch1')
             .relu(name='HG4_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch2')
             .relu(name='HG4_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res2_batch3')
             .relu(name='HG4_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res2_conv3'))

        (self.feed('HG4_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res2_skip'))

        (self.feed('HG4_res2_conv3',
                   'HG4_res2_skip')
         .add(name='HG4_res2'))
# id:009 max-pooling
        # (self.feed('HG4_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG4_pool2'))


# res3
        (self.feed('HG4_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch1')
             .relu(name='HG4_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch2')
             .relu(name='HG4_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res3_batch3')
             .relu(name='HG4_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res3_conv3'))

        (self.feed('HG4_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res3_skip'))

        (self.feed('HG4_res3_conv3',
                   'HG4_res3_skip')
         .add(name='HG4_res3'))


# pool2
        (self.feed('HG4_res2')
             .max_pool(2, 2, 2, 2, name='HG4_pool2'))


# res4
        (self.feed('HG4_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch1')
             .relu(name='HG4_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch2')
             .relu(name='HG4_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res4_batch3')
             .relu(name='HG4_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res4_conv3'))

        (self.feed('HG4_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res4_skip'))

        (self.feed('HG4_res4_conv3',
                   'HG4_res4_skip')
         .add(name='HG4_res4'))
# id:013 max-pooling
        # (self.feed('HG4_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG4_pool4'))


# res5
        (self.feed('HG4_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch1')
             .relu(name='HG4_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch2')
             .relu(name='HG4_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res5_batch3')
             .relu(name='HG4_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res5_conv3'))

        (self.feed('HG4_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res5_skip'))

        (self.feed('HG4_res5_conv3',
                   'HG4_res5_skip')
         .add(name='HG4_res5'))


# pool3
        (self.feed('HG4_res4')
             .max_pool(2, 2, 2, 2, name='HG4_pool3'))


# res6
        (self.feed('HG4_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch1')
             .relu(name='HG4_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch2')
             .relu(name='HG4_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res6_batch3')
             .relu(name='HG4_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res6_conv3'))

        (self.feed('HG4_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res6_skip'))

        (self.feed('HG4_res6_conv3',
                   'HG4_res6_skip')
         .add(name='HG4_res6'))

# res7
        (self.feed('HG4_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch1')
             .relu(name='HG4_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch2')
             .relu(name='HG4_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res7_batch3')
             .relu(name='HG4_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res7_conv3'))

        (self.feed('HG4_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res7_skip'))

        (self.feed('HG4_res7_conv3',
                   'HG4_res7_skip')
         .add(name='HG4_res7'))


# pool4
        (self.feed('HG4_res6')
             .max_pool(2, 2, 2, 2, name='HG4_pool4'))

# res8
        (self.feed('HG4_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch1')
             .relu(name='HG4_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch2')
             .relu(name='HG4_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res8_batch3')
             .relu(name='HG4_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res8_conv3'))

        (self.feed('HG4_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res8_skip'))

        (self.feed('HG4_res8_conv3',
                   'HG4_res8_skip')
         .add(name='HG4_res8'))

# res9
        (self.feed('HG4_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch1')
             .relu(name='HG4_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch2')
             .relu(name='HG4_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res9_batch3')
             .relu(name='HG4_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res9_conv3'))

        (self.feed('HG4_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res9_skip'))

        (self.feed('HG4_res9_conv3',
                   'HG4_res9_skip')
         .add(name='HG4_res9'))

# res10
        (self.feed('HG4_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch1')
             .relu(name='HG4_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch2')
             .relu(name='HG4_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res10_batch3')
             .relu(name='HG4_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res10_conv3'))

        (self.feed('HG4_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res10_skip'))

        (self.feed('HG4_res10_conv3',
                   'HG4_res10_skip')
         .add(name='HG4_res10'))


# upsample1
        (self.feed('HG4_res10')
             .upsample(8, 8, name='HG4_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG4_upSample1',
                   'HG4_res7')
         .add(name='HG4_add1'))


# res11
        (self.feed('HG4_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch1')
             .relu(name='HG4_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch2')
             .relu(name='HG4_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res11_batch3')
             .relu(name='HG4_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res11_conv3'))

        (self.feed('HG4_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res11_skip'))

        (self.feed('HG4_res11_conv3',
                   'HG4_res11_skip')
         .add(name='HG4_res11'))


# upsample2
        (self.feed('HG4_res11')
             .upsample(16, 16, name='HG4_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG4_upSample2',
                   'HG4_res5')
         .add(name='HG4_add2'))


# res12
        (self.feed('HG4_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch1')
             .relu(name='HG4_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch2')
             .relu(name='HG4_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res12_batch3')
             .relu(name='HG4_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res12_conv3'))

        (self.feed('HG4_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res12_skip'))

        (self.feed('HG4_res12_conv3',
                   'HG4_res12_skip')
         .add(name='HG4_res12'))


# upsample3
        (self.feed('HG4_res12')
             .upsample(32, 32, name='HG4_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG4_upSample3',
                   'HG4_res3')
         .add(name='HG4_add3'))


# res13
        (self.feed('HG4_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch1')
             .relu(name='HG4_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch2')
             .relu(name='HG4_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res13_batch3')
             .relu(name='HG4_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res13_conv3'))

        (self.feed('HG4_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res13_skip'))

        (self.feed('HG4_res13_conv3',
                   'HG4_res13_skip')
         .add(name='HG4_res13'))


# upsample4
        (self.feed('HG4_res13')
             .upsample(64, 64, name='HG4_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG4_upSample4',
                   'HG4_res1')
         .add(name='HG4_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass4  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass4 postprocess #################

# id:025  Res14
        (self.feed('HG4_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res14_batch1')
             .relu(name='HG4_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG4_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res14_batch2')
             .relu(name='HG4_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG4_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_res14_batch3')
             .relu(name='HG4_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res14_conv3')

         )

        (self.feed('HG4_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_res14_skip'))

        (self.feed('HG4_res14_conv3',
                   'HG4_res14_skip')
         .add(name='HG4_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG4_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG4_linearfunc_batch1')
             .relu(name='HG4_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG4_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG4_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass4 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Hourglass ##########

# ll_
        (self.feed('HG4_linearfunc_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_ll_', padding='SAME'))
# tmpOut_
        (self.feed('HG4_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG4_tmpOut_', padding='SAME'))
# inter
        (self.feed('HG4_input',
                   'HG4_ll_',
                   'HG4_tmpOut_')
         .add(name='HG5_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###




#######################################  Hourglass5  #####################
# res1
        (self.feed('HG5_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch1')
             .relu(name='HG5_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch2')
             .relu(name='HG5_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res1_batch3')
             .relu(name='HG5_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res1_conv3'))

        (self.feed('HG5_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res1_skip'))

        (self.feed('HG5_res1_conv3',
                   'HG5_res1_skip')
         .add(name='HG5_res1'))

#   pool1
        (self.feed('HG5_input')
             .max_pool(2, 2, 2, 2, name='HG5_pool1'))


# res2
        (self.feed('HG5_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch1')
             .relu(name='HG5_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch2')
             .relu(name='HG5_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res2_batch3')
             .relu(name='HG5_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res2_conv3'))

        (self.feed('HG5_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res2_skip'))

        (self.feed('HG5_res2_conv3',
                   'HG5_res2_skip')
         .add(name='HG5_res2'))
# id:009 max-pooling
        # (self.feed('HG5_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG5_pool2'))


# res3
        (self.feed('HG5_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch1')
             .relu(name='HG5_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch2')
             .relu(name='HG5_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res3_batch3')
             .relu(name='HG5_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res3_conv3'))

        (self.feed('HG5_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res3_skip'))

        (self.feed('HG5_res3_conv3',
                   'HG5_res3_skip')
         .add(name='HG5_res3'))


# pool2
        (self.feed('HG5_res2')
             .max_pool(2, 2, 2, 2, name='HG5_pool2'))


# res4
        (self.feed('HG5_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch1')
             .relu(name='HG5_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch2')
             .relu(name='HG5_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res4_batch3')
             .relu(name='HG5_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res4_conv3'))

        (self.feed('HG5_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res4_skip'))

        (self.feed('HG5_res4_conv3',
                   'HG5_res4_skip')
         .add(name='HG5_res4'))
# id:013 max-pooling
        # (self.feed('HG5_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG5_pool4'))


# res5
        (self.feed('HG5_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch1')
             .relu(name='HG5_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch2')
             .relu(name='HG5_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res5_batch3')
             .relu(name='HG5_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res5_conv3'))

        (self.feed('HG5_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res5_skip'))

        (self.feed('HG5_res5_conv3',
                   'HG5_res5_skip')
         .add(name='HG5_res5'))


# pool3
        (self.feed('HG5_res4')
             .max_pool(2, 2, 2, 2, name='HG5_pool3'))


# res6
        (self.feed('HG5_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch1')
             .relu(name='HG5_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch2')
             .relu(name='HG5_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res6_batch3')
             .relu(name='HG5_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res6_conv3'))

        (self.feed('HG5_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res6_skip'))

        (self.feed('HG5_res6_conv3',
                   'HG5_res6_skip')
         .add(name='HG5_res6'))

# res7
        (self.feed('HG5_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch1')
             .relu(name='HG5_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch2')
             .relu(name='HG5_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res7_batch3')
             .relu(name='HG5_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res7_conv3'))

        (self.feed('HG5_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res7_skip'))

        (self.feed('HG5_res7_conv3',
                   'HG5_res7_skip')
         .add(name='HG5_res7'))


# pool4
        (self.feed('HG5_res6')
             .max_pool(2, 2, 2, 2, name='HG5_pool4'))

# res8
        (self.feed('HG5_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch1')
             .relu(name='HG5_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch2')
             .relu(name='HG5_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res8_batch3')
             .relu(name='HG5_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res8_conv3'))

        (self.feed('HG5_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res8_skip'))

        (self.feed('HG5_res8_conv3',
                   'HG5_res8_skip')
         .add(name='HG5_res8'))

# res9
        (self.feed('HG5_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch1')
             .relu(name='HG5_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch2')
             .relu(name='HG5_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res9_batch3')
             .relu(name='HG5_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res9_conv3'))

        (self.feed('HG5_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res9_skip'))

        (self.feed('HG5_res9_conv3',
                   'HG5_res9_skip')
         .add(name='HG5_res9'))

# res10
        (self.feed('HG5_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch1')
             .relu(name='HG5_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch2')
             .relu(name='HG5_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res10_batch3')
             .relu(name='HG5_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res10_conv3'))

        (self.feed('HG5_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res10_skip'))

        (self.feed('HG5_res10_conv3',
                   'HG5_res10_skip')
         .add(name='HG5_res10'))


# upsample1
        (self.feed('HG5_res10')
             .upsample(8, 8, name='HG5_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG5_upSample1',
                   'HG5_res7')
         .add(name='HG5_add1'))


# res11
        (self.feed('HG5_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch1')
             .relu(name='HG5_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch2')
             .relu(name='HG5_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res11_batch3')
             .relu(name='HG5_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res11_conv3'))

        (self.feed('HG5_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res11_skip'))

        (self.feed('HG5_res11_conv3',
                   'HG5_res11_skip')
         .add(name='HG5_res11'))


# upsample2
        (self.feed('HG5_res11')
             .upsample(16, 16, name='HG5_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG5_upSample2',
                   'HG5_res5')
         .add(name='HG5_add2'))


# res12
        (self.feed('HG5_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch1')
             .relu(name='HG5_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch2')
             .relu(name='HG5_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res12_batch3')
             .relu(name='HG5_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res12_conv3'))

        (self.feed('HG5_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res12_skip'))

        (self.feed('HG5_res12_conv3',
                   'HG5_res12_skip')
         .add(name='HG5_res12'))


# upsample3
        (self.feed('HG5_res12')
             .upsample(32, 32, name='HG5_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG5_upSample3',
                   'HG5_res3')
         .add(name='HG5_add3'))


# res13
        (self.feed('HG5_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch1')
             .relu(name='HG5_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch2')
             .relu(name='HG5_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res13_batch3')
             .relu(name='HG5_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res13_conv3'))

        (self.feed('HG5_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res13_skip'))

        (self.feed('HG5_res13_conv3',
                   'HG5_res13_skip')
         .add(name='HG5_res13'))


# upsample4
        (self.feed('HG5_res13')
             .upsample(64, 64, name='HG5_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG5_upSample4',
                   'HG5_res1')
         .add(name='HG5_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass5  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass5 postprocess #################

# id:025  Res14
        (self.feed('HG5_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res14_batch1')
             .relu(name='HG5_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG5_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res14_batch2')
             .relu(name='HG5_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG5_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_res14_batch3')
             .relu(name='HG5_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res14_conv3')

         )

        (self.feed('HG5_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_res14_skip'))

        (self.feed('HG5_res14_conv3',
                   'HG5_res14_skip')
         .add(name='HG5_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG5_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG5_linearfunc_batch1')
             .relu(name='HG5_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG5_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG5_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass5 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Hourglass ##########

# ll_
        (self.feed('HG5_linearfunc_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_ll_', padding='SAME'))
# tmpOut_
        (self.feed('HG5_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG5_tmpOut_', padding='SAME'))
# inter
        (self.feed('HG5_input',
                   'HG5_ll_',
                   'HG5_tmpOut_')
         .add(name='HG6_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###





#######################################  Hourglass6  #####################
# res1
        (self.feed('HG6_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch1')
             .relu(name='HG6_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch2')
             .relu(name='HG6_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res1_batch3')
             .relu(name='HG6_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res1_conv3'))

        (self.feed('HG6_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res1_skip'))

        (self.feed('HG6_res1_conv3',
                   'HG6_res1_skip')
         .add(name='HG6_res1'))

#   pool1
        (self.feed('HG6_input')
             .max_pool(2, 2, 2, 2, name='HG6_pool1'))


# res2
        (self.feed('HG6_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch1')
             .relu(name='HG6_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch2')
             .relu(name='HG6_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res2_batch3')
             .relu(name='HG6_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res2_conv3'))

        (self.feed('HG6_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res2_skip'))

        (self.feed('HG6_res2_conv3',
                   'HG6_res2_skip')
         .add(name='HG6_res2'))
# id:009 max-pooling
        # (self.feed('HG6_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG6_pool2'))


# res3
        (self.feed('HG6_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch1')
             .relu(name='HG6_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch2')
             .relu(name='HG6_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res3_batch3')
             .relu(name='HG6_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res3_conv3'))

        (self.feed('HG6_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res3_skip'))

        (self.feed('HG6_res3_conv3',
                   'HG6_res3_skip')
         .add(name='HG6_res3'))


# pool2
        (self.feed('HG6_res2')
             .max_pool(2, 2, 2, 2, name='HG6_pool2'))


# res4
        (self.feed('HG6_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch1')
             .relu(name='HG6_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch2')
             .relu(name='HG6_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res4_batch3')
             .relu(name='HG6_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res4_conv3'))

        (self.feed('HG6_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res4_skip'))

        (self.feed('HG6_res4_conv3',
                   'HG6_res4_skip')
         .add(name='HG6_res4'))
# id:013 max-pooling
        # (self.feed('HG6_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG6_pool4'))


# res5
        (self.feed('HG6_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch1')
             .relu(name='HG6_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch2')
             .relu(name='HG6_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res5_batch3')
             .relu(name='HG6_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res5_conv3'))

        (self.feed('HG6_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res5_skip'))

        (self.feed('HG6_res5_conv3',
                   'HG6_res5_skip')
         .add(name='HG6_res5'))


# pool3
        (self.feed('HG6_res4')
             .max_pool(2, 2, 2, 2, name='HG6_pool3'))


# res6
        (self.feed('HG6_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch1')
             .relu(name='HG6_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch2')
             .relu(name='HG6_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res6_batch3')
             .relu(name='HG6_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res6_conv3'))

        (self.feed('HG6_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res6_skip'))

        (self.feed('HG6_res6_conv3',
                   'HG6_res6_skip')
         .add(name='HG6_res6'))

# res7
        (self.feed('HG6_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch1')
             .relu(name='HG6_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch2')
             .relu(name='HG6_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res7_batch3')
             .relu(name='HG6_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res7_conv3'))

        (self.feed('HG6_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res7_skip'))

        (self.feed('HG6_res7_conv3',
                   'HG6_res7_skip')
         .add(name='HG6_res7'))


# pool4
        (self.feed('HG6_res6')
             .max_pool(2, 2, 2, 2, name='HG6_pool4'))

# res8
        (self.feed('HG6_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch1')
             .relu(name='HG6_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch2')
             .relu(name='HG6_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res8_batch3')
             .relu(name='HG6_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res8_conv3'))

        (self.feed('HG6_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res8_skip'))

        (self.feed('HG6_res8_conv3',
                   'HG6_res8_skip')
         .add(name='HG6_res8'))

# res9
        (self.feed('HG6_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch1')
             .relu(name='HG6_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch2')
             .relu(name='HG6_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res9_batch3')
             .relu(name='HG6_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res9_conv3'))

        (self.feed('HG6_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res9_skip'))

        (self.feed('HG6_res9_conv3',
                   'HG6_res9_skip')
         .add(name='HG6_res9'))

# res10
        (self.feed('HG6_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch1')
             .relu(name='HG6_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch2')
             .relu(name='HG6_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res10_batch3')
             .relu(name='HG6_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res10_conv3'))

        (self.feed('HG6_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res10_skip'))

        (self.feed('HG6_res10_conv3',
                   'HG6_res10_skip')
         .add(name='HG6_res10'))


# upsample1
        (self.feed('HG6_res10')
             .upsample(8, 8, name='HG6_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG6_upSample1',
                   'HG6_res7')
         .add(name='HG6_add1'))


# res11
        (self.feed('HG6_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch1')
             .relu(name='HG6_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch2')
             .relu(name='HG6_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res11_batch3')
             .relu(name='HG6_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res11_conv3'))

        (self.feed('HG6_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res11_skip'))

        (self.feed('HG6_res11_conv3',
                   'HG6_res11_skip')
         .add(name='HG6_res11'))


# upsample2
        (self.feed('HG6_res11')
             .upsample(16, 16, name='HG6_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG6_upSample2',
                   'HG6_res5')
         .add(name='HG6_add2'))


# res12
        (self.feed('HG6_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch1')
             .relu(name='HG6_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch2')
             .relu(name='HG6_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res12_batch3')
             .relu(name='HG6_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res12_conv3'))

        (self.feed('HG6_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res12_skip'))

        (self.feed('HG6_res12_conv3',
                   'HG6_res12_skip')
         .add(name='HG6_res12'))


# upsample3
        (self.feed('HG6_res12')
             .upsample(32, 32, name='HG6_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG6_upSample3',
                   'HG6_res3')
         .add(name='HG6_add3'))


# res13
        (self.feed('HG6_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch1')
             .relu(name='HG6_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch2')
             .relu(name='HG6_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res13_batch3')
             .relu(name='HG6_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res13_conv3'))

        (self.feed('HG6_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res13_skip'))

        (self.feed('HG6_res13_conv3',
                   'HG6_res13_skip')
         .add(name='HG6_res13'))


# upsample4
        (self.feed('HG6_res13')
             .upsample(64, 64, name='HG6_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG6_upSample4',
                   'HG6_res1')
         .add(name='HG6_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass6  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass6 postprocess #################

# id:025  Res14
        (self.feed('HG6_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res14_batch1')
             .relu(name='HG6_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG6_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res14_batch2')
             .relu(name='HG6_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG6_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_res14_batch3')
             .relu(name='HG6_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res14_conv3')

         )

        (self.feed('HG6_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_res14_skip'))

        (self.feed('HG6_res14_conv3',
                   'HG6_res14_skip')
         .add(name='HG6_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG6_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG6_linearfunc_batch1')
             .relu(name='HG6_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG6_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG6_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass6 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Hourglass ##########

# ll_
        (self.feed('HG6_linearfunc_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_ll_', padding='SAME'))
# tmpOut_
        (self.feed('HG6_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG6_tmpOut_', padding='SAME'))
# inter
        (self.feed('HG6_input',
                   'HG6_ll_',
                   'HG6_tmpOut_')
         .add(name='HG7_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###






#######################################  Hourglass7  #####################
# res1
        (self.feed('HG7_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch1')
             .relu(name='HG7_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch2')
             .relu(name='HG7_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res1_batch3')
             .relu(name='HG7_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res1_conv3'))

        (self.feed('HG7_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res1_skip'))

        (self.feed('HG7_res1_conv3',
                   'HG7_res1_skip')
         .add(name='HG7_res1'))

#   pool1
        (self.feed('HG7_input')
             .max_pool(2, 2, 2, 2, name='HG7_pool1'))


# res2
        (self.feed('HG7_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch1')
             .relu(name='HG7_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch2')
             .relu(name='HG7_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res2_batch3')
             .relu(name='HG7_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res2_conv3'))

        (self.feed('HG7_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res2_skip'))

        (self.feed('HG7_res2_conv3',
                   'HG7_res2_skip')
         .add(name='HG7_res2'))
# id:009 max-pooling
        # (self.feed('HG7_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG7_pool2'))


# res3
        (self.feed('HG7_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch1')
             .relu(name='HG7_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch2')
             .relu(name='HG7_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res3_batch3')
             .relu(name='HG7_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res3_conv3'))

        (self.feed('HG7_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res3_skip'))

        (self.feed('HG7_res3_conv3',
                   'HG7_res3_skip')
         .add(name='HG7_res3'))


# pool2
        (self.feed('HG7_res2')
             .max_pool(2, 2, 2, 2, name='HG7_pool2'))


# res4
        (self.feed('HG7_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch1')
             .relu(name='HG7_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch2')
             .relu(name='HG7_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res4_batch3')
             .relu(name='HG7_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res4_conv3'))

        (self.feed('HG7_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res4_skip'))

        (self.feed('HG7_res4_conv3',
                   'HG7_res4_skip')
         .add(name='HG7_res4'))
# id:013 max-pooling
        # (self.feed('HG7_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG7_pool4'))


# res5
        (self.feed('HG7_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch1')
             .relu(name='HG7_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch2')
             .relu(name='HG7_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res5_batch3')
             .relu(name='HG7_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res5_conv3'))

        (self.feed('HG7_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res5_skip'))

        (self.feed('HG7_res5_conv3',
                   'HG7_res5_skip')
         .add(name='HG7_res5'))


# pool3
        (self.feed('HG7_res4')
             .max_pool(2, 2, 2, 2, name='HG7_pool3'))


# res6
        (self.feed('HG7_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch1')
             .relu(name='HG7_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch2')
             .relu(name='HG7_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res6_batch3')
             .relu(name='HG7_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res6_conv3'))

        (self.feed('HG7_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res6_skip'))

        (self.feed('HG7_res6_conv3',
                   'HG7_res6_skip')
         .add(name='HG7_res6'))

# res7
        (self.feed('HG7_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch1')
             .relu(name='HG7_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch2')
             .relu(name='HG7_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res7_batch3')
             .relu(name='HG7_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res7_conv3'))

        (self.feed('HG7_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res7_skip'))

        (self.feed('HG7_res7_conv3',
                   'HG7_res7_skip')
         .add(name='HG7_res7'))


# pool4
        (self.feed('HG7_res6')
             .max_pool(2, 2, 2, 2, name='HG7_pool4'))

# res8
        (self.feed('HG7_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch1')
             .relu(name='HG7_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch2')
             .relu(name='HG7_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res8_batch3')
             .relu(name='HG7_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res8_conv3'))

        (self.feed('HG7_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res8_skip'))

        (self.feed('HG7_res8_conv3',
                   'HG7_res8_skip')
         .add(name='HG7_res8'))

# res9
        (self.feed('HG7_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch1')
             .relu(name='HG7_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch2')
             .relu(name='HG7_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res9_batch3')
             .relu(name='HG7_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res9_conv3'))

        (self.feed('HG7_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res9_skip'))

        (self.feed('HG7_res9_conv3',
                   'HG7_res9_skip')
         .add(name='HG7_res9'))

# res10
        (self.feed('HG7_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch1')
             .relu(name='HG7_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch2')
             .relu(name='HG7_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res10_batch3')
             .relu(name='HG7_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res10_conv3'))

        (self.feed('HG7_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res10_skip'))

        (self.feed('HG7_res10_conv3',
                   'HG7_res10_skip')
         .add(name='HG7_res10'))


# upsample1
        (self.feed('HG7_res10')
             .upsample(8, 8, name='HG7_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG7_upSample1',
                   'HG7_res7')
         .add(name='HG7_add1'))


# res11
        (self.feed('HG7_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch1')
             .relu(name='HG7_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch2')
             .relu(name='HG7_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res11_batch3')
             .relu(name='HG7_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res11_conv3'))

        (self.feed('HG7_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res11_skip'))

        (self.feed('HG7_res11_conv3',
                   'HG7_res11_skip')
         .add(name='HG7_res11'))


# upsample2
        (self.feed('HG7_res11')
             .upsample(16, 16, name='HG7_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG7_upSample2',
                   'HG7_res5')
         .add(name='HG7_add2'))


# res12
        (self.feed('HG7_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch1')
             .relu(name='HG7_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch2')
             .relu(name='HG7_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res12_batch3')
             .relu(name='HG7_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res12_conv3'))

        (self.feed('HG7_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res12_skip'))

        (self.feed('HG7_res12_conv3',
                   'HG7_res12_skip')
         .add(name='HG7_res12'))


# upsample3
        (self.feed('HG7_res12')
             .upsample(32, 32, name='HG7_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG7_upSample3',
                   'HG7_res3')
         .add(name='HG7_add3'))


# res13
        (self.feed('HG7_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch1')
             .relu(name='HG7_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch2')
             .relu(name='HG7_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res13_batch3')
             .relu(name='HG7_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res13_conv3'))

        (self.feed('HG7_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res13_skip'))

        (self.feed('HG7_res13_conv3',
                   'HG7_res13_skip')
         .add(name='HG7_res13'))


# upsample4
        (self.feed('HG7_res13')
             .upsample(64, 64, name='HG7_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG7_upSample4',
                   'HG7_res1')
         .add(name='HG7_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass7  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass7 postprocess #################

# id:025  Res14
        (self.feed('HG7_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res14_batch1')
             .relu(name='HG7_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG7_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res14_batch2')
             .relu(name='HG7_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG7_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_res14_batch3')
             .relu(name='HG7_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res14_conv3')

         )

        (self.feed('HG7_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_res14_skip'))

        (self.feed('HG7_res14_conv3',
                   'HG7_res14_skip')
         .add(name='HG7_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG7_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG7_linearfunc_batch1')
             .relu(name='HG7_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG7_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG7_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass7 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

############################# generate input for next Hourglass ##########

# ll_
        (self.feed('HG7_linearfunc_relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_ll_', padding='SAME'))
# tmpOut_
        (self.feed('HG7_Heatmap')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG7_tmpOut_', padding='SAME'))
# inter
        (self.feed('HG7_input',
                   'HG7_ll_',
                   'HG7_tmpOut_')
         .add(name='HG8_input'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^ generate input for next Hourglass ^^^^^^^^^^^^^^^^^^^^^^^^^^^^###
























#######################################  Hourglass8  #####################
# res1
        (self.feed('HG8_input')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch1')
             .relu(name='HG8_res1_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res1_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch2')
             .relu(name='HG8_res1_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res1_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res1_batch3')
             .relu(name='HG8_res1_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res1_conv3'))

        (self.feed('HG8_input')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res1_skip'))

        (self.feed('HG8_res1_conv3',
                   'HG8_res1_skip')
         .add(name='HG8_res1'))

#   pool1
        (self.feed('HG8_input')
             .max_pool(2, 2, 2, 2, name='HG8_pool1'))


# res2
        (self.feed('HG8_pool1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch1')
             .relu(name='HG8_res2_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res2_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch2')
             .relu(name='HG8_res2_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res2_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res2_batch3')
             .relu(name='HG8_res2_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res2_conv3'))

        (self.feed('HG8_pool1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res2_skip'))

        (self.feed('HG8_res2_conv3',
                   'HG8_res2_skip')
         .add(name='HG8_res2'))
# id:009 max-pooling
        # (self.feed('HG8_pool1')
        #      .max_pool(2, 2, 2, 2, name='HG8_pool2'))


# res3
        (self.feed('HG8_res2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch1')
             .relu(name='HG8_res3_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res3_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch2')
             .relu(name='HG8_res3_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res3_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res3_batch3')
             .relu(name='HG8_res3_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res3_conv3'))

        (self.feed('HG8_res2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res3_skip'))

        (self.feed('HG8_res3_conv3',
                   'HG8_res3_skip')
         .add(name='HG8_res3'))


# pool2
        (self.feed('HG8_res2')
             .max_pool(2, 2, 2, 2, name='HG8_pool2'))


# res4
        (self.feed('HG8_pool2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch1')
             .relu(name='HG8_res4_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res4_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch2')
             .relu(name='HG8_res4_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res4_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res4_batch3')
             .relu(name='HG8_res4_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res4_conv3'))

        (self.feed('HG8_pool2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res4_skip'))

        (self.feed('HG8_res4_conv3',
                   'HG8_res4_skip')
         .add(name='HG8_res4'))
# id:013 max-pooling
        # (self.feed('HG8_pool3')
        #      .max_pool(2, 2, 2, 2, name='HG8_pool4'))


# res5
        (self.feed('HG8_res4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch1')
             .relu(name='HG8_res5_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res5_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch2')
             .relu(name='HG8_res5_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res5_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res5_batch3')
             .relu(name='HG8_res5_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res5_conv3'))

        (self.feed('HG8_res4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res5_skip'))

        (self.feed('HG8_res5_conv3',
                   'HG8_res5_skip')
         .add(name='HG8_res5'))


# pool3
        (self.feed('HG8_res4')
             .max_pool(2, 2, 2, 2, name='HG8_pool3'))


# res6
        (self.feed('HG8_pool3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch1')
             .relu(name='HG8_res6_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res6_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch2')
             .relu(name='HG8_res6_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res6_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res6_batch3')
             .relu(name='HG8_res6_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res6_conv3'))

        (self.feed('HG8_pool3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res6_skip'))

        (self.feed('HG8_res6_conv3',
                   'HG8_res6_skip')
         .add(name='HG8_res6'))

# res7
        (self.feed('HG8_res6')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch1')
             .relu(name='HG8_res7_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res7_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch2')
             .relu(name='HG8_res7_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res7_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res7_batch3')
             .relu(name='HG8_res7_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res7_conv3'))

        (self.feed('HG8_res6')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res7_skip'))

        (self.feed('HG8_res7_conv3',
                   'HG8_res7_skip')
         .add(name='HG8_res7'))


# pool4
        (self.feed('HG8_res6')
             .max_pool(2, 2, 2, 2, name='HG8_pool4'))

# res8
        (self.feed('HG8_pool4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch1')
             .relu(name='HG8_res8_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res8_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch2')
             .relu(name='HG8_res8_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res8_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res8_batch3')
             .relu(name='HG8_res8_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res8_conv3'))

        (self.feed('HG8_pool4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res8_skip'))

        (self.feed('HG8_res8_conv3',
                   'HG8_res8_skip')
         .add(name='HG8_res8'))

# res9
        (self.feed('HG8_res8')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch1')
             .relu(name='HG8_res9_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res9_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch2')
             .relu(name='HG8_res9_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res9_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res9_batch3')
             .relu(name='HG8_res9_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res9_conv3'))

        (self.feed('HG8_res8')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res9_skip'))

        (self.feed('HG8_res9_conv3',
                   'HG8_res9_skip')
         .add(name='HG8_res9'))

# res10
        (self.feed('HG8_res9')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch1')
             .relu(name='HG8_res10_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res10_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch2')
             .relu(name='HG8_res10_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res10_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res10_batch3')
             .relu(name='HG8_res10_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res10_conv3'))

        (self.feed('HG8_res9')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res10_skip'))

        (self.feed('HG8_res10_conv3',
                   'HG8_res10_skip')
         .add(name='HG8_res10'))


# upsample1
        (self.feed('HG8_res10')
             .upsample(8, 8, name='HG8_upSample1'))

# upsample1 + up1(Hg_res7)
        (self.feed('HG8_upSample1',
                   'HG8_res7')
         .add(name='HG8_add1'))


# res11
        (self.feed('HG8_add1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch1')
             .relu(name='HG8_res11_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res11_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch2')
             .relu(name='HG8_res11_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res11_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res11_batch3')
             .relu(name='HG8_res11_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res11_conv3'))

        (self.feed('HG8_add1')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res11_skip'))

        (self.feed('HG8_res11_conv3',
                   'HG8_res11_skip')
         .add(name='HG8_res11'))


# upsample2
        (self.feed('HG8_res11')
             .upsample(16, 16, name='HG8_upSample2'))

# upsample2 + up1(Hg_res5)
        (self.feed('HG8_upSample2',
                   'HG8_res5')
         .add(name='HG8_add2'))


# res12
        (self.feed('HG8_add2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch1')
             .relu(name='HG8_res12_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res12_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch2')
             .relu(name='HG8_res12_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res12_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res12_batch3')
             .relu(name='HG8_res12_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res12_conv3'))

        (self.feed('HG8_add2')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res12_skip'))

        (self.feed('HG8_res12_conv3',
                   'HG8_res12_skip')
         .add(name='HG8_res12'))


# upsample3
        (self.feed('HG8_res12')
             .upsample(32, 32, name='HG8_upSample3'))

# upsample3 + up1(Hg_res3)
        (self.feed('HG8_upSample3',
                   'HG8_res3')
         .add(name='HG8_add3'))


# res13
        (self.feed('HG8_add3')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch1')
             .relu(name='HG8_res13_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res13_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch2')
             .relu(name='HG8_res13_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res13_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res13_batch3')
             .relu(name='HG8_res13_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res13_conv3'))

        (self.feed('HG8_add3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res13_skip'))

        (self.feed('HG8_res13_conv3',
                   'HG8_res13_skip')
         .add(name='HG8_res13'))


# upsample4
        (self.feed('HG8_res13')
             .upsample(64, 64, name='HG8_upSample4'))

# upsample4 + up1(Hg_res1)
        (self.feed('HG8_upSample4',
                   'HG8_res1')
         .add(name='HG8_add4'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Hourglass8  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###

################################# Hourglass8 postprocess #################

# id:025  Res14
        (self.feed('HG8_add4')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res14_batch1')
             .relu(name='HG8_res14_relu1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='HG8_res14_conv1')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res14_batch2')
             .relu(name='HG8_res14_relu2')
             .pad(np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
             .conv(3, 3, 128, 1, 1, biased=True, relu=False, name='HG8_res14_conv2')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_res14_batch3')
             .relu(name='HG8_res14_relu3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res14_conv3')

         )

        (self.feed('HG8_add4')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_res14_skip'))

        (self.feed('HG8_res14_conv3',
                   'HG8_res14_skip')
         .add(name='HG8_res14'))

# Linear layer to produce first set of predictions
# ll
        (self.feed('HG8_res14')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='HG8_linearfunc_conv1', padding='SAME')
             .tf_batch_normalization(is_training=is_training, activation_fn=None, name='HG8_linearfunc_batch1')
             .relu(name='HG8_linearfunc_relu'))

#************************************* Predicted heatmaps tmpOut ****************************#
# tmpOut
        (self.feed('HG8_linearfunc_relu')
             .conv(1, 1, n_classes, 1, 1, biased=True, relu=False, name='HG8_Heatmap', padding='SAME'))

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Hourglass8 postprocess ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^###
