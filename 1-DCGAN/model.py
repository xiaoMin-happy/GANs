from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return math.ceil(float(size) / float(stride))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    # 定义DCGAN对象与网络结构相关的超参数属性
    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = sample_num   # 训练中测试用

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim             # 噪音向量大小

    self.gf_dim = gf_dim           # filter的基数，即输出通道基数，设置filter的个数时，在这个基础上乘倍即可
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim             #最终输出图像通道数

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')   # BN在卷积后，激活函数前
    self.d_bn2 = batch_norm(name='d_bn2')   # 对于判别网络共有3层，则需要2个bn层，这里先设定好bn层空间，之后直接调用即可

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')   # 对于生成网络共有4层，则需要3个bn层
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name        # 数据集名称
    self.input_fname_pattern = input_fname_pattern  # 输入图片格式
    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    # 定义要feed的数据：大小及数据格式
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    if self.is_crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_height, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')   # 真实的输入四维[B,H,W,C] +相当于extend
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs') # 生成的图片

    inputs = self.inputs
    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')          # 噪音向量
    self.z_sum = histogram_summary("z", self.z)          # tensorflow中的数据格式通常设置为tf.float32（GPU和CPU均支持）


    # 网络结构的定义：G和D模块组合
    if self.y_dim:
      self.G = self.generator(self.z, self.y)
      self.D, self.D_logits = \
          self.discriminator(inputs, self.y, reuse=False)

      self.sampler = self.sampler(self.z, self.y)
      self.D_, self.D_logits_ = \
          self.discriminator(self.G, self.y, reuse=True)
    else:
      self.G = self.generator(self.z)
      self.D, self.D_logits = self.discriminator(inputs)     # 注意：D有两个，一个输入是真实数据，一个输入是生成数据

      self.sampler = self.sampler(self.z)
      self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)


    # loss的定义：有几个优化对象，就有几个loss
    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(self.D), logits=self.D_logits))  # tf.ones_like(tensor): 给出与tensor维度,数据类型一致的全0张量
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(self.D_), logits=self.D_logits_))
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(self.D_), logits=self.D_logits_))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    # 提取d和g的训练参数
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()


  #定义训练过程
  def train(self, config):
    """Train DCGAN"""
    if config.dataset == 'mnist':
      data_X, data_y = self.load_mnist()
    else:
      #glob.glob是返回所有匹配的文件路径的列表
      data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
    #np.random.shuffle(data)


    # 定义训练的优化器：
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    #得到测试的输入sample_z
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))  # 造出batch的噪音向量

    # 得到测试数据sample，所以之前要读一遍数据
    if config.dataset == 'mnist':
      sample_inputs = data_X[0:self.sample_num]
      sample_labels = data_y[0:self.sample_num]
    else:
      sample_files = data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    is_crop=self.is_crop,
                    is_grayscale=self.is_grayscale) for sample_file in sample_files]
      if (self.is_grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):  # 判断是否之前训练过模型则
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):

      # 为训练再读取一边数据
      if config.dataset == 'mnist':
        batch_idxs = min(len(data_X), config.train_size) // config.batch_size
      else:      
        data = glob(os.path.join(
          "./data", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(data), config.train_size) // config.batch_size   # 多少个batch

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        is_crop=self.is_crop,
                        is_grayscale=self.is_grayscale) for batch_file in batch_files]
          if (self.is_grayscale):
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        # 训练输入的batch的z
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z, self.y:batch_labels })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        #  每训练100次，利用sampler模块(reuse G)，使用sample_data测试一次模型以及生成相应结果
        if np.mod(counter, 100) == 1:
          if config.dataset == 'mnist':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            save_images(samples, [8, 8],
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, [8, 8],
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
              print("one pic error!...")
        # 每100次保存一次模型
        if np.mod(counter, 200) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:                                                    # 64*64*3
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))            # 32*32*64
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # 16*16*128
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'))) # 8*8*256
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'))) # 4*4*512
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')   # 全连接层

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  # G模块搭建
  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      # 反卷积前，先把每层输出特征图大小计算出来。除法操作，计算特征图大小，从上到下，从大到小
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width                       # 64*64
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)    # 32*32
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 16*16
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)  # 8*8
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)# 4*4

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))                                            # 4*4*512

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True) # 8*8*256
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True) # 16*16*128
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True) # 8*8*64
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)      # 16*16*3

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))


  # sampler相当于训练过程中测试时的G，reuse了G
  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
