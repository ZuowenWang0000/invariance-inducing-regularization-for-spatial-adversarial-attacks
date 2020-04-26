"""
Implementation of a PGD attack bounded in L_infty.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class LinfPGDAttack:
  def __init__(self, model, config, epsilon, step_size, num_steps):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = config.random_start

    if config.loss_function == 'xent':
      loss = model.xent
    elif config.loss_function == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                                    - 1e4 * label_mask, axis=1)
      loss = wrong_logit - correct_logit
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]


  def perturb(self, x_nat, y, sess, trans=None):
    """
    Given a set of examples (x_nat, y), returns a set of adversarial
    examples within epsilon of x_nat in l_infinity norm. An optional
    spatial perturbations can be given as (trans_x, trans_y, rot).
    """

    if self.rand:
        x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
        x = np.copy(x_nat)

    if trans is None:
        trans = np.zeros([len(x_nat), 3])

    no_op = np.zeros([len(x_nat), 3])
    f_x_dict = {self.model.x_input: x,
                self.model.y_input: y,
                self.model.is_training: False,
                self.model.transform: no_op}

    f_x = sess.run(self.model.predictions, feed_dict=f_x_dict)

    for i in range(self.num_steps):
        curr_dict = {self.model.x_input: x,
                     self.model.y_input: f_x,
                     self.model.transform: trans,
                     self.model.is_training: False}
        grad = sess.run(self.grad, feed_dict=curr_dict)

        x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

        x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
        x = np.clip(x, 0, 255) # ensure valid pixel range

    return x


class SpatialPGDAttack:
  def __init__(self, model, config, epsilon, step_size, num_steps, 
               attack_limits=None):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = config.random_start
    self.limits = config.spatial_limits
    self.loss_function = config.loss_function
    # Attack parameters
    if attack_limits == None:
      self.limits = config.spatial_limits
    else:
      self.limits = attack_limits
    if config.loss_function == 'xent':
      loss = model.xent
    elif config.loss_function == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                                    - 1e4 * label_mask, axis=1)
      loss = wrong_logit - correct_logit
    elif config.loss_function == 'reg_kl':
      loss = model.reg_loss
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.transform)[0]


  def perturb(self, x_nat, y, sess):
    """
    Given a set of examples (x_nat, y), returns a set of adversarial
    examples within epsilon of x_nat in l_infinity norm. An optional
    spatial perturbations can be given as (trans_x, trans_y, rot).
    """
    # This is Tsipras code
    if self.rand:
      # For random restart
      n = len(x_nat)
      t = np.stack((np.random.uniform(-l, l, n) for l in self.limits),
                  axis=1)
    else:
      t = np.zeros([len(x_nat), 3])

    lim_arr = np.array(self.limits)

    if self.loss_function == "reg_kl":
      # x_input needs to contain but original exampes and adversarial ones
      # i.e. first transformations are noop, then adv. t
      n = len(x_nat)
      x_in = np.concatenate((x_nat, x_nat), axis=0)
      y = np.concatenate((y, y), axis=0)
      noop = np.zeros([n, 3])

      for i in range(self.num_steps):
          t_in = np.concatenate((noop, t), axis=0)
          curr_dict = {self.model.x_input: x_in,
                      self.model.y_input: y,
                      self.model.is_training: False,
                      self.model.transform: t_in}
          grad = sess.run(self.grad, feed_dict=curr_dict)
          grad_adv = grad[n:2*n]
          t = np.add(t, [self.step_size] * np.sign(grad_adv), out=t, casting='unsafe')
          t = np.clip(t, -lim_arr, lim_arr)
    else:
      for i in range(self.num_steps):
        curr_dict = {self.model.x_input: x_nat,
                     self.model.y_input: y,
                     self.model.is_training: False,
                     self.model.transform: t}
        grad = sess.run(self.grad, feed_dict=curr_dict)

        t = np.add(t, [self.step_size] * np.sign(grad), out=t, casting='unsafe')

        t = np.clip(t, -lim_arr, lim_arr)

    x = x_nat

    return x, t
