import tensorflow as tf
import numpy as np

class SquaredExponential(object):
    #Kzz
    @staticmethod
    def get_kernel(log_length_scale, log_scaling_factor, x, z):
        n_inducing_points = z.get_shape().as_list()[0]
        length_scale = tf.exp(log_length_scale)
        scaling_factor = tf.exp(log_scaling_factor)
        n_points = tf.shape(x)[0]
        x_z = tf.expand_dims(x, 1) - tf.expand_dims(z, 0)
        t1 = tf.reduce_sum(x_z * x_z / tf.expand_dims(length_scale, 1), 2)
        return scaling_factor * tf.exp(-0.5 * t1)

    # Expected value of the kernel Kxz or l (equation 33)
    @staticmethod
    def get_psi1(log_length_scale, log_scaling_factor, x_mean, x_var, z):
        n_inducing_points = z.get_shape().as_list()[0]
        length_scale = tf.exp(log_length_scale)
        scaling_factor = tf.exp(log_scaling_factor)

        lspxvar = length_scale + x_var
        const_t1 = length_scale / lspxvar
        const_t2 = tf.reduce_prod(tf.sqrt(const_t1), 1, keep_dims=True)
        x_z = tf.expand_dims(x_mean, 1) - tf.expand_dims(z, 0)
        t1 = tf.reduce_sum(x_z * x_z / tf.expand_dims(lspxvar, 1), 2)
        psi1 = scaling_factor * const_t2 * tf.exp(-0.5 * t1)
        return psi1

    # Equation 37 returns [N, n, n] this operation is crazy expensive
    @staticmethod
    def get_L(log_length_scale, log_scaling_factor, z, x_mean, x_var):
        length_scale = tf.exp(log_length_scale)
        scaling_factor = tf.exp(log_scaling_factor)
        b = length_scale / 2.0
        t1 = tf.reduce_prod(tf.sqrt(b / (b + x_var)), 1)
        t1 = tf.expand_dims(tf.expand_dims(t1, 1), 1)
        
        z_z = tf.expand_dims(z, 0) - tf.expand_dims(z, 1)
        ls_exp = tf.expand_dims(length_scale, 1)
        logt2 = tf.reduce_sum(z_z * z_z / (2.0 * ls_exp), 2)
        t2 = tf.expand_dims(tf.exp(-0.5 * logt2), 0)

        zpz = tf.expand_dims(z, 0) + tf.expand_dims(z, 1)
        x_exp = tf.expand_dims(tf.expand_dims(x_mean, 1), 1)
        x_zpz = x_exp - 0.5 * tf.expand_dims(zpz, 0)
        ls_exp_exp = tf.expand_dims(ls_exp, 0)
        x_var_exp = tf.expand_dims(tf.expand_dims(x_var, 1), 1)
        logt3 = tf.reduce_sum(x_zpz * x_zpz / (ls_exp_exp * 0.5 + x_var_exp),
                              3)
        t3 = tf.exp(-0.5 * logt3)

        return tf.square(scaling_factor) * t1 * t2 * t3
        