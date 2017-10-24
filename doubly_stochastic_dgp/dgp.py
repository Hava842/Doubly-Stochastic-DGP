# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:35:36 2017

@author: hrs13
"""

import tensorflow as tf
import numpy as np
from scipy.cluster.vq import kmeans2

from gpflow.param import Param, ParamList, Parameterized, AutoFlow, DataHolder
from gpflow.minibatch import MinibatchData
from gpflow.conditionals import conditional
from gpflow.model import Model
from gpflow.mean_functions import Linear, Zero
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.kullback_leiblers import gauss_kl_white
from gpflow._settings import settings

float_type = settings.dtypes.float_type

from utils import normal_sample, shape_as_list, tile_over_samples

class Node(Parameterized):
    def __init__(self, kern, q_mu, q_sqrt, Z):
        Parameterized.__init__(self)
        self.q_mu, self.q_sqrt, self.Z = Param(q_mu), Param(q_sqrt), Param(Z.copy())
        self.kern = kern
        
    def conditional(self, X, full_cov=False):
        mean, var = conditional(X, self.Z, self.kern,
                                self.q_mu, q_sqrt=self.q_sqrt,
                                full_cov=full_cov, whiten=True)
        
        return mean, var
        
    def KL(self):
        return gauss_kl_white(self.q_mu, self.q_sqrt)

class Layer(Parameterized):
    def __init__(self, kern, q_mu, q_sqrt, Z, mean_function):
        Parameterized.__init__(self)
        nodes = []
        for n_kern, n_q_mu, n_q_s in zip(kern, q_mu, q_sqrt):
            nodes.append(Node(n_kern, n_q_mu, n_q_s, Z))
        self.nodes = ParamList(nodes)
        self.mean_function = mean_function
    
    def conditional(self, X, full_cov=False):
        mv_list = [(n.conditional(X, full_cov)) for n in self.nodes]
        mean = tf.concat([m for m, v in mv_list], axis=1)
        var = tf.concat([v for m, v in mv_list], axis=1)
        return mean + self.mean_function(X), var
    
    def multisample_conditional(self, X, full_cov=False):
        f = lambda a: self.conditional(a, full_cov=full_cov)
        mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
        
        return tf.stack(mean), tf.stack(var)
        
    def KL(self):
        all_mu = tf.concat([n.q_mu for n in self.nodes], 1)
        all_sqrt = tf.concat([n.q_sqrt for n in self.nodes], 2)
        return gauss_kl_white(all_mu, all_sqrt)
        
def init_layers(X, dims_in, dims_out,
                M, final_inducing_points,
                share_inducing_inputs):
    q_mus, q_sqrts, mean_functions, Zs = [], [], [], []
    X_running = X.copy()
    
    for dim_in, dim_out in zip(dims_in[:-1], dims_out[:-1]):
        if dim_in == dim_out: # identity for same dims
            W = np.eye(dim_in)
        elif dim_in > dim_out: # use PCA mf for stepping down
            _, _, V = np.linalg.svd(X_running, full_matrices=False)
            W = V[:dim_out, :].T
        elif dim_in < dim_out: # identity + pad with zeros for stepping up
            I = np.eye(dim_in)
            zeros = np.zeros((dim_out - dim_in, dim_in))
            W = np.concatenate([I, zeros], 0).T

        mean_functions.append(Linear(A=W))
        Zs.append(kmeans2(X_running, M, minit='points')[0])
        if share_inducing_inputs:
            q_mus.append([np.zeros((M, dim_out))])
            q_sqrts.append([np.eye(M)[:, :, None] * np.ones((1, 1, dim_out))])
        else:
            q_mus.append([np.zeros((M, 1))] * dim_out)
            q_sqrts.append([np.eye(M)[:, :, None] * np.ones((1, 1, 1))] * dim_out)
         
        X_running = X_running.dot(W)

    # final layer (as before but no mean function)
    mean_functions.append(Zero())
    Zs.append(kmeans2(X_running, final_inducing_points, minit='points')[0])
    q_mus.append([np.zeros((final_inducing_points, 1))])
    q_sqrts.append([np.eye(final_inducing_points)[:, :, None] * np.ones((1, 1, 1))])

    return q_mus, q_sqrts, Zs, mean_functions


class DGP(Model):
    def __init__(self, X, Y,
                 inducing_points,
                 final_inducing_points,
                 hidden_units,
                 units,
                 share_inducing_inputs=True):
        Model.__init__(self)

        assert X.shape[0] == Y.shape[0]
        
        self.num_data, D_X = X.shape
        self.D_Y = 1
        self.num_samples = 100
        
        kernels = []
        for l in range(hidden_units+1):
            ks = []
            if (l > 0):
                D = units
            else:
                D = D_X
            if (l < hidden_units):
                for w in range(units):
                    ks.append(RBF(D, lengthscales=1., variance=1.) + White(D, variance=1e-5))
            else:
                ks.append(RBF(D, lengthscales=1., variance=1.))
            kernels.append(ks)                

        self.dims_in = [D_X] + [units] * hidden_units
        self.dims_out = [units] * hidden_units + [1]
        q_mus, q_sqrts, Zs, mean_functions = init_layers(X,
                                                         self.dims_in,
                                                         self.dims_out,
                                                         inducing_points,
                                                         final_inducing_points,
                                                         share_inducing_inputs)
                                                         
        layers = []
        for q_mu, q_sqrt, Z, mean_function, kernel in zip(q_mus, q_sqrts, Zs, 
                                                          mean_functions, 
                                                          kernels):
            layers.append(Layer(kernel, q_mu, q_sqrt, Z, mean_function))
        self.layers = ParamList(layers)
        
        for layer in self.layers[:-1]: # fix the inner layer mean functions 
            layer.mean_function.fixed = True
            
        self.likelihood = Gaussian()
        
        minibatch_size = 10000 if X.shape[0] > 10000 else None 
        if minibatch_size is not None:
            self.X = MinibatchData(X, minibatch_size)
            self.Y = MinibatchData(Y, minibatch_size)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

    def propagate(self, X, full_cov=False, S=1):
        Fs = [tile_over_samples(X, S), ]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            mean, var = layer.multisample_conditional(Fs[-1], full_cov=full_cov)
            F = normal_sample(mean, var, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars # don't return Fs[0] as this is just X

    def build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov, S)
        return Fmeans[-1], Fvars[-1]
    
    def build_likelihood(self):
        Fmean, Fvar = self.build_predict(self.X, full_cov=False, S=self.num_samples)
        S, N, D = shape_as_list(Fmean)
        Y = tile_over_samples(self.Y, self.num_samples)
        
        f = lambda a: self.likelihood.variational_expectations(a[0], a[1], a[2])
        var_exp = tf.map_fn(f, (Fmean, Fvar, Y), dtype=float_type)
        var_exp = tf.stack(var_exp) #SN
        
        var_exp = tf.reduce_mean(var_exp, 0) # S,N -> N. Average over samples
        L = tf.reduce_sum(var_exp) # N -> scalar. Sum over data (minibatch)

        KL = 0.
        for layer in self.layers:
            KL += layer.KL()

        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self.build_predict(Xnew, full_cov=False, S=num_samples)
    
    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        return self.build_predict(Xnew, full_cov=True, S=1)
    
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)[0]

    @AutoFlow((float_type, [None, None]))
    def predict_all_layers_full_cov(self, Xnew):
        return self.propagate(Xnew, full_cov=True, S=1)[0]
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self.build_predict(Xnew, full_cov=False, S=num_samples)
        S, N, D = shape_as_list(Fmean)
        flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar]]
        Y_mean, Y_var = self.likelihood.predict_mean_and_var(*flat_arrays)
        return [tf.reshape(a, [S, N, self.D_Y]) for a in [Y_mean, Y_var]]
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self.build_predict(Xnew, full_cov=False, S=num_samples)
        S, N, D = shape_as_list(Fmean)
        Ynew = tile_over_samples(Ynew, num_samples)
        flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar, Ynew]]
        l_flat = self.likelihood.predict_density(*flat_arrays)
        l = tf.reshape(l_flat, [S, N, -1])
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)








