import time
import tensorflow as tf
import numpy as np

from scipy.cluster.vq import kmeans2

from layers import InputLayer, OutputLayerRegression, NoisyLayer, GPLayer

class GPNetwork(object):
    # The training_targets are the Y's which are real numbers
    def __init__(self, 
                 training_data,
                 training_targets,
                 transform_targets=True,
                 layer_shape=None,
                 linear_mean=False):
        self.training_data = training_data
        self.training_targets = training_targets
        self.target_mean = np.mean(self.training_targets, axis=0)
        self.target_std = np.std(self.training_targets, axis=0)
        self.transform_targets = transform_targets
        if transform_targets:
            self.training_targets = (self.training_targets - self.target_mean) / self.target_std
        self.n_points = training_data.shape[0]
        self.input_d = training_data.shape[1]
        self.output_d = training_targets.shape[1]
        self.linear_mean = linear_mean
        
        if layer_shape is None:
            layer_shape = [(5, 30), (self.output_d, 100)]
            
            
        self.n_points_tf = tf.Variable(self.n_points, 
                                       trainable=False,
                                       dtype=tf.float32)
        self.set_for_training = tf.Variable(1.0,
                                            trainable=False,
                                            dtype=tf.float32)
        self.data_placeholder = tf.placeholder(tf.float32,
                                               [None, self.input_d])
        self.target_placeholder = tf.placeholder(tf.float32,
                                                 [None, self.output_d])
                        
        self.layers = []
        self.addInputLayer()
        data_running = training_data.copy()
        dim_in = self.input_d
        for l, shape in enumerate(layer_shape):
            print('Layer {0}'.format(l))
            if self.linear_mean:
                init_z = kmeans2(data_running, shape[1], minit='points')[0]
                if dim_in == shape[0]: # identity for same dims
                    W = np.eye(dim_in)
                elif dim_in > shape[0]: # use PCA mf for stepping down
                    _, _, V = np.linalg.svd(data_running, full_matrices=False)
                    W = V[:shape[0], :].T
                elif dim_in < shape[0]: # identity + pad with zeros for stepping up
                    I = np.eye(dim_in)
                    zeros = np.random.uniform(-0.01, 0.01, [shape[0] - dim_in, dim_in])
                    W = np.concatenate([I, zeros], 0).T
                data_running = data_running.dot(W)
                if l == len(layer_shape)-1: # No mean for the final layer
                    W = np.zeros([dim_in, shape[0]])
            else:
                if l==0:
                    init_z = kmeans2(data_running, shape[1], minit='points')[0]
                else:
                    init_z = np.random.uniform(-1, 1, [shape[1], dim_in])
                W = None
            dim_in = shape[0]
            
            self.addGPLayer(shape[1],
                            shape[0],
                            init_z=init_z,
                            initial_layer=(l == 0),
                            final_layer=(l == len(layer_shape)-1),
                            linear_mean=W)
            self.addNoisyLayer()
        self.addOutputLayerRegression()

        layer_energies = [l.getEnergyContribution() for l in self.layers]
        self.energy = tf.add_n(layer_energies)
        
        self.learning_rate = tf.Variable(0.01, trainable=False, dtype=tf.float32)
        adam = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = adam.minimize(-self.energy)
        
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(init_op) 

    def predictBatch(self, test_data):
        self.session.run(self.set_for_training.assign(0.0))
        fd = {self.data_placeholder:test_data}
        m, v = self.session.run((self.output_mean, self.output_var),
                                feed_dict=fd)
        if self.transform_targets:
            m = m * self.target_std + self.target_mean
            v = v * self.target_std * self.target_std
        return m, v
    def predictBatchDensity(self, test_data, test_targets):
        if self.transform_targets:
            test_targets = (test_targets - self.target_mean) / self.target_std
        self.session.run(self.set_for_training.assign(0.0))
        fd = {self.data_placeholder:test_data,
              self.target_placeholder:test_targets}
        return self.session.run((self.density),
                                feed_dict=fd)
        

    def addInputLayer(self):
        assert len(self.layers) == 0

        self.layers.append(InputLayer(self.data_placeholder))

    def addNoisyLayer(self):
        assert len(self.layers) != 0

        means, vars = self.layers[-1].getOutput()
        new_layer = NoisyLayer(means, vars)
        self.layers.append(new_layer)

    def addGPLayer(self, n_inducing_points, n_nodes, init_z, initial_layer, final_layer, linear_mean=None):
        assert len(self.layers) != 0

        means, vars = self.layers[-1].getOutput()                
        new_layer = GPLayer(self.n_points_tf,
                            n_inducing_points,
                            n_nodes,  
                            means,
                            vars,
                            init_z,
                            self.set_for_training,
                            initial_layer,
                            final_layer,
                            linear_mean)
        self.layers.append(new_layer)

    def addOutputLayerRegression(self):
        assert len(self.layers) != 0

        means, vars = self.layers[-1].getOutput()
        new_layer = OutputLayerRegression(self.target_placeholder,
                                          means,
                                          vars)
        self.layers.append(new_layer)
        self.output_mean, self.output_var = new_layer.getOutput()   
        if self.transform_targets:
            unnorm_output_mean = self.output_mean * self.target_std + self.target_mean
            unnorm_output_var = self.output_var * self.target_std * self.target_std
            unnorm_target = self.target_placeholder * self.target_std + self.target_mean
        else:
            unnorm_target = self.target_placeholder
        output_distribution = tf.contrib.distributions.Normal(unnorm_output_mean,
                                                              tf.sqrt(unnorm_output_var))
        
        logprob = output_distribution.log_prob(unnorm_target)
        self.density = logprob
        
    def train(self, maxiter=200, learning_rate=0.001, minibatch_size=1000): 
        self.learning_rate.assign(learning_rate)
        self.session.run(self.set_for_training.assign(1.0))

        n_batches = int(np.ceil(1.0 * self.n_points / minibatch_size))
        for iter in range(maxiter):
            shuffle = np.random.permutation(self.n_points)
            training_data = self.training_data[ shuffle, : ]
            training_targets = self.training_targets[ shuffle, : ]
            start_epoch  = time.time()
            epoch_energy = 0.0
            for i in range(n_batches):
                start_i = i * minibatch_size
                end_i = min((i + 1) * minibatch_size, self.n_points)
                minibatch_data = training_data[start_i : end_i, : ]
                minibatch_targets = training_targets[start_i : end_i, : ]
                fd = {self.data_placeholder:minibatch_data,
                      self.target_placeholder:minibatch_targets}

                _, e = self.session.run((self.optimizer, self.energy),
                                        feed_dict=fd)
                epoch_energy += e

            if (iter % 10 == 0):
                print('Epoch: {}, - Energy: {} Time: {}'
                        .format(iter, epoch_energy, time.time() - start_epoch))
