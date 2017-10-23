import tensorflow as tf
import numpy as np
import abc

from nodes import InputNode, NoisyNode, OutputNodeRegression, GPNode

class BaseLayer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getEnergyContribution(self):
        return 0.0

    @abc.abstractmethod
    def getOutput(self):
        return 0.0, 0.0

    def __init__(self, input_means, input_vars):
        self.input_means = input_means
        self.input_vars = input_vars

class InputLayer(BaseLayer):
    def __init__(self, data_placeholder):
        self.input_means = data_placeholder
        self.input_vars = tf.zeros_like(data_placeholder)
        BaseLayer.__init__(self, self.input_means, self.input_vars)
        
        self.input_node = InputNode(self.input_means, self.input_vars)
        self.output_means, self.output_vars = self.input_node.getOutput()

    def getEnergyContribution(self):
        return self.input_node.getEnergyContribution()

    def getOutput(self):
        return self.output_means, self.output_vars
    
class NoisyLayer(BaseLayer):
    def __init__(self, input_means, input_vars):
        BaseLayer.__init__(self, input_means, input_vars)
        
        self.noisy_node = NoisyNode(input_means, input_vars)
        self.output_means, self.output_vars = self.noisy_node.getOutput()

    def getEnergyContribution(self):
        return self.noisy_node.getEnergyContribution()

    def getOutput(self):
        return self.output_means, self.output_vars
    
class OutputLayerRegression(BaseLayer):
    def __init__(self, target_placeholder, input_means, input_vars):
        BaseLayer.__init__(self, input_means, input_vars)
        
        self.output_node = OutputNodeRegression(target_placeholder,
                                                input_means,
                                                input_vars)
        self.output_means, self.output_vars = self.output_node.getOutput()

    def getEnergyContribution(self):
        return self.output_node.getEnergyContribution()

    def getOutput(self):
        return self.output_means, self.output_vars

class GPLayer(BaseLayer):
    def __init__(self,
                 n_points,
                 n_inducing_points,
                 n_nodes,
                 input_means,
                 input_vars,
                 init_z,
                 set_for_training,
                 initial_layer,
                 final_layer,
                 linear_mean=None):
        BaseLayer.__init__(self, input_means, input_vars)
        self.nodes = []
        self.output_means_list = []
        self.output_vars_list = []

        for i in range(n_nodes):
            gp_node = GPNode(input_means,
                             input_vars,
                             n_points,
                             n_inducing_points,
                             init_z,
                             set_for_training,
                             initial_layer,
                             final_layer)
            output_mean, output_var = gp_node.getOutput()
            self.output_means_list.append(output_mean)
            self.output_vars_list.append(output_var)
            self.nodes.append(gp_node)
        
        if linear_mean is None or final_layer:
            mean_term = 0.0
        else:
            mean_term = tf.matmul(self.input_means,
                                  tf.constant(linear_mean, dtype=tf.float32))
        self.output_means = tf.concat(self.output_means_list, 1) + mean_term
        self.output_vars = tf.concat(self.output_vars_list, 1)

        self.energy = tf.add_n([n.getEnergyContribution() for n in self.nodes])

    def getEnergyContribution(self):
        return self.energy

    def getOutput(self):
        return self.output_means, self.output_vars