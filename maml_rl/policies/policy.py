from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras as keras


def weight_init(module):
    if isinstance(module, keras.layers.Dense):
        keras.initializers.glorot_uniform(module.weight)
        module.bias.data.zero_()


class Policy(tf.keras.Model):
    def __init__(self, input_size, output_size, name=None):
        """
        Base class for the policies.

        Args:
            input_size:     Shape of the observations
            output_size:    Shape of the resulting actions
            name:           Name of the scope this policy created in
        """
        super(Policy, self).__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size
        self.all_params = OrderedDict()

    def update_params(self, grads, step_size=0.1):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.

        Arguments:
            grads:      The gradients calculated w.r.t the parameters
            step_size:  Learning rate
        """
        updated_params = OrderedDict()
        params_with_name = [(x.name, x) for x in self.get_trainable_variables()]
        for (name, param), grad in zip(params_with_name, grads):
            updated_params[name] = tf.subtract(param, tf.multiply(step_size, grad))

        return updated_params

    def set_params_with_name(self, var_list):
        old_var_list = self.get_trainable_variables()
        for (name, var), old_var in zip(var_list.items(), old_var_list):
            old_var.assign(var)

    def set_params(self, var_list):
        old_var_list = self.get_trainable_variables()
        for var, old_var in zip(var_list, old_var_list):
            old_var.assign(var)

    def get_trainable_variables(self):
        return NotImplementedError

    def __call__(self, x, params=None):
        return self.forward(x, params)

    def forward(self, x, params=None):
        raise NotImplementedError
