# encoding:utf-8

'''
    LSTM Matching Network
    Match between Q and S
    h_t, C_t = LSTM(Q, [h_(t-1), S], C_(t-1))
    with input Q, hidden state [h_(t-1), S], and cell state C_t

    This is a keras version
'''

import numpy as np 
import tensorflow as tf
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.layers import Layer, RNN
from tensorflow.python.keras import initializers, activations

class MinimalRNNCell(Layer):
    def __init__(self, units, **kwargs):
        self.units = units 
        self.state_size = units 
        super(MinimalRNNCell, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            dtype=tf.float32, trainable=True,
            name="kernel")
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer="glorot_uniform",
            dtype=tf.float32, trainable=True,
            name="recurrent_kernel")                        
        super(MinimalRNNCell, self).build(input_shape)
    
    def call(self, inputs, states):        
        prev_output = states[0]
        h = tf.tensordot(inputs, self.kernel, axes=(-1, 0))
        output = h + tf.tensordot(prev_output, self.recurrent_kernel, axes=(-1, 0))
        return output, [output]

class MinimalLSTMCell(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        # Control the output size
        self.state_size = [units, units]
        self.output_size = units

        self.activation = activations.get("tanh")
        self.recurrent_activation = activations.get("hard_sigmoid")
        super(MinimalLSTMCell, self).__init__(**kwargs)
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer="glorot_uniform",
            dtype=tf.float32, trainable=True,
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.units * 4, ),
            initializer="Zeros",
            dtype=tf.float32, trainable=True,
            name="bias"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer="glorot_uniform",
            dtype=tf.float32, trainable=True,
            name="recurrent_kernel"
        )
        super(MinimalLSTMCell, self).build(input_shape)
    
    def call(self, inputs, states):
        h_tm1, c_tm1 = states
         
        inputs_i, inputs_f, inputs_c, inputs_o = inputs, inputs, inputs, inputs
        W_xi, W_xf, W_xc, W_xo = tf.split(self.kernel, num_or_size_splits=4, axis=1)
        b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1, h_tm1, h_tm1, h_tm1

        x_i = tf.nn.bias_add(tf.tensordot(inputs_i, W_xi, axes=(-1, 0)), b_i)
        x_f = tf.nn.bias_add(tf.tensordot(inputs_f, W_xf, axes=(-1, 0)), b_f)
        x_c = tf.nn.bias_add(tf.tensordot(inputs_c, W_xc, axes=(-1, 0)), b_c)
        x_o = tf.nn.bias_add(tf.tensordot(inputs_o, W_xo, axes=(-1, 0)), b_o)

        i = self.recurrent_activation(x_i + tf.tensordot(h_tm1_i, self.recurrent_kernel[:, :self.units], axes=(-1, 0)) )
        f = self.recurrent_activation(x_f + tf.tensordot(h_tm1_f, self.recurrent_kernel[:, self.units: self.units * 2], axes=(-1, 0)) )
        c = f * c_tm1 + i * self.activation(x_c + tf.tensordot(h_tm1_c, self.recurrent_kernel[:, self.units * 2: self.units * 3], axes=(-1, 0)) )
        o = self.recurrent_activation(x_o + tf.tensordot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:], axes=(-1, 0)) )

        h = o * self.activation(c)

        return h, [h, c]

from tensorflow.python.training.tracking import data_structures
class CustomLSTMCell(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        # Control the output size
        self.state_size = [units, units]
        self.output_size = units

        self.activation = activations.get("tanh")
        self.recurrent_activation = activations.get("hard_sigmoid")

        super(CustomLSTMCell, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # (input_dim + 1 + units * 2) * (units * 4)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer="glorot_uniform",
            dtype=tf.float32, trainable=True,
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.units * 4, ),
            initializer="Zeros",
            dtype=tf.float32, trainable=True,
            name="bias"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer="glorot_uniform",
            dtype=tf.float32, trainable=True,
            name="recurrent_kernel"
        )
        self.additional_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer="glorot_uniform",
            dtype=tf.float32, trainable=True,
            name="additional_kernel"
        )
        super(CustomLSTMCell, self).build(input_shape)
    
    def call(self, inputs, states, additional_states):
        h_tm1, c_tm1 = states
        s = additional_states
         
        inputs_i, inputs_f, inputs_c, inputs_o = inputs, inputs, inputs, inputs
        W_xi, W_xf, W_xc, W_xo = tf.split(self.kernel, num_or_size_splits=4, axis=1)
        b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1, h_tm1, h_tm1, h_tm1

        x_i = tf.nn.bias_add(tf.tensordot(inputs_i, W_xi, axes=(-1, 0)), b_i)
        x_f = tf.nn.bias_add(tf.tensordot(inputs_f, W_xf, axes=(-1, 0)), b_f)
        x_c = tf.nn.bias_add(tf.tensordot(inputs_c, W_xc, axes=(-1, 0)), b_c)
        x_o = tf.nn.bias_add(tf.tensordot(inputs_o, W_xo, axes=(-1, 0)), b_o)

        i = self.recurrent_activation(x_i \
            + tf.tensordot(h_tm1_i, self.recurrent_kernel[:, :self.units], axes=(-1, 0)) \
            + tf.tensordot(s, self.additional_kernel[:, :self.units], axes=(-1, 0)) )
        f = self.recurrent_activation(x_f \
            + tf.tensordot(h_tm1_f, self.recurrent_kernel[:, self.units: self.units * 2], axes=(-1, 0)) \
            + tf.tensordot(s, self.additional_kernel[:, self.units: self.units * 2], axes=(-1, 0)) )
        c = f * c_tm1 + i * self.activation(x_c \
            + tf.tensordot(h_tm1_c, self.recurrent_kernel[:, self.units * 2: self.units * 3], axes=(-1, 0)) \
            + tf.tensordot(s, self.additional_kernel[:, self.units * 2: self.units * 3], axes=(-1, 0)) )
        o = self.recurrent_activation(x_o \
            + tf.tensordot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:], axes=(-1, 0)) \
            + tf.tensordot(s, self.additional_kernel[:, self.units * 3:], axes=(-1, 0)) )

        h = o * self.activation(c)

        return h, [h, c]


from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.layers.recurrent import StackedRNNCells
from tensorflow.python.keras.engine.input_spec import InputSpec

class CustomRNN(tf.keras.layers.RNN):
    def __init__(self, **kwargs):
        super(CustomRNN, self).__init__(**kwargs)
    
    def __call__(self, inputs, additional_state, initial_state=None, constants=None, **kwargs):

        inputs, initial_state, constants = _standardize_args(inputs,
                                                             initial_state,
                                                             constants,
                                                             self._num_constants)

        if initial_state is None and constants is None:
            # return super(CustomRNN, self).__call__([inputs, additional_state], **kwargs)
            return Layer.__call__(self, inputs, additional_state, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            additional_inputs += initial_state
            self.state_spec = nest.map_structure(
                lambda s: InputSpec(shape=K.int_shape(s)), initial_state)
            additional_specs += self.state_spec
        if constants is not None:
            additional_inputs += constants
            self.constants_spec = [
                InputSpec(shape=K.int_shape(constant)) for constant in constants
            ]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # additional_inputs can be empty if initial_state or constants are provided
        # but empty (e.g. the cell is stateless).
        flat_additional_inputs = nest.flatten(additional_inputs)
        is_keras_tensor = K.is_keras_tensor(
            flat_additional_inputs[0]) if flat_additional_inputs else True
        for tensor in flat_additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            if self.built:
                # Keep the input_spec since it has been populated in build() method.
                full_input_spec = self.input_spec + additional_specs
            else:
                # The original input_spec is None since there could be a nested tensor
                # input. Update the input_spec to match the inputs.
                full_input_spec = generic_utils.to_list(
                    nest.map_structure(lambda _: None, inputs)) + additional_specs
            # Perform the call with temporarily replaced input_spec
            self.input_spec = full_input_spec
            output = super(CustomRNN, self).__call__(full_input, **kwargs)
            # Remove the additional_specs from input spec and keep the rest. It is
            # important to keep since the input spec was populated by build(), and
            # will be reused in the stateful=True.
            self.input_spec = self.input_spec[:-len(additional_specs)]
            return output
        else:
            if initial_state is not None:
                kwargs['initial_state'] = initial_state
            if constants is not None:
                kwargs['constants'] = constants
            return super(CustomRNN, self).__call__(inputs, **kwargs)
    
    def call(self,
             inputs,
             additional_states,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.

        inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
        is_ragged_input = (row_lengths is not None)
        self._validate_args_if_ragged(is_ragged_input, mask)

        inputs, initial_state, constants = self._process_inputs(
            inputs, initial_state, constants)

        self._maybe_reset_cell_dropout_mask(self.cell)
        if isinstance(self.cell, StackedRNNCells):
            for cell in self.cell.cells:
                self._maybe_reset_cell_dropout_mask(cell)

        if mask is not None:
            # Time step masks must be the same for each input.
            # TODO(scottzhu): Should we accept multiple different masks?
            mask = nest.flatten(mask)[0]

        if nest.is_nested(inputs):
            # In the case of nested input, use the first element for shape check.
            input_shape = K.int_shape(nest.flatten(inputs)[0])
        else:
            input_shape = K.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]
        if self.unroll and timesteps is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        if generic_utils.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        # TF RNN cells expect single tensor as state instead of list wrapped tensor.
        is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
        # Use the __call__ function for callable objects, eg layers, so that it
        # will have the proper name scopes for the ops, etc.
        cell_call_fn = self.cell.__call__ if callable(self.cell) else self.cell.call
        if constants:
            if not generic_utils.has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
                states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type

                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                # output, new_states = cell_call_fn(
                #     inputs, states, constants=constants, **kwargs)
                output, new_states = cell_call_fn(
                    inputs, states, additional_states, constants=constants, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return output, new_states
        else:

            def step(inputs, states):
                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                # output, new_states = cell_call_fn(inputs, states, **kwargs)
                output, new_states = cell_call_fn(inputs, states, additional_states, **kwargs)
                if not nest.is_nested(new_states):
                    new_states = [new_states]
                return output, new_states
        
        # inputs = [inputs, additional_states]
        last_output, outputs, states = K.rnn(
            step,
            inputs,
            initial_state,
            constants=constants,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=row_lengths if row_lengths is not None else timesteps,
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask)

        if self.stateful:
            updates = [
                state_ops.assign(self_state, state) for self_state, state in zip(
                    nest.flatten(self.states), nest.flatten(states))
            ]
            self.add_update(updates)

        if self.return_sequences:
            output = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
        else:
            output = last_output

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return generic_utils.to_list(output) + states
        else:
            return output 

class CustomLSTM(tf.keras.layers.RNN):

    def __init__(self, units, **kwargs):
        cell = MinimalLSTMCell(units)
        super(CustomLSTM, self).__init__(cell)
        self.input_spect = [tf.keras.layers.InputSpec(ndim=3)]
    
    def call(self, inputs):
        return super(CustomLSTM, self).call(inputs)

class LSTMMatcher(Layer):

    def __init__(self, **kwargs):
        super(LSTMMatcher, self).__init__(**kwargs) 

    def build(self, input_shape):
        return super(LSTMMatcher, self).build(input_shape)

    def call(self, inputs, steps, **kwargs):
        '''
        inputs:
            S: bs x dim
            Q: bs x dim
        '''
        assert len(inputs) == 2
        assert steps > 0
        S, Q = inputs

        Q = tf.reshape(tf.tile(Q, multiples=(1, steps)), shape=(-1, steps, tf.shape(Q)[-1])) # bs x steps x dim

        return None  

    def get_config(self):
        return super(LSTMMatcher, self).get_config()



# # encoding:utf-8

# import tensorflow as tf 
# from tensorflow.python.keras.layers import Layer
# from tensorflow.python.keras.layers.recurrent import LSTMCell, GRUCell, SimpleRNNCell, StackedRNNCells

# class Matcher(Layer):

#     def __init__(self,
#                  units = [1, ],
#                  cell_type="lstm",
#                  steps = 2,
#                  **kwargs):

#         super(Matcher, self).__init__(**kwargs)

#         self.cell_type = cell_type
#         self.steps = steps
#         self.units = units # array
        
    
#     def build(self, input_shape):
        
#         super(Matcher, self).build(input_shape)

#         if self.cell_type.lower() == "lstm":
#             self.core_cell = [tf.compat.v1.nn.rnn_cell.LSTMCell(units) for units in self.units]
#             self.cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(self.core_cell)
#         else:
#             raise ValueError("bad cell type=%s" % self.cell_type)

#     def call(self, inputs):

#         '''
#         inputs:
#             s: N x dim
#             q: N x dim
#         '''
#         s, q = inputs
#         batch_size = tf.shape(inputs[0])[0]

#         eos_time_slice = tf.ones_like(inputs[0], dtype=tf.float32, name="eos")
#         pad_time_slice = tf.zeros_like(inputs[0], dtype=tf.float32, name="pad")

#         iteration_steps = tf.multiply(tf.ones((batch_size,)), self.steps)
#         iteration_steps = tf.cast(iteration_steps, dtype=tf.int32)

#         def loop_fn_initial():

#             initial_elements_finished = (iteration_steps <= 0) # All Flase
#             initial_input = q

#             # initial_cell_state = [tf.concat([q, s], axis=1)]
#             initial_cell_state = [tf.concat([q, s], axis=1)]
#             for i in range(1, len(self.units)):
#                 initial_cell_state.append(self.core_cell[i].zero_state(batch_size, dtype=tf.float32) )
            
#             return (initial_elements_finished,
#                     initial_input,
#                     tuple(initial_cell_state),
#                     None, None)
        
#         def loop_fn_transition(time, cell_output, cell_state, loop_state):

#             _elements_finished = (iteration_steps <= time)

#             _finished = tf.reduce_all(_elements_finished)
#             _inputs = tf.cond(_finished, lambda:pad_time_slice, q )

#             _states = tf.concat()
#             _outputs = cell_output
#             _loop_state = None
#             return (_elements_finished,
#                     _inputs,
#                     _states,
#                     _outputs,
#                     _loop_state)
        
#         def loop_fn(time, cell_output, cell_state, loop_state):
#             if cell_state is None:
#                 return loop_fn_initial()
#             else:
#                 return loop_fn_transition(time, cell_output, cell_state, loop_state)

#         # with tf.variable_scope("matcher"):
#         outputs_ta, final_state, _ = tf.compat.v1.nn.raw_rnn(self.cells, loop_fn)
#         outputs = outputs_ta.stack()
        
#         return outputs, final_state
            
#     def get_config(self):
#         config = {
#         }
#         base_config = super(Matcher, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
            


