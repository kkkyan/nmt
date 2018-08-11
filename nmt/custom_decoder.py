"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# pylint: disable=E0611
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _prepare_memory


__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder_att",
]


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output","before_output_layer","sample_id"))):
  pass

class BasicDecoder_att(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, helper_bak, initial_state, output_layer=None):
    """Initialize BasicDecoder.
    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    rnn_cell_impl.assert_like_rnncell("cell", cell)
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))

    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))

    self._cell = cell
    self._helper = helper
    self._helper_bak = helper_bak
    self._initial_state = initial_state
    self._output_layer = output_layer
    
  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer.compute_output_shape(
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  def _before_output_size(self):
    return self._cell.output_size

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        before_output_layer=self._before_output_size(),
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and the sample_ids_dtype from the helper.
    dtype = nest.flatten(self._initial_state)[0].dtype

    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        nest.map_structure(lambda _: dtype, self._before_output_size()),
        self._helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.
    Args:
      name: Name scope for any created operations.
    Returns:
      `(finished, first_inputs, initial_state)`.
    """

    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.
    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.
    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      before_output_layer, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(before_output_layer)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)

      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)

    outputs = BasicDecoderOutput(cell_outputs, before_output_layer, sample_ids)
    return (outputs, next_state, next_inputs, finished)

  def set_attention_values(self, memory, memory_sequence_length, check_inner_dims_defined=True):
    att_mes = self._cell._attention_mechanisms
    att_mes_len = len(att_mes)
    tensors = []
    with tf.name_scope("AssignAttentionValues", nest.flatten(memory)):
        for i in range(att_mes_len):
            att_me = att_mes[i]
            att_me._values =  _prepare_memory(
                        memory, memory_sequence_length,
                        check_inner_dims_defined=check_inner_dims_defined)
            att_me._keys = (
                    att_me.memory_layer(att_me._values) if att_me.memory_layer  # pylint: disable=not-callable
                    else att_me._values)
            
            tensors.append(att_me._values)
            tensors.append(att_me._keys)
    
    return tensors