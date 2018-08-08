"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math
import tensorflow as tf

import numpy as np
# pylint: disable-msg=E0611
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _prepare_memory
from tensorflow.contrib.framework import nest
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl

__all__ = [
    "BasicDecoder_att"
]


class BasicDecoder_att(basic_decoder.BasicDecoder):
    def __init__(self, cell, helper, initial_state, output_layer=None):
        self.decoder_output_size = cell._attention_mechanisms[-1].get_shape()[-1].value
        super(BasicDecoder_att,self).__init__(cell, helper, initial_state, output_layer=output_layer)

    def set_attention_values(self, memory, memory_sequence_length, check_inner_dims_defined=True):
        att_mes = self._cell._attention_mechanisms
        att_mes_len = len(att_mes)
        with tf.name_scope("AssignAttentionValues", nest.flatten(memory)):
            for i in range(att_mes_len):
                att_me = att_mes[i]
                att_me._values =  _prepare_memory(
                            memory, memory_sequence_length,
                            check_inner_dims_defined=check_inner_dims_defined)
                att_me._keys = (
                        att_me.memory_layer(att_me._values) if att_me.memory_layer  # pylint: disable=not-callable
                        else att_me._values)

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
        print("this is new step")
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
          origin_cell_outputs, cell_state = self._cell(inputs, state)
          if self._output_layer is not None:
            cell_outputs = self._output_layer(origin_cell_outputs)
          sample_ids = self._helper.sample(
              time=time, outputs=cell_outputs, state=cell_state)
          (finished, next_inputs, next_state) = self._helper.next_inputs(
              time=time,
              outputs=cell_outputs,
              state=cell_state,
              sample_ids=sample_ids)
        outputs = basic_decoder.BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished, origin_cell_outputs)
