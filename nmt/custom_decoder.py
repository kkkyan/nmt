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


__all__ = [
    "BasicDecoder_att"
]


class BasicDecoder_att(basic_decoder.BasicDecoder):
    def __init__(self, cell, helper, initial_state, output_layer=None):
        super(BasicDecoder_att,self).__init__(cell, helper, initial_state, output_layer=output_layer)

    def set_attention_values(self, memory, memory_sequence_length, check_inner_dims_defined=True):
        att_mes = self._cell._attention_mechanisms
        att_mes_len = len(att_mes)
        with tf.name_scope("AssignAttentionValues", nest.flatten(memory)):
            for i in range(att_mes_len):
                att_me = att_mes[i]
                assign_op = tf.assign(att_me._values, _prepare_memory(
                            memory, memory_sequence_length,
                            check_inner_dims_defined=check_inner_dims_defined))
                with tf.control_dependencies([assign_op]):
                    att_me._keys = tf.identity(att_me._keys)
