from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

# pylint: disable-msg=E0611
import tensorflow as tf 
from tensorflow.contrib.seq2seq.python.ops.decoder import Decoder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.decoder import _create_zero_outputs


__all__ = ["dynamic_bidecode", "dynamic_bidecode_att", "dynamic_decode"]


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access
def dynamic_bidecode(fw_decoder, bw_decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None,return_seq=False):
  """Perform dynamic decoding with `bidecoder`."""
  if not isinstance(fw_decoder, Decoder):
    raise TypeError("Expected fw_decoder to be type Decoder, but saw: %s" %
                 type(fw_decoder))

  if not isinstance(bw_decoder, Decoder):
    raise TypeError("Expected bw_decoder to be type Decoder, but saw: %s" %
                 type(bw_decoder))

  with tf.variable_scope(scope,"bi_decoder") as scope:
    # Forward
    with tf.variable_scope("fw") as fw_scope:
      fw_final_outputs, fw_final_state, fw_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
          fw_decoder, output_time_major=output_time_major, 
          impute_finished=impute_finished, 
          maximum_iterations=maximum_iterations,
          parallel_iterations=parallel_iterations, 
          swap_memory=swap_memory,
          scope=fw_scope
        )

    # Backward direction
    if not output_time_major:
      time_dim = 1
      batch_dim = 0
    else:
      time_dim = 0
      batch_dim = 1

    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return tf.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
      else:
        return tf.reverse(input_, axis=[seq_dim])

    with tf.variable_scope("bw") as bw_scope:
      bw_final_outputs, bw_final_state, bw_final_sequence_lengths= tf.contrib.seq2seq.dynamic_decode(
          bw_decoder, output_time_major=output_time_major, 
          impute_finished=impute_finished, 
          maximum_iterations=maximum_iterations,
          parallel_iterations=parallel_iterations, 
          swap_memory=swap_memory,
          scope=bw_scope
        )
  
  if not isinstance(fw_decoder, tf.contrib.seq2seq.BeamSearchDecoder):
    # no beam search
    fw_rnn_output = fw_final_outputs.rnn_output
    bw_rnn_output = bw_final_outputs.rnn_output
    fw_before_output_layer = fw_final_outputs.before_output_layer
    bw_before_output_layer = bw_final_outputs.before_output_layer
  else:
    fw_rnn_output = tf.no_op()
    bw_rnn_output = tf.no_op()
    fw_before_output_layer = tf.no_op()
    bw_before_output_layer = tf.no_op()

  rnn_outputs = (fw_rnn_output, bw_rnn_output)
  output_states = (fw_final_state, bw_final_state)
  decoder_outputs = (fw_final_outputs, bw_final_outputs)
  final_seq_lengths = (fw_final_sequence_lengths, bw_final_sequence_lengths)
  origin_rnn_outputs = (fw_before_output_layer, bw_before_output_layer)

  if return_seq == False:
    return (rnn_outputs, output_states, decoder_outputs)
  else:
    return (rnn_outputs, output_states, decoder_outputs, final_seq_lengths, origin_rnn_outputs)



def dynamic_bidecode_att(fw_decoder, bw_decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
  # first, run fw and bw decode with default attention
  (rnn_outputs, _, _, final_seq_lengths, origin_outputs) = dynamic_bidecode(fw_decoder, bw_decoder,
                                    output_time_major,
                                    impute_finished,
                                    maximum_iterations,
                                    parallel_iterations,
                                    swap_memory,
                                    scope, True)
  fw_rnn_output, bw_rnn_output = rnn_outputs
  fw_rnn_lengths, bw_rnn_lengths = final_seq_lengths
  fw_origin_output, bw_origin_output = origin_outputs

  with tf.control_dependencies([fw_rnn_output]):
    #second, assign rnn_output to new attention.values
    if output_time_major:
      #[T, B, N] => [B, T, N]
      fw_origin_output = tf.transpose(fw_origin_output, [1, 0, 2])
      bw_origin_output = tf.transpose(bw_origin_output, [1, 0, 2])

    fw_assgin = fw_decoder.set_attention_values(fw_origin_output, fw_rnn_lengths)
    bw_assgin = bw_decoder.set_attention_values(bw_origin_output, bw_rnn_lengths)
    fw_decoder._helper = fw_decoder._helper_bak
    bw_decoder._helper = bw_decoder._helper_bak

    with tf.control_dependencies(fw_assgin + bw_assgin):
      # third, re-run dynamic_bidecode and return
      return dynamic_bidecode(fw_decoder, bw_decoder,
                                    output_time_major,
                                    impute_finished,
                                    maximum_iterations,
                                    parallel_iterations,
                                    swap_memory,
                                    scope)