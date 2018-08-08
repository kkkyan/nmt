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
      if not isinstance(fw_decoder, tf.contrib.seq2seq.BeamSearchDecoder):
        fw_final_outputs, fw_final_state, fw_final_sequence_lengths,fw_origin_outputs = dynamic_decode(
          fw_decoder, output_time_major=output_time_major, 
          impute_finished=impute_finished, 
          maximum_iterations=maximum_iterations,
          parallel_iterations=parallel_iterations, 
          swap_memory=swap_memory,
          scope=fw_scope
        )
      else:
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
      if not isinstance(bw_decoder, tf.contrib.seq2seq.BeamSearchDecoder):
        bw_final_outputs, bw_final_state, bw_final_sequence_lengths, bw_origin_outputs= dynamic_decode(
          bw_decoder, output_time_major=output_time_major, 
          impute_finished=impute_finished, 
          maximum_iterations=maximum_iterations,
          parallel_iterations=parallel_iterations, 
          swap_memory=swap_memory,
          scope=bw_scope
        )
      else:
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
  else:
    fw_rnn_output = tf.no_op()
    bw_rnn_output = tf.no_op()
    fw_origin_outputs = tf.no_op()
    bw_origin_outputs = tf.no_op()

  rnn_outputs = (fw_rnn_output, bw_rnn_output)
  output_states = (fw_final_state, bw_final_state)
  decoder_outputs = (fw_final_outputs, bw_final_outputs)
  final_seq_lengths = (fw_final_sequence_lengths, bw_final_sequence_lengths)
  origin_rnn_outputs = (fw_origin_outputs, bw_origin_outputs)

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
  # first, run fw and bw decode with default attention
  with tf.control_dependencies([fw_rnn_output]):
    #second, assign rnn_output to new attention.values

    fw_assgin = fw_decoder.set_attention_values(fw_origin_output, fw_rnn_lengths)
    bw_assgin = bw_decoder.set_attention_values(bw_origin_output, bw_rnn_lengths)

    with tf.control_dependencies(fw_assgin + bw_assgin):
      # third, re-run dynamic_bidecode and return
      return dynamic_bidecode(fw_decoder, bw_decoder,
                                    output_time_major,
                                    impute_finished,
                                    maximum_iterations,
                                    parallel_iterations,
                                    swap_memory,
                                    scope)

def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
  """Perform dynamic decoding with `decoder`.

  Calls initialize() once and step() repeatedly on the Decoder object.

  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.

  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.

  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  """
  if not isinstance(decoder, Decoder):
    raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                    type(decoder))

  def _is_xla_tensor(tensor):
    try:
      op = tensor.op
    except AttributeError:
      return False
    if control_flow_util.IsInXLAContext(op):
      return True
    return False

  with variable_scope.variable_scope(scope, "decoder") as varscope:
    # Properly cache variable values inside the while_loop
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
      if maximum_iterations.get_shape().ndims != 0:
        raise ValueError("maximum_iterations must be a scalar")

    initial_finished, initial_inputs, initial_state = decoder.initialize()

    zero_outputs = _create_zero_outputs(decoder.output_size,
                                        decoder.output_dtype,
                                        decoder.batch_size)
    origin_zero_outputs = _create_zero_outputs(decoder.decoder_output_size,
                                        dtypes.float32,
                                        decoder.batch_size)

    is_xla = False
    if any([_is_xla_tensor(i) for i in nest.flatten(initial_inputs)]):
      is_xla = True
    if is_xla and maximum_iterations is None:
      raise ValueError("maximum_iterations is required for XLA compilation.")
    if maximum_iterations is not None:
      initial_finished = math_ops.logical_or(
          initial_finished, 0 >= maximum_iterations)
    initial_sequence_lengths = array_ops.zeros_like(
        initial_finished, dtype=dtypes.int32)
    initial_time = constant_op.constant(0, dtype=dtypes.int32)

    def _shape(batch_size, from_shape):
      if (not isinstance(from_shape, tensor_shape.TensorShape) or
          from_shape.ndims == 0):
        return tensor_shape.TensorShape(None)
      else:
        batch_size = tensor_util.constant_value(
            ops.convert_to_tensor(
                batch_size, name="batch_size"))
        return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

    dynamic_size = maximum_iterations is None or not is_xla

    def _create_ta(s, d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=_shape(decoder.batch_size, s))

    initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                            decoder.output_dtype)
    initial_origin_outputs_ta = nest.map_structure(_create_ta, decoder.decoder_output_size,
                                            tf.float32)

    def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                  finished, unused_sequence_lengths, unused_origin_outputs_ta):
      return math_ops.logical_not(math_ops.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished, sequence_lengths, origin_outputs_ta):
      """Internal while_loop body.

      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: bool tensor (keeping track of what's finished).
        sequence_lengths: int32 tensor (keeping track of time of finish).

      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
          next_sequence_lengths)`.
        ```
      """
      (next_outputs, decoder_state, next_inputs,
       decoder_finished, next_origin_outputs) = decoder.step(time, inputs, state)
      if decoder.tracks_own_finished:
        next_finished = decoder_finished
      else:
        next_finished = math_ops.logical_or(decoder_finished, finished)
      next_sequence_lengths = array_ops.where(
          math_ops.logical_not(finished),
          array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
          sequence_lengths)

      nest.assert_same_structure(state, decoder_state)
      nest.assert_same_structure(outputs_ta, next_outputs)
      nest.assert_same_structure(inputs, next_inputs)

      # Zero out output values past finish
      if impute_finished:
        emit = nest.map_structure(
            lambda out, zero: array_ops.where(finished, zero, out),
            next_outputs,
            zero_outputs)
        origin_emit = nest.map_structure(
            lambda out, zero: array_ops.where(finished, zero, out),
            next_origin_outputs,
            origin_zero_outputs)
      else:
        emit = next_outputs
        origin_emit = next_origin_outputs

      # Copy through states past finish
      def _maybe_copy_state(new, cur):
        # TensorArrays and scalar states get passed through.
        if isinstance(cur, tensor_array_ops.TensorArray):
          pass_through = True
        else:
          new.set_shape(cur.shape)
          pass_through = (new.shape.ndims == 0)
        return new if pass_through else array_ops.where(finished, cur, new)

      if impute_finished:
        next_state = nest.map_structure(
            _maybe_copy_state, decoder_state, state)
      else:
        next_state = decoder_state

      outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs_ta, emit)
      origin_outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      origin_outputs_ta, origin_emit)
      return (time + 1, outputs_ta, next_state, next_inputs, next_finished,
              next_sequence_lengths, origin_outputs_ta)

    res = control_flow_ops.while_loop(
        condition,
        body,
        loop_vars=(
            initial_time,
            initial_outputs_ta,
            initial_state,
            initial_inputs,
            initial_finished,
            initial_sequence_lengths,
            initial_origin_outputs_ta
        ),
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
        swap_memory=swap_memory)

    final_outputs_ta = res[1]
    final_state = res[2]
    final_sequence_lengths = res[5]
    final_origin_outputs_ta = res[6]

    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)
    final_origin_outputs = nest.map_structure(lambda ta: ta.stack(), final_origin_outputs_ta)

    try:
      final_outputs, final_state = decoder.finalize(
          final_outputs, final_state, final_sequence_lengths)
    except NotImplementedError:
      pass

    if not output_time_major:
      final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)
      final_origin_outputs = nest.map_structure(_transpose_batch_time, final_origin_outputs)

  return final_outputs, final_state, final_sequence_lengths, final_origin_outputs