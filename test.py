import tensorflow as tf
from tensorflow.python.ops import lookup_ops


tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
src_dataset = tf.data.Dataset.from_tensor_slices(
    tf.constant(["f e a g", "c c a", "d", "c a"]))
src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
src_dataset = src_dataset.map(lambda src: tf.reverse(src, [-1]))
# src_dataset = src_dataset.map(
#       lambda src: (tf.cast(src_vocab_table.lookup(src), tf.int32)))
# def batching_func(x):
#     return x.padded_batch(
#     2,
#     # The first three entries are the source and target line rows;
#     # these have unknown-length vectors.  The last two entries are
#     # the source and target row sizes; these are scalars.
#     padded_shapes=(
#         tf.TensorShape([None])),  # src,
#     # Pad the source and target sequences with eos tokens.
#     # (Though notice we don't generally need to do this since
#     # later on we will be masking out calculations past the true sequence.
#     padding_values=(
#         0))  # src
# batched_dataset = batching_func(src_dataset)
# batch = batched_dataset.make_initializable_iterator()
batch = src_dataset.make_one_shot_iterator()
tensor = batch.get_next()


with tf.Session() as sess:
    print(sess.run(tensor))