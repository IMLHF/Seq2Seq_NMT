import tensorflow as tf
from ..FLAGS import PARAM


def masked_cross_entropy_loss(logits, target_output, target_sequence_length, batch_size):
  """Compute optimization loss."""
  time_axis = 1
  max_time = target_output.shape[time_axis].value or tf.shape(target_output)[time_axis]

  # print(target_output.get_shape().as_list(),logits.get_shape().as_list())
  # labels/target_output :[batch, time]
  # logits :[batch, time, vocab_num]
  crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_output, logits=logits)
  # print(crossent.get_shape().as_list())
  # crossent :[batch, time]

  target_weights = tf.sequence_mask(
      target_sequence_length, max_time, dtype=tf.float32)

  # crossent: [batch, time]
  mat_loss = tf.multiply(crossent, target_weights) # [batch, time]
  loss = tf.reduce_sum(mat_loss, axis=-1) # [batch]
  seq_lengths = tf.cast(target_sequence_length,dtype=loss.dtype)
  # loss = tf.reduce_mean(loss / seq_lengths) # reduce_mean batch&time
  loss = tf.reduce_sum(loss / seq_lengths) # reduce_sum batch && reduce_mean time

  return mat_loss, loss
