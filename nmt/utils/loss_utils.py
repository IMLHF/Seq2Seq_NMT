import tensorflow as tf
from ..FLAGS import PARAM


def masked_cross_entropy_loss(logits, crossent, decoder_cell_outputs, target_output, target_sequence_length, batch_size):
  """Compute optimization loss."""
  time_axis = 1
  max_time = target_output.shape[time_axis].value or tf.shape(target_output)[time_axis]

  # crossent = self._softmax_cross_entropy_loss(
  #     logits, decoder_cell_outputs, target_output)

  target_weights = tf.sequence_mask(
      target_sequence_length, max_time, dtype=tf.float32)

  # crossent: [batch, time]
  mat_loss = tf.multiply(crossent, target_weights) # [batch, time]
  loss = tf.reduce_sum(mat_loss, axis=-1) # [batch]
  seq_lengths = tf.cast(target_sequence_length,dtype=loss.dtype)
  # loss = tf.reduce_mean(loss / seq_lengths) # reduce_mean batch&time
  loss = tf.reduce_sum(loss / seq_lengths) # reduce_sum batch && reduce_mean time

  return mat_loss, loss
