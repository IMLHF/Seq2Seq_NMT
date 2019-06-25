import tensorflow as tf
from ..FLAGS import PARAM


def cross_entropy_loss(logits, crossent, decoder_cell_outputs, target_output, target_sequence_length, batch_size):
  """Compute optimization loss."""
  target_output = target_output
  if PARAM.time_major:
    target_output = tf.transpose(target_output)
  time_axis = 0 if PARAM.time_major else 1
  max_time = target_output.shape[time_axis].value or tf.shape(target_output)[time_axis]

  # crossent = self._softmax_cross_entropy_loss(
  #     logits, decoder_cell_outputs, target_output)

  target_weights = tf.sequence_mask(
      target_sequence_length, max_time, dtype=tf.float32)

  # crossent [time, batch] if time_major else [batch, time]
  if PARAM.time_major:
    target_weights = tf.transpose(target_weights) # [time, batch]
    loss = tf.reduce_sum(crossent * target_weights, axis=0) # [batch]
  else:
    loss = tf.reduce_sum(crossent * target_weights, axis=-1) # [batch]

  seq_lengths = tf.cast(target_sequence_length,dtype=loss.dtype)
  loss = tf.reduce_mean(loss / seq_lengths)

  return loss
