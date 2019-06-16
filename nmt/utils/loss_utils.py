import tensorflow as tf


def cross_entropy_loss(logits, crossent, decoder_cell_outputs, target_output, target_sequence_length, time_major, batch_size):
  """Compute optimization loss."""
  target_output = target_output
  if time_major:
    target_output = tf.transpose(target_output)
  time_axis = 0 if time_major else 1
  max_time = target_output.shape[time_axis].value or tf.shape(target_output)[time_axis]

  # crossent = self._softmax_cross_entropy_loss(
  #     logits, decoder_cell_outputs, target_output)

  target_weights = tf.sequence_mask(
      target_sequence_length, max_time, dtype=tf.float32)
  if time_major:
    target_weights = tf.transpose(target_weights)

  loss = tf.reduce_sum(
      crossent * target_weights) / tf.to_float(batch_size)
  return loss
