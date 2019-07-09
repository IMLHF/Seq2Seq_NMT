import codecs
import collections
import os
import tensorflow as tf

from ..FLAGS import PARAM
from ..utils import vocab_utils

__all__ = ['load_data', 'get_batch_inputs_form_dataset']

class DataSetsOutputs(
    collections.namedtuple("DataSetsOutputs",
                           ("initializer",
                            "source_id_seq", "target_in_id_seq", "target_out_id_seq",
                            "source_seq_lengths", "target_seq_lengths",
                            "src_textline_file_ph", "tgt_textline_file_ph"))):
  pass

def _batching_func_with_labels(dataset, batch_size, src_eos_id, tgt_eos_id):
  return dataset.padded_batch(
      batch_size,
      # The first three entries are the source and target line rows;
      # these have unknown-length vectors.  The last two entries are
      # the source and target row sizes; these are scalars.
      padded_shapes=(
          tf.TensorShape([None]),  # src
          tf.TensorShape([None]),  # tgt_input
          tf.TensorShape([None]),  # tgt_output
          tf.TensorShape([]),  # src_len
          tf.TensorShape([])),  # tgt_len
      # Pad the source and target sequences with eos tokens.
      # (Though notice we don't generally need to do this since
      # later on we will be masking out calculations past the true sequence.
      padding_values=(
          src_eos_id,  # src
          tgt_eos_id,  # tgt_input
          tgt_eos_id,  # tgt_output
          0,  # src_len -- unused
          0))  # tgt_len -- unused

def _bucket_dataset_by_length(dataset, batch_size, src_eos_id, tgt_eos_id):
  def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if PARAM.src_max_len:
        bucket_width = (PARAM.src_max_len + PARAM.num_buckets - 1) // PARAM.num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(PARAM.num_buckets, bucket_id))

  def reduce_func(unused_key, windowed_data):
    return _batching_func_with_labels(windowed_data, batch_size, src_eos_id, tgt_eos_id)

  batched_dataset = dataset.apply(
      tf.data.experimental.group_by_window(
          key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  return batched_dataset

def get_batch_inputs_form_dataset(log_file,
                                  source_textline_file,
                                  target_textline_file,
                                  source_vocab_table,
                                  target_vocab_table,
                                  shuffle=True):

  '''
  source_vocab_table: word->id
  '''
  src_dataset = tf.data.TextLineDataset(source_textline_file)
  tgt_dataset = tf.data.TextLineDataset(target_textline_file)

  output_buffer_size = PARAM.output_buffer_size
  if not PARAM.output_buffer_size:
    output_buffer_size = PARAM.batch_size * 1000

  if PARAM.use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(source_vocab_table.lookup(tf.constant(PARAM.eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  if shuffle:
    src_tgt_dataset = src_tgt_dataset.shuffle(
        buffer_size=output_buffer_size, reshuffle_each_iteration=PARAM.reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=PARAM.num_parallel_calls)
  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if PARAM.src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:PARAM.src_max_len], tgt),
        num_parallel_calls=PARAM.num_parallel_calls).prefetch(output_buffer_size)
  if PARAM.tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:PARAM.tgt_max_len]),
        num_parallel_calls=PARAM.num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  if PARAM.use_char_encode:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.reshape(vocab_utils.tokens_to_bytes(src), [-1]),
                          tf.cast(target_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=PARAM.num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(source_vocab_table.lookup(src), tf.int32),
                          tf.cast(target_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=PARAM.num_parallel_calls)
  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  tgt_sos_id = tf.cast(target_vocab_table.lookup(tf.constant(PARAM.sos)), tf.int32)
  tgt_eos_id = tf.cast(target_vocab_table.lookup(tf.constant(PARAM.eos)), tf.int32)
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=PARAM.num_parallel_calls).prefetch(output_buffer_size)

  # Add sequence lengths.
  if PARAM.use_char_encode:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out,
            tf.to_int32(tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN),
            tf.size(tgt_in)),
        num_parallel_calls=PARAM.num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=PARAM.num_parallel_calls)
  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  if PARAM.num_buckets > 1:
    # Bucket sentence pairs by the length of their source sentence and target sentence.
    src_tgt_dataset = _bucket_dataset_by_length(src_tgt_dataset, PARAM.batch_size,
                                                src_eos_id, tgt_eos_id)
  else:
    src_tgt_dataset = _batching_func_with_labels(src_tgt_dataset, PARAM.batch_size,
                                                 src_eos_id, tgt_eos_id)

  batched_iter = src_tgt_dataset.make_initializable_iterator()
  src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len = batched_iter.get_next()
  return DataSetsOutputs(
    initializer=batched_iter.initializer,
    source_id_seq=src_ids,
    target_in_id_seq=tgt_input_ids,
    target_out_id_seq=tgt_output_ids,
    source_seq_lengths=src_seq_len,
    target_seq_lengths=tgt_seq_len,
    src_textline_file_ph=source_textline_file,
    tgt_textline_file_ph=target_textline_file,
  )

def load_data(inference_input_file, hparams=None):
  """Load inference data."""
  with codecs.getreader("utf-8")(
          tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  if hparams and hparams.inference_indices:
    inference_data = [inference_data[i] for i in hparams.inference_indices]
  return inference_data
