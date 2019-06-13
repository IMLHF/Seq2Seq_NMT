import collections
import tensorflow as tf
import os
from utils import vocab_utils
import codecs
from FLAGS import PARAM

__all__ = ['load_data', 'get_batch_inputs_form_dataset']


class BatchInputs(
    collections.namedtuple("BatchInputs",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass


def get_batch_inputs_form_dataset(log_file,
                                  source_textline_file,
                                  target_textline_file,
                                  source_vocab_table,
                                  target_vocab_table):
  src_dataset = tf.data.TextLineDataset(source_textline_file)
  tgt_dataset = tf.data.TextLineDataset(target_textline_file)

  if not PARAM.output_buffer_size:
    output_buffer_size = PARAM.batch_size*100

  if PARAM.use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(source_vocab_table.lookup(tf.constant(PARAM.eos)), tf.int32)

def load_data(inference_input_file, hparams=None):
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  if hparams and hparams.inference_indices:
    inference_data = [inference_data[i] for i in hparams.inference_indices]

  return inference_data

