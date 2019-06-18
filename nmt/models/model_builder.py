import collections
import tensorflow as tf
import tensorflow.contrib as contrib

from FLAGS import PARAM
# from models import gnmt_model
# from models import attention_model
from models import vanilla_model
from utils import dataset_utils
from utils import vocab_utils


def _get_model_creator():
  model_creator = vanilla_model.BaseModel
  # if PARAM.model_type in ['gnmt','gnmt_current']:
  #   model_creator = gnmt_model.GNMTModel
  # elif PARAM.model_type == 'standard_attention':
  #   model_creator = attention_model.AttentionModel
  # elif PARAM.model_type == 'vanilla':
  #   model_creator = vanilla_model.Model
  # else:
  #   raise ValueError('Unknown model type %s.' %
  # #                    PARAM.model_type)
  return model_creator


class GraphModelDataset(
    collections.namedtuple("GraphModelDataset", ("graph", "model", "dataset"))):
  pass

def build_train_model(log_file, scope='train'):
  """Build train graph, model, and iterator."""

  model_creator = _get_model_creator()
  graph = tf.Graph()
  with graph.as_default(), tf.container(scope):
    src_file = "%s.%s" % (PARAM.train_prefix, PARAM.src)
    tgt_file = "%s.%s" % (PARAM.train_prefix, PARAM.tgt)
    vocab_tables = vocab_utils.create_vocab_tables(log_file) # word->id
    src_vocab_table, tgt_vocab_table, src_table_size, tgt_table_size = vocab_tables

    train_set = dataset_utils.get_batch_inputs_form_dataset(log_file,
                                                            src_file,
                                                            tgt_file,
                                                            src_vocab_table,
                                                            tgt_vocab_table)

    train_model = model_creator(log_file=log_file,
                                mode=vanilla_model.PARAM.MODEL_TRAIN_KEY,
                                source_id_seq=train_set.source_id_seq,
                                target_in_id_seq=train_set.target_in_id_seq,
                                target_out_id_seq=train_set.target_out_id_seq,
                                source_seq_lenghts=train_set.source_seq_lengths,
                                target_seq_lengths=train_set.target_seq_lengths,
                                source_vocab_table=src_vocab_table,
                                target_vocab_table=tgt_vocab_table,
                                src_vocab_size=src_table_size,
                                tgt_vocab_size=src_table_size,
                                scope=scope)
  return GraphModelDataset(graph=graph, model=train_model, dataset=train_set)
