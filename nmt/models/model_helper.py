import collections
import tensorflow as tf
from models import gnmt_model, attention_model, vanilla_model

from FLAGS import PARAM
from utils import dataset_utils
from utils import vocab_utils


def get_model_creator():
  model_creator = vanilla_model.BaseModel
  # if PARAM.model_type in ['gnmt','gnmt_current']:
  #   model_creator = gnmt_model.GNMTModel
  # elif PARAM.model_type == 'standard_attention':
  #   model_creator = attention_model.AttentionModel
  # elif PARAM.model_type == 'vanilla':
  #   model_creator = vanilla_model.Model
  # else:
  #   raise ValueError('Unknown model type %s.' %
  #                    PARAM.model_type)
  return model_creator


class GraphModelDataset(
    collections.namedtuple("GraphModelDataset", ("graph", "model", "dataset"))):
  pass

def build_train_model(log_file, model_creator, scope='train'):
  """Build train graph, model, and iterator."""

  graph = tf.Graph()
  with graph.as_default(), tf.container(scope):
    src_file = "%s.%s" % (PARAM.train_prefix, PARAM.src)
    tgt_file = "%s.%s" % (PARAM.train_prefix, PARAM.tgt)
    vocab_tables = vocab_utils.create_vocab_tables(log_file)
    src_vocab_table, tgt_vocab_table, src_table_size, tgt_table_size = vocab_tables

    train_set = dataset_utils.get_batch_inputs_form_dataset(log_file,
                                                            src_file,
                                                            tgt_file,
                                                            src_vocab_table,
                                                            tgt_vocab_table)

    train_model = model_creator(log_file=log_file,
                                mode=vanilla_model.BaseModel.train_mode,
                                batch_inputs=train_set,
                                source_vocab_table=src_vocab_table,
                                target_vocab_table=tgt_vocab_table,
                                src_vocab_size=src_table_size,
                                tgt_vocab_size=src_table_size,
                                scope=scope)
  return GraphModelDataset(graph=graph, model=train_model, dataset=train_set)

