from FLAGS import PARAM
from models import gnmt_model, attention_model, vanilla_model
from utils import vocab_utils
from utils import dataset_utils
import tensorflow as tf
import collections


def get_model_creator():
  if PARAM.model_type in ['gnmt','gnmt_current']:
    model_creator = gnmt_model.GNMTModel
  elif PARAM.model_type == 'standard_attention':
    model_creator = attention_model.AttentionModel
  elif PARAM.model_type == 'vanilla':
    model_creator = vanilla_model.Model
  else:
    raise ValueError('Unknown model type %s.' %
                     PARAM.model_type)
  return model_creator


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "train_set"))):
  pass

def create_train_model(log_file, model_creator, scope='train'):
  """Create train graph, model, and iterator."""

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
  return TrainModel(graph=graph, model=train_model, train_set=train_set)

