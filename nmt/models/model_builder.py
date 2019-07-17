import collections
import os
import tensorflow as tf

from ..FLAGS import PARAM
from . import vanilla_model
from . import rnn_attention_model
from . import gnmt_model
from . import transformer_model
from ..utils import dataset_utils
from ..utils import misc_utils
from ..utils import vocab_utils

# ckpt_dir = os.path.join(PARAM.root_dir,'exp',PARAM.__class__.__name__,
#                         'ckpt')

def _get_model_creator():
  # model_creator = vanilla_model.RNNSeq2SeqModel
  if PARAM.model_type == 'vanilla':
    model_creator = vanilla_model.RNNSeq2SeqModel
  elif PARAM.model_type == 'standard_attention':
    model_creator = rnn_attention_model.RNNAttentionModel
  elif PARAM.model_type == 'gnmt':
    model_creator = gnmt_model.GNMTAttentionModel
  elif PARAM.model_type == 'transformer':
    model_creator = transformer_model.Transformer
  else:
    raise ValueError('Unknown model type %s.' %
                     PARAM.model_type)
  return model_creator


class BuildModelOutputs(
        collections.namedtuple("BuildModelOutputs", ("session", "graph", "model", "dataset"))):
  pass

def build_train_model(log_file, ckpt_dir, scope='train'):
  """Build train graph, model, and iterator."""

  model_creator = _get_model_creator()
  graph = tf.Graph()
  with graph.as_default(), tf.container(scope):
    # src_file = "%s.%s" % (PARAM.train_prefix, PARAM.src)
    # tgt_file = "%s.%s" % (PARAM.train_prefix, PARAM.tgt)
    # src_file = misc_utils.add_rootdir(src_file)
    # tgt_file = misc_utils.add_rootdir(tgt_file)
    src_file_ph = tf.placeholder(shape=(), dtype=tf.string)
    tgt_file_ph = tf.placeholder(shape=(), dtype=tf.string)
    vocab_tables = vocab_utils.create_vocab_word2id_tables(log_file) # word->id
    src_vocab_table, tgt_vocab_table, src_table_size, tgt_table_size = vocab_tables

    train_set = dataset_utils.get_batch_inputs_form_dataset(
        log_file,
        src_file_ph,
        tgt_file_ph,
        src_vocab_table,
        tgt_vocab_table,
    )

    train_model = model_creator(
        log_file=log_file,
        mode=PARAM.MODEL_TRAIN_KEY,
        source_id_seq=train_set.source_id_seq,
        source_seq_lengths=train_set.source_seq_lengths,
        tgt_vocab_table=tgt_vocab_table,
        src_vocab_size=src_table_size,
        tgt_vocab_size=tgt_table_size,
        target_in_id_seq=train_set.target_in_id_seq,
        target_out_id_seq=train_set.target_out_id_seq,
        target_seq_lengths=train_set.target_seq_lengths,
    )
    # return train_model, graph, train_set
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer(),
                    tf.tables_initializer())
    config_proto = misc_utils.get_session_config_proto()
    train_sess = tf.Session(config=config_proto, graph=graph)
    train_sess.run(init)
    ckpt_name = 'tmp'
    train_model.saver.save(train_sess, os.path.join(ckpt_dir, ckpt_name))
  return BuildModelOutputs(session=train_sess,
                           graph=graph,
                           model=train_model,
                           dataset=train_set)

def build_val_model(log_file, ckpt_dir, scope='validation'):
  model_creator = _get_model_creator()
  graph = tf.Graph()
  with graph.as_default(), tf.container(scope):
    # src_file = "%s.%s" % (PARAM.val_prefix, PARAM.src)
    # tgt_file = "%s.%s" % (PARAM.val_prefix, PARAM.tgt)
    # src_file = misc_utils.add_rootdir(src_file)
    # tgt_file = misc_utils.add_rootdir(tgt_file)
    src_file_ph = tf.placeholder(shape=(), dtype=tf.string)
    tgt_file_ph = tf.placeholder(shape=(), dtype=tf.string)
    vocab_tables = vocab_utils.create_vocab_word2id_tables(log_file) # word->id
    src_vocab_table, tgt_vocab_table, src_table_size, tgt_table_size = vocab_tables

    val_set = dataset_utils.get_batch_inputs_form_dataset(
        log_file,
        src_file_ph,
        tgt_file_ph,
        src_vocab_table,
        tgt_vocab_table,
        shuffle=False,
        bucket=False,
        filter_zero_seq=False,
    )

    val_model = model_creator(
        log_file=log_file,
        mode=PARAM.MODEL_VALIDATE_KEY,
        source_id_seq=val_set.source_id_seq,
        source_seq_lengths=val_set.source_seq_lengths,
        tgt_vocab_table=tgt_vocab_table,
        src_vocab_size=src_table_size,
        tgt_vocab_size=tgt_table_size,
        target_in_id_seq=val_set.target_in_id_seq,
        target_out_id_seq=val_set.target_out_id_seq,
        target_seq_lengths=val_set.target_seq_lengths,
    )
    # init = tf.group(tf.global_variables_initializer(),
    #                 tf.local_variables_initializer())
    config_proto = misc_utils.get_session_config_proto()
    val_sess = tf.Session(config=config_proto, graph=graph)
    # val_sess.run(init)
    val_sess.run(tf.tables_initializer())

    # restore model
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.set_verbosity(tf.logging.WARN)
      val_model.saver.restore(val_sess, ckpt.model_checkpoint_path)
      tf.logging.set_verbosity(tf.logging.INFO)
    else:
      msg = 'Checkpoint not found. code:fweikgn2394jasdjf2'
      tf.logging.fatal(msg)
      misc_utils.printinfo(msg,log_file,noPrt=True)

  return BuildModelOutputs(session=val_sess,
                           graph=graph,
                           model=val_model,
                           dataset=val_set)

def build_infer_model(log_file, ckpt_dir, scope='infer'):
  model_creator = _get_model_creator()
  graph = tf.Graph()
  with graph.as_default(), tf.container(scope):
    # src_file = "%s.%s" % (PARAM.val_prefix, PARAM.src)
    # tgt_file = "%s.%s" % (PARAM.val_prefix, PARAM.tgt)
    # src_file = misc_utils.add_rootdir(src_file)
    # tgt_file = misc_utils.add_rootdir(tgt_file)
    src_file_ph = tf.placeholder(shape=(), dtype=tf.string)
    tgt_file_ph = tf.placeholder(shape=(), dtype=tf.string)
    vocab_tables = vocab_utils.create_vocab_word2id_tables(log_file) # word->id
    src_vocab_table, tgt_vocab_table, src_table_size, tgt_table_size = vocab_tables

    infer_set = dataset_utils.get_batch_inputs_form_dataset(
        log_file,
        src_file_ph,
        tgt_file_ph,
        src_vocab_table,
        tgt_vocab_table,
        shuffle=False,
        bucket=False,
        filter_zero_seq=False,
    )

    infer_model = model_creator(
        log_file=log_file,
        mode=PARAM.MODEL_INFER_KEY,
        source_id_seq=infer_set.source_id_seq,
        source_seq_lengths=infer_set.source_seq_lengths,
        tgt_vocab_table=tgt_vocab_table,
        src_vocab_size=src_table_size,
        tgt_vocab_size=tgt_table_size,
    )
    # init = tf.group(tf.global_variables_initializer(),
    #                 tf.local_variables_initializer())
    config_proto = misc_utils.get_session_config_proto()
    infer_sess = tf.Session(config=config_proto, graph=graph)
    # infer_sess.run(init)
    infer_sess.run(tf.tables_initializer())

    # restore model
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.set_verbosity(tf.logging.WARN)
      infer_model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
      tf.logging.set_verbosity(tf.logging.INFO)
    else:
      msg = 'Checkpoint not found. code:fau598942trghhi78kj'
      tf.logging.fatal(msg)
      misc_utils.printinfo(msg,log_file,noPrt=True)
  return BuildModelOutputs(session=infer_sess,
                           graph=graph,
                           model=infer_model,
                           dataset=infer_set)
