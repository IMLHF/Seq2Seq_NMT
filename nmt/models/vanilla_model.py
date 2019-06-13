from utils import misc_utils
import collections
from utils import vocab_utils
from FLAGS import PARAM
import tensorflow as tf

misc_utils.check_tensorflow_version()

__all__ = []

class TrainOutputs(collections.namedtuple(
    "TrainOutputs", ("train_summary", "train_loss", "predict_count",
                     "global_step", "word_count", "batch_size", "grad_norm",
                     "learning_rate"))):
  """To allow for flexibily in returing different outputs."""
  pass

class BaseModel(object):
  '''
  sequence-to-sequence base class
  '''
  train_mode = 'train' # key
  val_mode = 'val'
  infer_mode = 'infer'

  def __init__(self,
               log_file,
               mode,
               batch_inputs,
               source_vocab_table,
               target_vocab_table,
               src_vocab_size,
               tgt_vocab_size,
               scope):
    '''
    Args:
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      scope: scope of the model.
    '''
    self.log_file = log_file
    self.batch_inputs = batch_inputs
    self.mode = mode
    self.src_vocab_file = "%s.%s" % (PARAM.vocab_prefix, PARAM.src)
    self.tgt_vocab_file = "%s.%s" % (PARAM.vocab_prefix, PARAM.tgt)
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.num_gpus = PARAM.num_gpus
    self.time_major = PARAM.time_major

    if PARAM.use_char_encoder:
      assert (not self.time_major), ("Can't use time major for char-level inputs.")

    self.dtype=tf.float32
    self.num_sampled_softmax = PARAM.num_sampled_softmax
    self.single_cell_fn = None
    self.num_units = PARAM.num_units
    self.num_encoder_layers = PARAM.num_encoder_layers or PARAM.num_layers
    self.num_decoder_layers = PARAM.num_decoder_layers or PARAM.num_layers
    assert self.num_encoder_layers and self.num_decoder_layers, 'layers num error'

    self.batch_size = tf.size(self.batch_inputs.source_sequence_length)
    initializer = misc_utils.get_initializer(
        init_op=PARAM.init_op, init_weight=PARAM.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # init word embedding
    self.encoder_emb_lookup_fn = tf.nn.embedding_lookup
    self.embedding_encoder = None
    self.embedding_decoder = None
    with tf.variable_scope('embeddings',dtype=self.dtype):
      src_embed_size = self.num_units
      tgt_embed_size = self.num_units
      src_embed_file = "%s.%s" % (PARAM.embed_prefix, PARAM.src) if not PARAM.embed_prefix else None
      tgt_embed_file = "%s.%s" % (PARAM.embed_prefix, PARAM.tgt) if not PARAM.embed_prefix else None
      if PARAM.share_vocab:
        assert self.src_vocab_size == self.tgt_vocab_size, "" + \
            "Share embedding but different src/tgt vocab sizes"
        assert src_embed_file == tgt_embed_file, 'Share embedding but different src/tgt embed_file'
        misc_utils.printinfo('# Use the same embedding for source and target.')
        vocab_file = self.src_vocab_file or self.tgt_vocab_file
        vocab_size = self.src_vocab_size or self.tgt_vocab_size
        embed_file = src_embed_file or tgt_embed_file
        embed_size = src_embed_size or tgt_embed_size

        self.embedding_encoder = vocab_utils.new_or_pretrain_embed(log_file,
                                                                   "embedding_share",
                                                                   vocab_file,
                                                                   embed_file,
                                                                   vocab_size,
                                                                   embed_size,
                                                                   self.dtype)
        self.embedding_decoder = self.embedding_encoder
      else:
        if not PARAM.use_char_encode:
          with tf.variable_scope("encoder"):
            self.embedding_encoder = vocab_utils.new_or_pretrain_embed(log_file,
                                                                       "embedding_encoder",
                                                                       self.src_vocab_file,
                                                                       src_embed_file,
                                                                       self.src_vocab_size,
                                                                       src_embed_size,
                                                                       self.dtype)
        # else:
        #   self.embedding_encoder = None

        with tf.variable_scope("decoder"):
          self.embedding_decoder = vocab_utils.new_or_pretrain_embed(log_file,
                                                                     "embedding_decoder",
                                                                     self.tgt_vocab_file,
                                                                     tgt_embed_file,
                                                                     self.tgt_vocab_size,
                                                                     tgt_embed_size,
                                                                     self.dtype)

    # train graph
    with tf.variable_scope('dynamic_seq2seq',dtype=self.dtype):
      # encoder
      if PARAM.language_model:
        misc_utils.printinfo("language modeling: no encoder", log_file)
        self.encoder_outputs = None
        self.encoder_final_state = None
      else:
        self.encoder_outputs, self.encoder_final_state = self._build_encoder()

      # decoder

      self.logits, self.decoder_cell_outputs, self.sample_id, self.final_context_state = (
        self._build_decoder(self.encoder_outputs,self.encoder_final_state))

