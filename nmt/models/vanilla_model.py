import abc
import collections
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from FLAGS import PARAM
from utils import loss_utils
from utils import misc_utils
from utils import vocab_utils

misc_utils.check_tensorflow_version()

__all__ = [
    # 'TrainOutputs',
    'BaseModel']

# class TrainOutputs(collections.namedtuple(
#     "TrainOutputs", ("train_summary", "train_loss", "predict_count",
#                      "global_step", "word_count", "batch_size", "grad_norm",
#                      "learning_rate"))):
#   """To allow for flexibily in returing different outputs."""
#   pass

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

    if PARAM.use_char_encode:
      assert (not self.time_major), ("Can't use time major for char-level inputs.")

    self.dtype=tf.float32
    self.num_sampled_softmax = PARAM.num_sampled_softmax
    self.single_cell_fn = None
    self.num_units = PARAM.num_units
    self.num_encoder_layers = PARAM.num_encoder_layers or PARAM.num_layers
    self.num_decoder_layers = PARAM.num_decoder_layers or PARAM.num_layers
    assert self.num_encoder_layers and self.num_decoder_layers, 'layers num error'

    self.batch_size = tf.size(self.batch_inputs.source_sequence_lengths)
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
    with tf.variable_scope(scope or "build_network"):
      with tf.variable_scope("decoder/output_projection"):
        # Projection
        self.output_layer = tf.layers.Dense(
            self.tgt_vocab_size, use_bias=False, name="output_projection")

    with tf.variable_scope('dynamic_seq2seq',dtype=self.dtype):
      # encoder
      if PARAM.language_model:
        misc_utils.printinfo("language modeling: no encoder", log_file)
        self.encoder_outputs = None
        self.encoder_final_state = None
      else:
        self.encoder_outputs, self.encoder_final_state = self._build_encoder() #TODO build_encoder

      # decoder
      self.logits, self.decoder_cell_outputs, self.sample_id, self.final_context_state = (
        self._build_decoder(self.encoder_outputs,self.encoder_final_state)) #TODO build_decoder

      # infer end
      if self.mode == BaseModel.infer_mode:
        self.reverse_target_vocab_table = lookup_ops.index_to_string_table_from_file(
            self.tgt_vocab_file, default_value=vocab_utils.UNK)
        self.sample_words = self.reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))
        return

      # loss
      crossent = self._softmax_cross_entropy_loss(self.logits,self.decoder_cell_outputs,self.batch_inputs.target_output_id_seq)
      self.loss = loss_utils.cross_entropy_loss(self.logits,
                                                crossent,
                                                self.decoder_cell_outputs,
                                                self.batch_inputs.target_output_id_seq,
                                                self.batch_inputs.target_sequence_lengths,
                                                self.time_major,
                                                self.batch_size)

      # eval end
      if self.mode == BaseModel.val_mode:
        return

      # Count the number of predicted words for compute ppl.
      self.word_count = tf.reduce_sum(
        self.batch_inputs.source_sequence_lengths+self.batch_inputs.target_sequence_lengths)
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_length)

      # region apply gradient
      # Gradients and SGD update operation for training the model.
      # Arrange for the embedding vars to appear at the beginning.
      self.learning_rate = tf.constant(PARAM.learning_rate)
      #TODO set learning_rate -> variable
      # warm-up # not use
      # self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay # not use
      # self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      if PARAM.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif PARAM.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise ValueError("Unknown optimizer type %s" % PARAM.optimizer)
      # Gradients
      params = tf.trainable_variables()
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=PARAM.colocate_gradients_with_ops)
      clipped_grads, grad_norm_summary, grad_norm = misc_utils.gradient_clip(
          gradients, max_gradient_norm=PARAM.max_gradient_norm)
      self.grad_norm_summary = grad_norm_summary
      self.grad_norm = grad_norm

      self.update = opt.apply_gradients(zip(clipped_grads, params))
      # endregion

      # Summary
      self.train_summary = tf.summary.merge(
          [tf.summary.scalar("lr", self.learning_rate),
           tf.summary.scalar("train_loss", self.train_loss)] +
          self.grad_norm_summary)
      # self.saver = tf.train.Saver(tf.global_variables(),
      self.saver = tf.train.Saver(tf.trainable_variables(),
                                  max_to_keep=PARAM.num_keep_ckpts)


  def _softmax_cross_entropy_loss(
    self, logits, decoder_cell_outputs, labels):
    """Compute softmax loss or sampled softmax loss."""
    if self.num_sampled_softmax > 0:

      is_sequence = (decoder_cell_outputs.shape.ndims == 3)

      if is_sequence:
        labels = tf.reshape(labels, [-1, 1])
        inputs = tf.reshape(decoder_cell_outputs, [-1, self.num_units])

      crossent = tf.nn.sampled_softmax_loss(
          weights=tf.transpose(self.output_layer.kernel),
          biases=self.output_layer.bias or tf.zeros([self.tgt_vocab_size]),
          labels=labels,
          inputs=inputs,
          num_sampled=self.num_sampled_softmax,
          num_classes=self.tgt_vocab_size,
          partition_strategy="div")

      if is_sequence:
        if self.time_major:
          crossent = tf.reshape(crossent, [-1, self.batch_size])
        else:
          crossent = tf.reshape(crossent, [self.batch_size, -1])

    else:
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
    return crossent

  @abc.abstractmethod
  def _build_encoder(self):
    '''
    Returns:
      a tuple: (encoder_outputs, encoder_final_state)
    '''
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "_build_encoder not implement,code: dkfkasd0395jfa0295jf")

  def _build_decoder(self, encoder_outputs, encoder_final_state):
    '''
    Args:
      a tuple: (encoder_outputs, encoder_final_state)
    Returns:
      A tuple: (final_logits, decoder_final_state)
        logits dim: [time, batch_size, vocab_size] when time_major=True
    '''
    tgt_sos_id = tf.cast(
      self.tgt_vocab_table.lookup(tf.constant(PARAM.sos)), tf.int32)
    tgt_eos_di = tf.cast(
      self.tgt_vocab_table.lookup(tf.constant(PARAM.eos)), tf.int32)


