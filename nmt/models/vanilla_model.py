import abc
import collections
import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.python.ops import lookup_ops

from FLAGS import PARAM
from utils import loss_utils
from utils import misc_utils
from utils import vocab_utils
from models import model_helper

misc_utils.check_tensorflow_version()

__all__ = [
    # 'TrainOutputs',
    'BaseModel',
    'Model']

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

  def __init__(self,
               log_file,
               mode,
               source_id_seq,
               target_in_id_seq,
               target_out_id_seq,
               source_seq_lengths,
               target_seq_lengths,
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
    self.source_id_seq = source_id_seq # [batch, time(ids_num)]
    self.target_in_id_seq = target_in_id_seq # [batch, time]
    self.target_out_id_seq = target_out_id_seq # [baatch, time]
    self.source_seq_lengths = source_seq_lengths # [batch]
    self.target_seq_lengths = target_seq_lengths # [batch]
    self.mode = mode
    self.src_vocab_file = "%s.%s" % (PARAM.vocab_prefix, PARAM.src)
    self.tgt_vocab_file = "%s.%s" % (PARAM.vocab_prefix, PARAM.tgt)
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.batch_size = tf.size(self.source_seq_lengths)
    self.dtype=tf.float32
    self.single_cell_fn = None
    self.num_encoder_layers = PARAM.num_encoder_layers or PARAM.num_layers
    self.num_decoder_layers = PARAM.num_decoder_layers or PARAM.num_layers
    assert self.num_encoder_layers and self.num_decoder_layers, 'layers num error'

    if PARAM.use_char_encode:
      assert (not PARAM.time_major), ("Can't use time major for char-level inputs.")

    initializer = misc_utils.get_initializer(
        init_op=PARAM.init_op, init_weight=PARAM.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # init word embedding
    self.embedding_encoder = None
    self.embedding_decoder = None
    with tf.variable_scope('embeddings',dtype=self.dtype):
      src_embed_size = PARAM.encoder_num_units
      tgt_embed_size = PARAM.decoder_num_units
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
        encoder_outputs = None
        encoder_final_state = None
      else:
        encoder_outputs, encoder_final_state = self._build_encoder(
            self.source_id_seq,
            self.source_seq_lengths
        )  # build_encoder

      # decoder
      self.logits, self.decoder_cell_outputs, self.sample_id, self.final_context_state = (
        self._build_decoder(encoder_outputs, encoder_final_state)
      )  # build_decoder

      # infer end
      if self.mode == PARAM.MODEL_INFER_KEY:
        self.reverse_target_vocab_table = lookup_ops.index_to_string_table_from_file(
            self.tgt_vocab_file, default_value=vocab_utils.UNK)
        self.sample_words = self.reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))
        return

      # loss
      crossent = self._softmax_cross_entropy_loss(self.logits,self.decoder_cell_outputs,self.target_out_id_seq)
      self.loss = loss_utils.cross_entropy_loss(self.logits,
                                                crossent,
                                                self.decoder_cell_outputs,
                                                self.target_out_id_seq,
                                                self.target_seq_lengths,
                                                self.batch_size)

      # eval end
      if self.mode == PARAM.MODEL_VALIDATE_KEY:
        return

      # Count the number of predicted words for compute ppl.
      self.word_count = tf.reduce_sum(
        self.source_seq_lengths+self.target_seq_lengths)
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_lengths)

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
    if PARAM.num_sampled_softmax > 0:

      is_sequence = (decoder_cell_outputs.shape.ndims == 3)

      if is_sequence:
        labels = tf.reshape(labels, [-1, 1])
        inputs = tf.reshape(decoder_cell_outputs, [-1, PARAM.decoder_num_units])

      crossent = tf.nn.sampled_softmax_loss(
          weights=tf.transpose(self.output_layer.kernel),
          biases=self.output_layer.bias or tf.zeros([self.tgt_vocab_size]),
          labels=labels,
          inputs=inputs,
          num_sampled=PARAM.num_sampled_softmax,
          num_classes=self.tgt_vocab_size,
          partition_strategy="div")

      if is_sequence:
        if PARAM.time_major:
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
        "_build_encoder not implement, code: dkfkasd0395jfa0295jf")

  @abc.abstractmethod
  def _build_decode(self,
                    encoder_outputs,
                    encoder_final_state):
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "_build_decoder not implement, code: jqwirjjg992jgaldjf-0")


class Model(BaseModel):
  def _build_encoder(self,seq,seq_lengths):
    '''
    Returns:
      encoder_outputs, encoder_final_state
    '''
    if PARAM.time_major:
      seq = tf.transpose(seq)

    with tf.variable_scope("encoder"):
      encoder_inputs = tf.nn.embedding_lookup(self.embedding_encoder,
                                              seq)
      if PARAM.encoder_type == "uni":
        misc_utils.printinfo("  num_layers = %d, num_residual_layers=%d" %
                             (PARAM.num_layers, PARAM.num_residual_layers), self.log_file)
        multi_cell = model_helper.multiRNNCell(
            unit_type=PARAM.encoder_unit_type,
            num_units=PARAM.encoder_num_units,
            num_layers=PARAM.encoder_num_layers,
            layer_start_residual=PARAM.encoder_layer_start_residual,
            forget_bias=PARAM.encoder_forget_bias,
            droprate=PARAM.encoder_drop_rate,
            mode=self.mode,
            num_gpus=PARAM.num_gpus
        )
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            multi_cell,
            encoder_inputs,
            dtype=self.dtype,
            sequence_length=seq_lengths,
            time_major=PARAM.time_major,
            swap_memory=True
        )
      elif PARAM.encoder_type == "bi":
        fw_multi_cell = model_helper.multiRNNCell(
            unit_type=PARAM.encoder_unit_type,
            num_units=PARAM.encoder_num_units,
            num_layers=PARAM.encoder_num_layers,
            layer_start_residual=PARAM.encoder_layer_start_residual,
            forget_bias=PARAM.encoder_forget_bias,
            droprate=PARAM.encoder_drop_rate,
            mode=self.mode,
            num_gpus=PARAM.num_gpus
        )
        bw_multi_cell = model_helper.multiRNNCell(
            unit_type=PARAM.encoder_unit_type,
            num_units=PARAM.encoder_num_units,
            num_layers=PARAM.encoder_num_layers,
            layer_start_residual=PARAM.encoder_layer_start_residual,
            forget_bias=PARAM.encoder_forget_bias,
            droprate=PARAM.encoder_drop_rate,
            mode=self.mode,
            num_gpus=PARAM.num_gpus
        )
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
          fw_multi_cell,
          bw_multi_cell,
          encoder_inputs,
          dtype=self.dtype,
          sequence_length=seq_lengths,
          time_major=PARAM.time_major,
          swap_memory=True
        )
        encoder_outputs, bi_encoder_state = tf.concat(bi_outputs,-1), bi_state
        if PARAM.encoder_num_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(PARAM.encoder_num_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % PARAM.encoder_type)

    self.encoder_state_list = [encoder_outputs]
    return encoder_outputs, encoder_state

  def _build_encoder_cell(
          self, encoder_outputs, encoder_final_state, source_seq_length):
    multi_cell = model_helper.multiRNNCell(
        unit_type=PARAM.decoder_unit_type,
        num_units=PARAM.decoder_num_units,
        num_layers=PARAM.decoder_num_layers,
        layer_start_residual=PARAM.decoder_layer_start_residual,
        forget_bias=PARAM.decoder_forget_bias,
        droprate=PARAM.decoder_drop_rate,
        mode=self.mode,
        num_gpus=PARAM.num_gpus
    )
    if PARAM.language_model:
      encoder_final_state = multi_cell.zero_state(self.batch_size, self.dtype)

    if self.mode == PARAM.MODEL_INFER_KEY and PARAM.infer_mode == "beam_search":
      decoder_init_state = contrib.seq2seq.tile_batch(
        encoder_final_state,multiplier=PARAM.beam_width
      )
    else:
      decoder_init_state = encoder_final_state
    return multi_cell, decoder_init_state


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
    tgt_eos_id = tf.cast(
      self.tgt_vocab_table.lookup(tf.constant(PARAM.eos)), tf.int32)
    start_tokens = tf.fill([self.batch_size], tgt_sos_id)
    end_token = tgt_eos_id

    max_rnn_iterations = None
    if PARAM.tgt_max_len_infer:
      max_rnn_iterations = PARAM.tgt_max_len_infer
      misc_utils.printinfo("  decoding max_rnn_iterations %d" % max_rnn_iterations)
    else:
      max_src_len = tf.reduce_max(self.source_seq_lengths)
      max_rnn_iterations = tf.to_int32(tf.round(
        tf.to_float(max_src_len) * PARAM.tgt_max_len_infer_factor))

    with tf.variable_scope("decoder") as decoder_scope:
      # region decoder
      cell, decoder_init_state = self._build_decoder_cell(encoder_outputs,  #
                                                          encoder_final_state,
                                                          self.source_seq_lengths)
      ## inference
      if self.mode == PARAM.MODEL_INFER_KEY:
        decoder = None
        if PARAM.infer_mode == "beam_search":
          beam_width = PARAM.beam_width
          length_penalty_weight = PARAM.length_penalty_weight
          coverage_penalty_weight = PARAM.coverage_penalty_weight

          decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_init_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight,
              coverage_penalty_weight=coverage_penalty_weight)
        elif PARAM.infer_mode == "sample":
          # Helper
          sampling_temperature = PARAM.sampling_temperature
          assert sampling_temperature > 0.0, (
              "sampling_temperature must greater than 0.0 when using sample"
              " decoder.")
          helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token,
              softmax_temperature=sampling_temperature,
              seed=self.random_seed)
        elif PARAM.infer_mode == "greedy":
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)
        else:
          raise ValueError("Unknown infer_mode '%s'", PARAM.infer_mode)

        if PARAM.infer_mode != "beam_search":
          decoder = contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_init_state,
              output_layer=self.output_layer  # applied per timestep
          )

        # Dynamic decoding
        decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=max_rnn_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        decoder_cell_outputs = None
        decoder_final_state = final_context_state
        if PARAM.infer_mode == "beam_search":
          logits = tf.no_op()
          sample_id = decoder_outputs.predicted_ids
        else:
          logits = decoder_outputs.rnn_output
          sample_id = decoder.sample_id

        # return logits, decoder_cell_outputs, sample_id, decoder_final_state

      ## train or val
      else:
        decoder_inputs = tf.nn.embedding_lookup(self.embedding_decoder, # ids->embeding
                                                self.target_in_id_seq) # [batch, time, embedding_vec]
        if PARAM.time_major:
          decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2]) # [time, batch, embedding_vec]

        decoder_helper = contrib.seq2seq.TrainingHelper(decoder_inputs,
                                                        self.target_seq_lengths,
                                                        time_major=PARAM.time_major)

        basic_decoder = contrib.seq2seq.BasicDecoder(cell,decoder_helper,decoder_init_state)

        decoder_outputs, final_context_state, decoder_final_seq_lengths = contrib.seq2seq.dynamic_decode(
            basic_decoder,
            output_time_major=PARAM.time_major,
            swap_memory=True,
            scope=decoder_scope)

        logits = self.output_layer(decoder_outputs.rnn_output)
        sample_id = decoder_outputs.sample_id
        if PARAM.num_sampled_softmax > 0:
          logits = tf.no_op()
          decoder_cell_outputs = decoder_outputs.rnn_output

    return logits, decoder_cell_outputs, sample_id, decoder_final_state









