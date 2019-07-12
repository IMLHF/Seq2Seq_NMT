import abc
# import collections
import time
import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.python.ops import lookup_ops

from ..FLAGS import PARAM
from ..utils import loss_utils
from ..utils import misc_utils
from ..utils import vocab_utils
from . import model_helper

misc_utils.check_tensorflow_version()

__all__ = [
    # 'TrainOutputs',
    'BaseModel',
    'RNNSeq2SeqModel']

# class TrainOutputs(collections.namedtuple(
#     "TrainOutputs", ("train_summary", "train_loss", "predict_count",
#                      "global_step", "word_count", "batch_size", "grad_norm",
#                      "learning_rate"))):
#   """To allow for flexibily in returing different outputs."""
#   pass

class BaseModel(object):
  """
  sequence-to-sequence base class
  """

  def __init__(self,
               log_file,
               mode,
               source_id_seq,
               source_seq_lengths,
               tgt_vocab_table,
               src_vocab_size,
               tgt_vocab_size,
               target_in_id_seq=None,
               target_out_id_seq=None,
               target_seq_lengths=None):
    """
    Args:
      mode: PARAM.MODEL_TRAIN_KEY | PARAM.MODEL_VALIDATION_KEY | PARAM.MODEL_INFER_KEY
      tgt_vocab_table: Lookup table mapping target words to ids.
    """
    self.global_step = tf.get_variable('global_step',dtype=tf.int32,
                                       initializer=tf.constant(0),trainable=False)
    self.learning_rate = tf.get_variable('learning_rate', dtype=tf.float32, trainable=False,
                                         initializer=tf.constant(PARAM.learning_rate))
    self.new_lr = tf.placeholder(tf.float32,name="new_lr")
    self.assign_lr = tf.assign(self.learning_rate, self.new_lr)
    self.save_variables = [self.global_step, self.learning_rate]
    self.log_file = log_file
    self.source_id_seq = source_id_seq # [batch, time(ids_num)]
    self.target_in_id_seq = target_in_id_seq # [batch, time]
    self.target_out_id_seq = target_out_id_seq # [baatch, time]
    self.source_seq_lengths = source_seq_lengths # [batch]
    self.target_seq_lengths = target_seq_lengths # [batch]
    self.mode = mode # train: get loss. val: get loss, ppl. infer: get bleu etc.

    src_vocab_file = "%s.%s" % (PARAM.vocab_prefix, PARAM.src)
    tgt_vocab_file = "%s.%s" % (PARAM.vocab_prefix, PARAM.tgt)
    self.src_vocab_file = misc_utils.add_rootdir(src_vocab_file)
    self.tgt_vocab_file = misc_utils.add_rootdir(tgt_vocab_file)

    # self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = tgt_vocab_table
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.batch_size = tf.size(self.source_seq_lengths)
    self.dtype=PARAM.dtype
    self.single_cell_fn = None
    self.num_encoder_layers = PARAM.encoder_num_layers
    self.num_decoder_layers = PARAM.decoder_num_layers
    assert self.num_encoder_layers and self.num_decoder_layers, 'layers num error'

    if PARAM.use_char_encode:
      assert (not PARAM.time_major), "Can't use time major for char-level inputs."

    initializer = misc_utils.get_initializer(
        init_op=PARAM.init_op, init_weight=PARAM.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # init word embedding
    self.embedding_encoder = None
    self.embedding_decoder = None
    with tf.variable_scope('embeddings',dtype=self.dtype):
      src_embed_size = PARAM.src_embed_size
      tgt_embed_size = PARAM.tgt_embed_size
      src_embed_file = "%s.%s" % (PARAM.embed_prefix, PARAM.src) if PARAM.embed_prefix else None
      tgt_embed_file = "%s.%s" % (PARAM.embed_prefix, PARAM.tgt) if PARAM.embed_prefix else None
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
    self.output_layer = tf.layers.Dense( # output projection
        self.tgt_vocab_size, use_bias=False, name="output_projection")

    with tf.variable_scope('dynamic_seq2seq',dtype=self.dtype):
      # encoder
      if PARAM.language_model:
        misc_utils.printinfo("language modeling: no encoder", log_file)
        encoder_outputs = None
        encoder_final_state = None
      else:
        # encoder_outputs: [batch, time, 2*units] if bidirection else [[batch, time, units]]
        # encoder_final_state: [layers, 2(c,h), batch, units]
        encoder_outputs, encoder_final_state = self._build_encoder(
            self.source_id_seq,
            self.source_seq_lengths
        )  # build_encoder
        # print(encoder_outputs.get_shape().as_list(),"---------------------")

      # projection encoder_final_state
      if PARAM.projection_encoder_final_state:
        # Projection
        self.state_projection = tf.layers.Dense(
            PARAM.decoder_num_units, use_bias=False, name="encoder2decoder/state_projection")
        # encoder_final_state = self.state_projection(encoder_final_state)
        if PARAM.decoder_unit_type=='lstm' and PARAM.encoder_unit_type=='lstm':
          new_c_h_tupe_list = []
          for c_h_tuple in encoder_final_state:
            new_c_h_tupe_list.append(tf.nn.rnn_cell.LSTMStateTuple(c=self.state_projection(c_h_tuple[0]),
                                                                   h=self.state_projection(c_h_tuple[1])))
          encoder_final_state = tuple(new_c_h_tupe_list)
        else:
          raise NotImplementedError('not implement, decoder:%s, encoder:%s.'% (
              PARAM.decoder_unit_type, PARAM.encoder_unit_type))


      else:
        assert PARAM.encoder_num_units == PARAM.decoder_num_units, 'encoder_num_units != decoder_num_units and not projection'

      # decoder
      self.logits, self.sample_id, _, rnn_outputs_for_sampled_sotmax = (
        self._build_decoder(encoder_outputs, encoder_final_state)
      )  # build_decoder, encoder_outputs is not used (for attention)

    trainable_variables = tf.trainable_variables()
    self.save_variables.extend([var for var in trainable_variables])
    # self.saver = tf.train.Saver(tf.trainable_variables(),
    # self.saver = tf.train.Saver(tf.global_variables(),
    self.saver = tf.train.Saver(self.save_variables,
                                max_to_keep=PARAM.num_keep_ckpts,
                                save_relative_paths=True)

    # infer end
    if self.mode == PARAM.MODEL_INFER_KEY:
      # sample_words for decoding
      self.reverse_target_vocab_table = lookup_ops.index_to_string_table_from_file(
          self.tgt_vocab_file, default_value=vocab_utils.UNK) # ids -> words
      self.sample_words = self.reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))
      return

    # loss TODO
    crossent = self._softmax_cross_entropy_loss(
        self.logits, rnn_outputs_for_sampled_sotmax, self.target_out_id_seq)
    # mat_loss:[time, batch] if time_major else [batch, time]
    # loss: shape=()
    self.mat_loss, self.loss = loss_utils.masked_cross_entropy_loss(self.logits,
                                                                    crossent,
                                                                    rnn_outputs_for_sampled_sotmax,
                                                                    self.target_out_id_seq,
                                                                    self.target_seq_lengths,
                                                                    self.batch_size)
    # val summary
    # self.val_summary = tf.summary.merge(
    #   [tf.summary.scalar('val_loss', self.loss)]
    # )

    # Count the number of predicted words for compute ppl.
    self.word_count = tf.reduce_sum(
      self.source_seq_lengths+self.target_seq_lengths)
    self.predict_count = tf.reduce_sum(self.target_seq_lengths)

    # ppl(perplexity)
    '''
    ppl_per_sentense = exp(sum[-log(y_est_i)]/num_word)
    -log(y_est_i) = cross_entropy = -y_true*log(y_est)
    so: ppl_per_sentense = exp(reduce_mean(cross_entropy,time_axis))
    '''
    if PARAM.time_major:
      self.batch_sum_ppl = tf.reduce_sum(tf.exp(tf.reduce_mean(self.mat_loss, 0))) # reduce_sum batch
    else:
      self.batch_sum_ppl = tf.reduce_sum(tf.exp(tf.reduce_mean(self.mat_loss, -1))) # reduce_sum batch

    # val end
    if self.mode == PARAM.MODEL_VALIDATE_KEY:
      return

    # region apply gradient
    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.

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
        self.loss,
        params,
        colocate_gradients_with_ops=PARAM.colocate_gradients_with_ops)
    clipped_grads, grad_norm_summary, grad_norm = misc_utils.gradient_clip(
        gradients, max_gradient_norm=PARAM.max_gradient_norm)
    self.grad_norm_summary = grad_norm_summary
    self.grad_norm = grad_norm

    self.train_op = opt.apply_gradients(zip(clipped_grads, params),
                                        global_step=self.global_step)
    # endregion

    # Train Summary
    self.train_summary = tf.summary.merge(
        [tf.summary.scalar("lr", self.learning_rate),
         tf.summary.scalar("train_loss", self.loss)] +
        self.grad_norm_summary
    )


  def _softmax_cross_entropy_loss(
    self, logits, rnn_outputs_for_sampled_sotmax, labels):
    """Compute softmax loss or sampled softmax loss."""
    if PARAM.time_major:
      labels = tf.transpose(labels)
    if PARAM.num_sampled_softmax > 0:

      is_sequence = (rnn_outputs_for_sampled_sotmax.shape.ndims == 3)

      inputs = rnn_outputs_for_sampled_sotmax
      if is_sequence:
        labels = tf.reshape(labels, [-1, 1]) # [time*batch, 1] if time_major else [batch*time, 1]
        inputs = tf.reshape(rnn_outputs_for_sampled_sotmax,
                            [-1, PARAM.decoder_num_units])  # [time*batch, depth] if time_major else [batch*time, depth]

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
      # print(labels.get_shape().as_list(),logits.get_shape().as_list())
      # logits :[batch, time, vocab_num]
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      # print(crossent.get_shape().as_list())
      # crossent :[batch, time]
    return crossent

  @abc.abstractmethod
  def _build_encoder(self, seq, seq_lengths):
    """
    Returns:
      a tuple: (encoder_outputs, encoder_final_state)
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "_build_encoder not implement, code: dkfkasd0395jfa0295jf")

  @abc.abstractmethod
  def _build_decoder(self,
                     encoder_outputs,
                     encoder_final_state):
    """
    Returns:
      A tuple: (final_logits, decoder_final_state)
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "_build_decoder not implement, code: jqwirjjg992jgaldjf-0")

  def change_lr(self, sess, new_lr):
    sess.run(self.assign_lr, feed_dict={self.new_lr:new_lr})



class RNNSeq2SeqModel(BaseModel):
  def _build_encoder(self,seq,seq_lengths):
    """
    Args:
      seq: [batch, time]
      seq_lengths: [batch]
    Returns:
      encoder_outputs: encoder hidden outputs, [batch, time, ...]
      encoder_final_state: encoder state, [layers, 2(c,h), batch, n_units]
    """
    if PARAM.time_major:
      seq = tf.transpose(seq)

    with tf.variable_scope("encoder"):
      encoder_inputs = tf.nn.embedding_lookup(self.embedding_encoder,
                                              seq)
      if PARAM.encoder_type == 'uni':
        misc_utils.printinfo("  encoder_num_layers = %d, encoder_layer_start_residual=%d" %
                             (PARAM.encoder_num_layers,
                              PARAM.encoder_layer_start_residual), self.log_file)
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
        # self._debug=encoder_state[0][0].get_shape().as_list()
        # encoder_state: [layers, 2(c,h), batch, units]
      elif PARAM.encoder_type == "bi":
        assert PARAM.encoder_num_layers*2 == PARAM.decoder_num_layers, "2*encoder_layers == decoder_layers if bidirectional rnn used."
        fw_multi_cell = model_helper.multiRNNCell(
            unit_type=PARAM.encoder_unit_type,
            num_units=PARAM.encoder_num_units,
            num_layers=PARAM.encoder_num_layers,
            layer_start_residual=PARAM.encoder_layer_start_residual,
            forget_bias=PARAM.encoder_forget_bias,
            droprate=PARAM.encoder_drop_rate,
            mode=self.mode,
            num_gpus=PARAM.num_gpus,
            stack_bi_rnn=PARAM.stack_bi_rnn,
        )
        bw_multi_cell = model_helper.multiRNNCell(
            unit_type=PARAM.encoder_unit_type,
            num_units=PARAM.encoder_num_units,
            num_layers=PARAM.encoder_num_layers,
            layer_start_residual=PARAM.encoder_layer_start_residual,
            forget_bias=PARAM.encoder_forget_bias,
            droprate=PARAM.encoder_drop_rate,
            mode=self.mode,
            num_gpus=PARAM.num_gpus,
            stack_bi_rnn=PARAM.stack_bi_rnn,
        )
        if PARAM.stack_bi_rnn:
          fw_multi_cell = fw_multi_cell._cells
          bw_multi_cell = bw_multi_cell._cells
          bi_outputs, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
              fw_multi_cell,
              bw_multi_cell,
              encoder_inputs,
              dtype=self.dtype,
              sequence_length=seq_lengths,
              time_major=PARAM.time_major,
          )
          encoder_outputs, bi_encoder_state = tf.concat(bi_outputs,-1), (fw_state, bw_state)
        else:
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

        #bi_encoder_state : [2(fw,bw), layers, 2(c,h), batch, units]
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

    return encoder_outputs, encoder_state

  def _build_decoder_cell(
          self, encoder_outputs, encoder_final_state):
    del encoder_outputs # no use for vanilla model
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
    """
    Args:
      a tuple: (encoder_outputs, encoder_final_state)
    Returns:
      A tuple: (rnn_noproj_outputs, logits, sample_id, decoder_final_state)
        logits dim: [time, batch_size, vocab_size] when time_major=True
    """
    tgt_sos_id = tf.cast(
      self.tgt_vocab_table.lookup(tf.constant(PARAM.sos)), tf.int32)
    tgt_eos_id = tf.cast(
      self.tgt_vocab_table.lookup(tf.constant(PARAM.eos)), tf.int32)
    start_tokens = tf.fill([self.batch_size], tgt_sos_id)
    end_token = tgt_eos_id

    if PARAM.tgt_max_len_infer:
      max_rnn_iterations = PARAM.tgt_max_len_infer
      misc_utils.printinfo("  decoding max_rnn_iterations %d" % max_rnn_iterations)
    else:
      max_src_len = tf.reduce_max(self.source_seq_lengths)
      max_rnn_iterations = tf.to_int32(tf.round(
        tf.to_float(max_src_len) * PARAM.tgt_max_len_infer_factor))

    with tf.variable_scope("decoder") as decoder_scope:
      # region decoder
      cell, decoder_init_state = self._build_decoder_cell(encoder_outputs, encoder_final_state)
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
              seed=time.time())
        elif PARAM.infer_mode == "greedy":
          helper = contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)
        else:
          raise ValueError("Unknown infer_mode '%s'", PARAM.infer_mode)

        if PARAM.infer_mode != "beam_search":
          decoder = contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_init_state,
              output_layer=self.output_layer  # outside outputlayer for flexible
          )

        # Dynamic decoding
        decoder_outputs, final_context_state, _ = contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=max_rnn_iterations,
            output_time_major=PARAM.time_major,
            swap_memory=True,
            scope=decoder_scope
        )
        rnn_outputs_for_sampled_sotmax = tf.no_op()
        if PARAM.infer_mode == "beam_search":
          logits = tf.no_op() # # beam_search decoder have no logits (just scores)
          sample_id = decoder_outputs.predicted_ids
        else:
          logits = decoder_outputs.rnn_output
          sample_id = decoder_outputs.sample_id

      ## train or val
      else:
        '''
        decoder_inputs: target_in_id_seq
        decoder_init_state: encoder_final_state
        '''
        decoder_inputs = tf.nn.embedding_lookup(self.embedding_decoder, # ids->embeding
                                                self.target_in_id_seq) # [batch, time, embedding_vec]
        if PARAM.time_major:
          decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2]) # [time, batch, embedding_vec]

        decoder_helper = contrib.seq2seq.TrainingHelper(decoder_inputs,
                                                        self.target_seq_lengths,
                                                        time_major=PARAM.time_major)

        decoder = contrib.seq2seq.BasicDecoder(
            cell,
            decoder_helper,
            decoder_init_state,
            # output_layer=self.output_layer # outside outputlayer for sampled_softmax
        )

        # Dynamic decoding
        decoder_outputs, final_context_state, _ = contrib.seq2seq.dynamic_decode(
            decoder,
            output_time_major=PARAM.time_major,
            swap_memory=True,
            scope=decoder_scope
        )

        # if BasicDecoder->output_layer=self.output_layer, cannot use sampled_sotmax
        rnn_outputs_for_sampled_sotmax = decoder_outputs.rnn_output

        # if BasicDecoder->output_layer=None
        logits = self.output_layer(decoder_outputs.rnn_output)
        sample_id = tf.argmax(logits, axis=-1, name='get_sample_id', output_type=tf.int32)

        # if BasicDecoder->output_layer=self.output_layer
        # logits = decoder_outputs.rnn_output
        # sample_id2 = decoder_outputs.sample_id #

        # with tf.control_dependencies([tf.assert_equal(sample_id,sample_id2)]): # never equal
        #   sample_id = tf.identity(sample_id2)

      # self._debug=logits.get_shape().as_list()
      # print(self._debug)
      decoder_final_state = final_context_state
    return logits, sample_id, decoder_final_state, rnn_outputs_for_sampled_sotmax
