import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from . import vanilla_model
from . import rnn_attention_model
from ..utils import misc_utils
from ..FLAGS import PARAM

def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
  '''Sinusoidal Positional_Encoding. See 3.5 in paper "Attention is all your need."
  inputs: 3d tensor. (N, T, E)
  maxlen: scalar. Must be >= T
  masking: Boolean. If True, padding positions are set to zeros.
  scope: Optional scope for `variable_scope`.

  returns
  3d tensor that has the same shape as inputs.
  '''

  E = inputs.get_shape().as_list()[-1]  # static
  N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # position indices
    position_ind = tf.tile(tf.expand_dims(
        tf.range(T), 0), [N, 1])  # (N, T)

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, (i-i % 2)/E) for i in range(E)]
        for pos in range(maxlen)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = tf.convert_to_tensor(
        position_enc, tf.float32)  # (maxlen, E)

    # lookup
    outputs = tf.nn.embedding_lookup(position_enc, position_ind)

    # masks
    if masking:
        outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

    return tf.to_float(outputs)


def attention_score_mask(scores, KV_lengths, mask_value=None):
  """
  Args:
    scores: [batch, time_query, time_kv], src_seq_length[0] is true length of scores[0, *]
    KV_lengths: [batch,], keys and values lengths.
  Return:
    masked_scores: [batch, time_query, timekv]
  Others:
    mask before softmax.
  """
  if mask_value is None:
    # mask_value = dtypes.as_dtype(scores.dtype).as_numpy_dtype(-np.inf)
    mask_value = -2 ** 32 + 1.0
  time_kv = tf.shape(scores)[2]
  mask = tf.sequence_mask(KV_lengths, maxlen=time_kv) # [batch, time_kv]
  mask = tf.expand_dims(mask, 1) # [batch, 1, time_kv]
  mask = tf.tile(mask, [1, tf.shape(scores)[1], 1]) # [batch, time_query, time_kv]
  score_mask_values = mask_value * tf.ones_like(scores)
  return tf.where(mask, scores, score_mask_values)


def causality_mask_for_self_attention(inputs, mask_value=None):
  """
  mask before softmax.
  """
  if mask_value is None:
    # mask_value = dtypes.as_dtype(inputs.dtype).as_numpy_dtype(-np.inf)
    mask_value = -2 ** 32 + 1.0
  diag_vals = tf.ones_like(inputs[0, :, :])  # (time_query, time_kv), always have "time_query == time_kv"
  tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (time_query, time_kv)
  masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (batch, time_query, time_kv)

  paddings = tf.ones_like(masks) * mask_value
  outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
  return outputs


def query_time_mask_for_train(inputs, query_lengths):
  """
  Args:
    inputs: [batch, time_query, time_kv], src_seq_length[0] is true length of scores[0, *]
    query_lengths: [batch,]
  Return:
    masked_scores: [batch, time_query, time_kv]
  Others:
    mask after softmax. before is ok so.
    same to rnn_seq_lengths, no use for inference.
    action as sequence_mask.
  """
  # if mask_value is None:
  #   mask_value = 0.0
  time_query = tf.shape(inputs)[1]
  mask = tf.sequence_mask(query_lengths, maxlen=time_query, dtype=inputs.dtype) # [batch, time_query]
  mask = tf.expand_dims(mask, 2) # [batch, time_query, 1]
  mask = tf.tile(mask, [1, 1, tf.shape(inputs)[2]]) # [batch, time_query, time_kv]
  # score_mask_values = mask_value * tf.ones_like(inputs)
  # return tf.where(mask, inputs, score_mask_values)
  outputs = tf.multiply(inputs, mask)
  return outputs


def scaled_dot_product_attention(Q, K, V, KV_lengths, Q_lengths=None,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
  '''See 3.2.1.
  Q: Packed queries. 3d tensor. [N, T_q, d_k].
  K: Packed keys. 3d tensor. [N, T_k, d_k].
  V: Packed values. 3d tensor. [N, T_k, d_v].
  causality: If True, applies masking for future blinding
  dropout_rate: A floating point number of [0, 1].
  training: boolean for controlling droput
  scope: Optional scope for `variable_scope`.
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    d_k = Q.get_shape().as_list()[-1]

    # dot product
    #
    K_T = tf.transpose(K, [0, 2, 1])
    outputs = tf.matmul(Q, K_T)  # (N, T_q, T_k)

    # scale
    outputs /= d_k ** 0.5

    # attention_score_mask
    outputs = attention_score_mask(outputs, KV_lengths)


    # causality or future blinding masking
    if causality:
      outputs = causality_mask_for_self_attention(outputs)

    # softmax
    outputs = tf.nn.softmax(outputs)
    attention = tf.transpose(outputs, [0, 2, 1])
    tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

    # query masking
    # remove query_time_mask_for_train and add tf.sequence_mask at calculate loss ?
    # if not PARAM.rm_query_mask:
    #   outputs = query_time_mask_for_train(outputs, Q_lengths)

    # dropout
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

    # weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

  return outputs


def layer_norm(inputs, epsilon=1e-8, scope="layer_norm"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries, keys, values,
                        d_model, KV_lengths, Q_lengths,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
  '''Applies multihead attention. See 3.2.2
  queries: A 3d tensor with shape of [N, T_q, d_model].
  keys: A 3d tensor with shape of [N, T_k, d_model].
  values: A 3d tensor with shape of [N, T_k, d_model].
  num_heads: An int. Number of heads.
  dropout_rate: A floating point number.
  training: Boolean. Controller of mechanism for dropout.
  causality: Boolean. If true, units that reference the future are masked.
  scope: Optional scope for `variable_scope`.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # Linear projections
    Q = tf.layers.dense(queries, d_model, use_bias=False) # (N, T_q, d_model)
    K = tf.layers.dense(keys, d_model, use_bias=False) # (N, T_k, d_model)
    V = tf.layers.dense(values, d_model, use_bias=False) # (N, T_k, d_model)

    # Split and concat
    assert d_model % num_heads == 0, "d_model % num_heads == 0 is required. d_model:%d, num_heads:%d." % (
        d_model, num_heads)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

    # Attention
    KV_lengths = tf.tile(KV_lengths, [num_heads])
    Q_lengths = tf.tile(Q_lengths, [num_heads])
    # print("QKV", Q_.get_shape().as_list(), K_.get_shape().as_list(), V_.get_shape().as_list(),)
    outputs = scaled_dot_product_attention(Q_, K_, V_, KV_lengths, Q_lengths,
                                           causality, dropout_rate, training)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, d_model)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = layer_norm(outputs)

  return outputs


def positionwise_FC(inputs, num_units, scope="positionwise_feedforward"):
  '''position-wise feed forward net. See 3.3

  inputs: A 3d tensor with shape of [N, T, C].
  num_units: A list of two integers.
  scope: Optional scope for `variable_scope`.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # Inner layer
    outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

    # Outer layer
    outputs = tf.layers.dense(outputs, num_units[1])

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = layer_norm(outputs)

  return outputs


class Transformer(rnn_attention_model.RNNAttentionModel):
  def _build_encoder(self, seq, src_seq_lengths):
    """
    Args:
      seq: src index sequence. [batch, time_src]
      src_seq_lengths: length list. [batch,]
    Returns:
      memory: encoder outputs. [batch, time_src, d_model]
    """
    if PARAM.transformer_use_rnn_encoder:
      return super(Transformer, self)._build_encoder(seq, src_seq_lengths)
    # param
    src_max_len = PARAM.src_max_len
    src_embed_size = PARAM.src_embed_size

    enc_droprate = PARAM.encoder_drop_rate
    enc_n_blocks = PARAM.encoder_num_layers

    enc_d_model = PARAM.encoder_num_units
    enc_num_heads = PARAM.enc_num_att_heads
    enc_d_positionwise_FC = PARAM.enc_d_positionwise_FC

    is_training = (self.mode == PARAM.MODEL_TRAIN_KEY)

    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      # inputs embedding
      enc = tf.nn.embedding_lookup(self.embedding_encoder,
                                   seq)
      enc *= src_embed_size**0.5 # scale

      enc += positional_encoding(enc, src_max_len)
      enc = tf.layers.dropout(enc, enc_droprate,
                              training=is_training)

      ## Blocks, use encoder_num_layers as encoder blocks
      for i in range(enc_n_blocks):
        with tf.variable_scope("blocks_{}".format(i), reuse=tf.AUTO_REUSE):
          # self-attention
          enc = multihead_attention(queries=enc,
                                    keys=enc,
                                    values=enc,
                                    d_model=enc_d_model,
                                    KV_lengths=src_seq_lengths,
                                    Q_lengths=src_seq_lengths,
                                    num_heads=enc_num_heads,
                                    dropout_rate=enc_droprate,
                                    training=is_training,
                                    causality=False)

          # position-wise feedforward
          enc = positionwise_FC(enc, num_units=[enc_d_positionwise_FC, enc_d_model])
    memory = enc # [batch, time_src, enc_d_model]
    return memory, None

  def _decoder_once(self, inputs, memory):
    """
    Args:
      inputs: [batch, time], decoder_in_id_seq
      memory: [batch, time_src, enc_d_model], encoder_outputs
    Others:
      self.target_seq_lengths
    """

    # param
    tgt_max_len = PARAM.tgt_max_len
    tgt_embed_size = PARAM.tgt_embed_size

    dec_droprate = PARAM.decoder_drop_rate
    dec_n_blocks = PARAM.decoder_num_layers

    dec_d_model = PARAM.decoder_num_units
    dec_num_heads = PARAM.dec_num_att_heads
    dec_d_positionwise_FC = PARAM.dec_d_positionwise_FC

    is_training = (self.mode == PARAM.MODEL_TRAIN_KEY)

    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      # embedding
      # TODO add calculate inputs_seq_lengths
      if is_training:
        dec_inputs_lengths = self.target_seq_lengths
      else:
        #inputs: [batch, time]
        tgt_eos_id = tf.cast(
            self.tgt_vocab_table.lookup(tf.constant(PARAM.eos)), tf.int32)
        ones = tf.ones_like(inputs)
        eos_mat = tgt_eos_id * ones
        zeros = tf.zeros_like(inputs)
        eos_onehot = tf.where(tf.equal(inputs, eos_mat), ones, zeros) # [batch, time]

        inputs_shape = tf.shape(inputs)
        cur_bs = inputs_shape[0]
        max_mask_len = inputs_shape[1]
        lengths = tf.constant([], dtype=tf.int32)

        def body(i, batch_size, lengths):
          eos_idx = tf.to_int32(tf.where(eos_onehot[i])) # [N, rank]
          cond = tf.equal(tf.shape(eos_idx)[0], 0) # tf.shape(eos_idx)[0]: num_true
          mask_len = tf.cond(cond, lambda:max_mask_len, lambda:eos_idx[0][0])
          lengths = tf.concat([lengths,[mask_len]],-1)
          return i+1, batch_size, lengths


        _, _, lengths = tf.while_loop(lambda i, batch_size, _: i < batch_size, body,
                                      (0, cur_bs, lengths),
                                      shape_invariants=(tf.TensorShape(None),
                                                        tf.TensorShape(None),
                                                        tf.TensorShape([None]),))

        dec_inputs_lengths = lengths

      dec = tf.nn.embedding_lookup(self.embedding_decoder,  # ids->embeding
                                   inputs)  # [batch, time_tgt, embedding_vec]
      dec *= tgt_embed_size ** 0.5
      dec += positional_encoding(dec, tgt_max_len)
      dec = tf.layers.dropout(dec, dec_droprate, training=is_training)

      ## Blocks
      for i in range(dec_n_blocks):
        with tf.variable_scope("blocks_{}".format(i), reuse=tf.AUTO_REUSE):
          # masked self_attention (causality system)
          KV_lengths = dec_inputs_lengths
          Q_lengths = dec_inputs_lengths
          dec = multihead_attention(queries=dec,
                                    keys=dec,
                                    values=dec,
                                    d_model=dec_d_model,
                                    KV_lengths=KV_lengths,
                                    Q_lengths=Q_lengths,
                                    num_heads=dec_num_heads,
                                    dropout_rate=dec_droprate,
                                    training=is_training,
                                    causality=PARAM.decoder_causality,
                                    scope="self_attention")

          # vanilla attention
          KV_lengths = self.source_seq_lengths
          Q_lengths = dec_inputs_lengths
          dec = multihead_attention(queries=dec,
                                    keys=memory,
                                    values=memory,
                                    d_model=dec_d_model,
                                    KV_lengths=KV_lengths,
                                    Q_lengths=Q_lengths,
                                    num_heads=dec_num_heads,
                                    dropout_rate=dec_droprate,
                                    training=is_training,
                                    causality=False,
                                    scope="vanilla_attention")

          # feed forward
          dec = positionwise_FC(dec, num_units=[dec_d_positionwise_FC, dec_d_model]) # [batch, time, dec_d_model]


      if PARAM.before_logits_is_tgt_embedding:
        if dec_d_model != tgt_embed_size:
          dec = tf.layers.dense(dec, tgt_embed_size, use_bias=False)
        weights = tf.transpose(self.embedding_decoder) # [tgt_embed_size, vocab_size]
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # [batch, time, vocab_size]
      else:
        logits = tf.layers.dense(dec, self.tgt_vocab_size, use_bias=False)

    sample_id = tf.argmax(logits, axis=-1, name='get_sample_id', output_type=tf.int32)
    # logits = tf.check_numerics(logits,"NaN_INF233333333333333%d"%1,name="tmp")
    return logits, sample_id, None, dec

  def _build_decoder(self, encoder_outputs, encoder_state):
    """
    Args:
      encoder_output: [batch, time_src, enc_d_model]
      encoder_state: None
    Others:
      self.target_seq_lengths
      self.target_in_id_seq, [<s>.id, idx...]
    """
    if PARAM.transformer_use_rnn_decoder:
      return super(Transformer, self)._build_decoder(encoder_outputs, encoder_state)

    del encoder_state
    if self.mode == PARAM.MODEL_TRAIN_KEY or self.mode == PARAM.MODEL_VALIDATE_KEY:
      logits, sample_id, _, dec = self._decoder_once(self.target_in_id_seq, encoder_outputs)
      return vanilla_model.DecodeOutputs(logits=logits,
                                         sample_id=sample_id,
                                         decoder_final_state=None,
                                         rnn_outputs_for_sampled_sotmax=dec)

    if self.mode == PARAM.MODEL_INFER_KEY:
      tgt_sos_id = tf.cast(
        self.tgt_vocab_table.lookup(tf.constant(PARAM.sos)), tf.int32)
      start_tokens = tf.fill([self.batch_size, 1], tgt_sos_id)

      if PARAM.tgt_max_len_infer:
        max_iter = PARAM.tgt_max_len_infer
        misc_utils.printinfo("  decoding max_decoder_iterations %d" % max_iter)
      else:
        max_src_len = tf.reduce_max(self.source_seq_lengths)
        max_iter = tf.to_int32(tf.round(
          tf.to_float(max_src_len) * PARAM.tgt_max_len_infer_factor))

      input_id_seq = start_tokens
      self.debug = input_id_seq

      if not PARAM.use_tf_while_loop_decode:
        for i in range(int(PARAM.tgt_max_len*PARAM.tgt_max_len_infer_factor)):
          logits, sample_id, _, dec = self._decoder_once(input_id_seq,
                                                         encoder_outputs)
          input_id_seq = tf.concat((start_tokens, sample_id), 1)
      else:
        def body(i, max_i, _, __, ___, input_id_seq_t):
          # input_id_seq [batch, time]
          logits, sample_id, _, dec = self._decoder_once(input_id_seq_t,
                                                         encoder_outputs)
          concat_sos_sample_id = tf.concat((start_tokens, sample_id), 1)
          return i+1, max_i, logits, sample_id, dec, concat_sos_sample_id
        _, _, logits, sample_id, dec, _ = tf.while_loop(lambda i, max_iter, _, __, ___, ____: i < max_iter,
                                                        body, (0, max_iter,
                                                               tf.zeros([1, 1, 1], dtype=self.dtype),
                                                               tf.zeros([1, 1], dtype=tf.int32),
                                                               tf.zeros([1, 1, 1], dtype=self.dtype),
                                                               input_id_seq),
                                                        shape_invariants=(tf.TensorShape(None), tf.TensorShape(None),
                                                                          tf.TensorShape([None, None, None]),
                                                                          tf.TensorShape([None, None]),
                                                                          tf.TensorShape([None, None, None]),
                                                                          tf.TensorShape([None, None])))

      # return logits, sample_id, None, dec
      return vanilla_model.DecodeOutputs(logits=logits,
                                         sample_id=sample_id,
                                         decoder_final_state=None,
                                         rnn_outputs_for_sampled_sotmax=dec)
