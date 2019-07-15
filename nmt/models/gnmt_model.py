import tensorflow as tf

from . import model_helper
from . import rnn_attention_model
from ..utils import misc_utils
from ..FLAGS import PARAM

__all__ = ["GNMTAttentionModel"]

class GNMTAttentionModel(rnn_attention_model.RNNAttentionModel):
  def _build_encoder(self, seq, seq_lengths):
    """
    Args:
      seq: [batch, time]
      seq_lengths: [batch]
    Returns:
      encoder_outputs: [time, batch, ...] if time_major else [batch, time, ...]
      encoder_state: [layer, 2(c, h: for lstm), batch, ...]
    """
    if PARAM.encoder_type == "uni" or PARAM.encoder_type == "bi":
      return super(GNMTAttentionModel, self)._build_decoder(seq, seq_lengths)
    if PARAM.encoder_type != "gnmt":
      raise ValueError("Unknown encoder type (%s) for gnmt_model." % PARAM.encoder_type)

    # GNMT model
    num_bi_layers = 1
    num_uni_layers = PARAM.encoder_num_layers - num_bi_layers
    misc_utils.printinfo("\n# Build a GNMT encoder", self.log_file)
    misc_utils.printinfo("  num_bi_layers = %d" % num_bi_layers)
    misc_utils.printinfo("  num_uni_layers = %d" % num_uni_layers)

    if PARAM.time_major:
      seq = tf.transpose(seq)
      # seq [time, batch, ...]

    with tf.variable_scope("encoder_gnmt"):
      encoder_inputs = tf.nn.embedding_lookup(self.embedding_decoder, seq)

      # region GNMT bidirectional RNN in encoder
      fw_multi_cell = model_helper.multiRNNCell(
          unit_type=PARAM.encoder_unit_type,
          num_units=PARAM.encoder_num_units,
          num_layers=num_bi_layers,
          layer_start_residual=99999, # no residual
          forget_bias=PARAM.encoder_forget_bias,
          droprate=PARAM.encoder_drop_rate,
          mode=self.mode,
          num_gpus=PARAM.num_gpus,
      )
      bw_multi_cell = model_helper.multiRNNCell(
          unit_type=PARAM.encoder_unit_type,
          num_units=PARAM.encoder_num_units,
          num_layers=num_bi_layers,
          layer_start_residual=99999,
          forget_bias=PARAM.encoder_forget_bias,
          droprate=PARAM.encoder_drop_rate,
          mode=self.mode,
          num_gpus=PARAM.num_gpus,
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
        bi_outputs, bi_state = tf.concat(bi_outputs,-1), (fw_state, bw_state)
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
        bi_outputs, bi_state = tf.concat(bi_outputs,-1), bi_state

      # bi_outputs: [time, batch, ...] if time_major else [batch, time, ...]
      # bi_state : [2(fw,bw), layers, 2(c,h), batch, units]
      # endregion GNMT bi-RNN in encoder

      # region get encoder_outputs, encoder_state
      assert num_uni_layers > 0, "GNMT encoder: num_uni_layers > 0 is required. now :%d." % num_uni_layers
      multi_uni_cell = model_helper.multiRNNCell(
          unit_type=PARAM.encoder_unit_type,
          num_units=PARAM.encoder_num_units,
          num_layers=num_uni_layers,
          layer_start_residual=PARAM.encoder_layer_start_residual-num_bi_layers,
          forget_bias=PARAM.encoder_forget_bias,
          droprate=PARAM.encoder_drop_rate,
          mode=self.mode,
          num_gpus=PARAM.num_gpus
      )
      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          multi_uni_cell,
          bi_outputs,
          dtype=self.dtype,
          sequence_length=seq_lengths,
          time_major=PARAM.time_major,
          swap_memory=True
      )


  def _build_decoder_cell(self, encoder_outputs, encoder_state):
    # GNMT attention

    memory = encoder_outputs # [time, batch, ...] if time_major else [batch, time, ...]
    if PARAM.time_major:
      # ensure memory is [batch, time, ...]
      memory = tf.transpose(encoder_outputs, [1,0,2]) # [batch, time, ...]

    # for beam_search
    batch_size = self.batch_size
    if (self.mode == tf.contrib.learn.ModeKeys.INFER and
            PARAM.infer_mode == "beam_search"):
      memory, source_seq_lengths, encoder_state, batch_size = (
          self._prepare_beam_search_decoder_inputs(PARAM.beam_width, memory,
                                                   self.source_seq_lengths, encoder_state))

    # attention mechanism
    attention_mechanism = self._create_attention_mechanism(PARAM.decoder_num_units,
                                                           # num_unit: set query dim to project to key dim
                                                           memory,
                                                           source_seq_lengths)
    # rnn_cells
    rnn_cells, fist_rnn_cell = model_helper.multiRNNCell(
        unit_type=PARAM.decoder_unit_type,
        num_units=PARAM.decoder_num_units,
        num_layers=PARAM.decoder_num_layers,
        layer_start_residual=PARAM.decoder_layer_start_residual,
        forget_bias=PARAM.decoder_forget_bias,
        droprate=PARAM.decoder_drop_rate,
        mode=self.mode,
        num_gpus=PARAM.num_gpus,
        GNMT_decoder=True, # get fist rnn_cell
        residual_fn=model_helper.gnmt_residual_fn,
    )

    # Only wrap the bottom layer with the attention mechanism.
    attention_cell = fist_rnn_cell

    # alignment (only in greedy INFER mode)
    alignment_history = (self.mode == PARAM.MODEL_INFER_KEY and PARAM.infer_mode == 'greedy')
    attentioned_cell = tf.contrib.seq2seq.AttentionWrapper(
      attention_cell,
      attention_mechanism,
      attention_layer_size=None, # not use attention layer in GNMT.
      output_attention=False, # not use attention as outputs
      alignment_history=alignment_history,
      name='GNMTAttention'
    )

    cell = GNMTAttentionMultiCell(attentioned_cell, rnn_cells._cells,
                                  use_new_attention=PARAM.GNMT_current_attention)

    if PARAM.pass_state_E2D:
      decoder_initial_state = tuple(
          zs.clone(cell_state=es)
          if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
          for zs, es in zip(
              cell.zero_state(batch_size, self.dtype), encoder_state))
    else:
      decoder_initial_state = cell.zero_state(batch_size, self.dtype)

    return cell, decoder_initial_state


class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell"):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]

          if self.use_new_attention:
            cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
          else:
            cur_inp = tf.concat([cur_inp, attention_state.attention], -1)

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)
