import tensorflow as tf

from . import vanilla_model
from . import model_helper
from ..FLAGS import PARAM

__all__ = ["RNNAttentionModel"]

class RNNAttentionModel(vanilla_model.RNNSeq2SeqModel):
  def _prepare_beam_search_decoder_inputs(
          self, beam_width, memory, source_sequence_length, encoder_state):
    memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=beam_width)
    source_sequence_length = tf.contrib.seq2seq.tile_batch(
        source_sequence_length, multiplier=beam_width)
    encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=beam_width)
    batch_size = self.batch_size * beam_width
    return memory, source_sequence_length, encoder_state, batch_size

  def _create_attention_mechanism(self, num_units, memory, source_sequence_length):
    """
    Args:
      num_units: The depth of the attention (query) mechanism.
      memory: The memory to query; usually the output of an RNN encoder. This tensor should be shaped [batch_size, max_time, ...].
      source_sequence_length: (optional) Sequence lengths for the batch entries in memory. If provided, the memory tensor rows are masked with zeros for values past the respective sequence lengths.
    Returns:
      Instance of tf.contrib.seq2seq.AttentionMachanism()
    """
    if PARAM.attention == "luong":
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
          num_units, memory, memory_sequence_length=source_sequence_length)
    elif PARAM.attention == "scaled_luong":
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
          num_units,
          memory,
          memory_sequence_length=source_sequence_length,
          scale=True)
    elif PARAM.attention == "bahdanau":
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          num_units, memory, memory_sequence_length=source_sequence_length)
    elif PARAM.attention == "normed_bahdanau":
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          num_units,
          memory,
          memory_sequence_length=source_sequence_length,
          normalize=True)
    else:
      raise ValueError("Unknown attention type %s" % PARAM.attention)
    return attention_mechanism


  def _build_decoder_cell(self, encoder_outputs, encoder_state):
    # memory [batch, time, ...]
    memory = encoder_outputs
    if PARAM.time_major:
      memory = tf.transpose(encoder_outputs, [1,0,2])

    source_seq_lengths = self.source_seq_lengths

    # beam_search
    batch_size = self.batch_size
    if self.mode == PARAM.MODEL_INFER_KEY and PARAM.infer_mode == 'beam_search':
      memory, source_seq_lengths, encoder_state, batch_size = (
          self._prepare_beam_search_decoder_inputs(PARAM.beam_width, memory,
                                                   source_seq_lengths, encoder_state))

    # attention_mechanism
    attention_mechanism = self._create_attention_mechanism(PARAM.decoder_num_units,
                                                           # num_unit: set query dim to project to key dim
                                                           memory,
                                                           source_seq_lengths)

    # rnn_cell
    rnn_cell = model_helper.multiRNNCell(
        unit_type=PARAM.decoder_unit_type,
        num_units=PARAM.decoder_num_units,
        num_layers=PARAM.decoder_num_layers,
        layer_start_residual=PARAM.decoder_layer_start_residual,
        forget_bias=PARAM.decoder_forget_bias,
        droprate=PARAM.decoder_drop_rate,
        mode=self.mode,
        num_gpus=PARAM.num_gpus
    )

    # alignment (only in greedy INFER mode)
    alignment_history = (self.mode == PARAM.MODEL_INFER_KEY and PARAM.infer_mode == 'greedy')
    attentioned_cell = tf.contrib.seq2seq.AttentionWrapper(
      rnn_cell,
      attention_mechanism,
      attention_layer_size=PARAM.attention_layer_size,
      alignment_history=alignment_history,
      cell_input_fn=PARAM.attention_cell_input_fn,
      output_attention=PARAM.output_attention,
      name='attention'
    )

    # whether pass encoder_state to decoder
    if PARAM.pass_state_using_attention:
      decoder_initial_state = attentioned_cell.zero_state(batch_size, self.dtype).clone(
          cell_state=encoder_state)
    else:
      decoder_initial_state = attentioned_cell.zero_state(batch_size, self.dtype)

    return attentioned_cell, decoder_initial_state
