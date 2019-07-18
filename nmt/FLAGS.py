import tensorflow as tf

class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_INFER_KEY = 'infer'
  MODEL_VALIDATE_KEY = 'val'

  def __class__name__(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  # root_dir = '/mnt/d/OneDrive/workspace/tf_recipe/Seq2Seq_NMT'
  # root_dir = '/mnt/f/OneDrive/workspace/tf_recipe/Seq2Seq_NMT'
  root_dir = '/home/room/worklhf/nmt_seq2seq_first/'
  min_TF_version = "1.12.0"
  num_keep_ckpts = 2
  '''
  # dir to store log, model and results files:
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/summary: tensorboard summary
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/decode: decode results
  $root_dir/exp/$config_name/hparams
  '''

  # dataset
  output_buffer_size = 512*1000 # must larger than batch_size
  reshuffle_each_iteration = True
  num_parallel_calls = 8

  # region network
  src_embed_size = 512
  tgt_embed_size = 512
  encoder_num_units = 512
  decoder_num_units = 512
  projection_encoder_final_state = False
  encoder_num_layers = 2
  decoder_num_layers = 4
  encoder_layer_start_residual = 2 # layer id start at 1
  decoder_layer_start_residual = 2
  dtype=tf.float32
  stack_bi_rnn = False # stack_bidirectional_dynamic_rnn if True else bidirectional_dynamic_rnn

  model_type = 'vanilla'
  '''
  vanilla | standard_attention | gnmt | transformer
  vanilla: vanilla seq2seq model
  standard: use top layer to compute attention.
  gnmt: google neural machine translation model.
  transformer: transformer.
  '''

  attention = 'scaled_luong'
  '''
  for model_type == 'standard_attention'
  luong | scaled_luong | bahdanau | normed_bahdanau
  '''

  attention_layer_size = 512
  '''
  If None (default), use the context as attention at each time step. Otherwise, feed the context and cell output into the attention layer to generate attention at each time step. #  (not None is needed).
  '''

  GNMT_current_attention = True
  """
  for GNMT
   True: use current bottom layer to compute attention.
   False: use pervious bottom layer to compute attention.
  """
  attention_cell_input_fn = None # A callable. The default is: lambda inputs, attention: array_ops.concat([inputs, attention], -1).
  output_attention = True # Whether use attention as the decoder cell output at each timestep.
  pass_state_E2D = True # Whether to pass encoder's hidden state to decoder when using an attention based model.
  encoder_type = 'bi'
  '''
  uni | bi | gnmt
  For uni, we build encoder_num_layers unidirectional layers.
  For bi, we build encoder_num_layers bidirectional layers.
  For gnmt, we build 1 bidirectional layer, and (encoder_num_layers - 1) unidirectional layers.
  '''
  # num_embeddings_partitions = 0 # ?
  # endregion

  # regiion optimizer & lr halving & stop criterion
  start_halving_impr = 0.001
  lr_halving_rate = 0.5
  max_lr_halving_time = 4

  optimizer = 'sgd' # 'sgd' or 'adam'
  loss = 'cross_entropy' #
  learning_rate = 1.0 # Adam: 0.001 or 0.0001; SGD: 1.0.

  colocate_gradients_with_ops = True # params of tf.gradients() to parallel
  # endregion

  init_op = 'uniform' # 'uniform' or 'glorot_normal' or 'glorot_uniform'
  init_weight = 0.1 # for uniform init_op, initialize weights between [-this, this].

  src = "vi" # Source suffix, e.g., en. Must be assigned.
  tgt = "en" # Target suffix.
  train_prefix = "iwslt15/train" # Train prefix, expect files with src/tgt suffixes.
  val_prefix = "iwslt15/tst2012" # Dev prefix.
  test_prefix = "iwslt15/tst2013" # Test prefix.
  vocab_prefix = "iwslt15/vocab" # Vocab prefix, expect files with src/tgt suffixes.
  embedding_on_device = '/cpu:0'
  embed_prefix = None
  # '''
  # Pretrained embedding prefix, expect files with src/tgt suffixes.
  # The embedding files should be Glove formated txt files.
  # '''

  unk = '<unk>'
  sos = '<s>' # Start-of-sentence symbol.
  eos = '</s>' # End- of-sentence symbol.
  share_vocab = False # use same vocab table for source and target
  check_special_token = True # ?
  src_max_len = 50 # for train
  tgt_max_len = 50

  src_max_len_infer = None # for decode
  tgt_max_len_infer = None
  tgt_max_len_infer_factor = 2.0 # tgt_max_len_infer = src_seq_max_len*tgt_max_len_infer_factor

  encoder_unit_type = 'lstm'
  decoder_unit_type = 'lstm' # lstm | layer_norm_lstm | gru | nas, (gru and nas is not tested.)
  encoder_forget_bias = 1.0
  decoder_forget_bias = 1.0
  encoder_drop_rate = 0.3
  decoder_drop_rate = 0.3
  max_gradient_norm = 5.0 # gradient clip
  batch_size = 128
  batches_to_logging = 300
  max_train = 0 # Limit on the size of training data (0: no limit).
  num_buckets = 5 # if > 1; Bucket sentence pairs by the length of their source sentence and target sentence.
  num_sampled_softmax = 0 # Use sampled_softmax_loss if > 0, else full softmax loss. default=
  subword_option = None # method format sample_words to text, not use

  # Misc
  num_gpus = 1
  metrics = ["bleu", "rouge", "accuracy"]  # Comma-separated list of evaluations "metrics (bleu,rouge,accuracy)"
  val_criterion = 'loss' # "loss", "bleu", "rouge", "accuracy"
  steps_per_external_eval = None
  scope = None # model scope
  avg_ckpts = False # Average the last N checkpoints for external evaluation. N can be controlled by setting --num_keep_ckpts.

  # inference
  infer_mode = 'greedy' # "greedy", "sample", "beam_search"
  beam_width = None # beam width for beam search decoder. If 0, use standard decoder with greedy helper.
  length_penalty_weight = 0.0 # Length penalty for beam search.
  coverage_penalty_weight = 0.0 # Coverage penalty for beam search.
  sampling_temperature = 0.0
  '''
  Softmax sampling temperature for inference decoding, 0.0 means greedy decoding.
  This option is ignored when using beam search.
  '''

  num_translations_per_input = 1 # Number of translations generated for each sentence, only for inference.
  log_device_placement = False # config of tf.Session(), print log
  allow_soft_placement = True
  gpu_allow_growth = True
  num_inter_threads = 0 # ?number of inter_op_parallelism_threads.
  num_intra_threads = 0 # ?number of intra_op_parallelism_threads.

  verbose_print_hparams = True

  start_epoch = 1
  max_epoch = 100

  #################
  # Extra
  #################
  language_model = False # True to train a language model, ignoring encoder
  use_char_encode = False # ?


  ##########################
  # Hparam for Transformer
  ##########################
  # n_blocks_enc = encoder_num_layers
  # n_blocks_dec = decoder_num_layers
  # d_model = encoder_num_units = decoder_num_units = src_embed_size = tgt_embed_size
  enc_d_positionwise_FC = 2048
  dec_d_positionwise_FC = 2048
  enc_num_att_heads = 8
  dec_num_att_heads = 8
  before_logits_is_tgt_embedding = True


class C001_adam_greedy(BaseConfig): # DONE 15123
  optimizer = 'adam'
  learning_rate = 0.001
  infer_mode = 'greedy'

class C001_adam_greedy_projection(BaseConfig): # DONE 15123
  optimizer = 'adam'
  learning_rate = 0.001
  infer_mode = 'greedy'
  projection_encoder_final_state = True

class C001_adam_greedy_stackbirnn(BaseConfig): # DONE 15123
  optimizer = 'adam'
  learning_rate = 0.001
  infer_mode = 'greedy'
  stack_bi_rnn = True

class C002_adam_sample(BaseConfig): # DONE 15123
  optimizer = 'adam'
  learning_rate = 0.001
  infer_mode = 'sample'
  sampling_temperature = 1.0

class C003_1_adam_beam_search5(BaseConfig): # DONE 15123
  optimizer = 'adam'
  learning_rate = 0.001
  infer_mode = 'beam_search'
  beam_width = 5

class C003_2_adam_beam_search10(BaseConfig): # DONE 15123
  optimizer = 'adam'
  learning_rate = 0.001
  infer_mode = 'beam_search'
  beam_width = 10

class C004_attention_scaled_luong(BaseConfig): # DONE 15123
  model_type = 'standard_attention'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001

class C004_attention_scaled_luong_RNNoutput(BaseConfig): # DONE 15123
  model_type = 'standard_attention'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001
  output_attention = False

class C004_attention_scaled_luong_nostate(BaseConfig): # RUNNING 15123
  model_type = 'standard_attention'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001
  pass_state_E2D = False

class C004_attention_scaled_luong_deep(BaseConfig): # DONE 15123
  model_type = 'standard_attention'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001
  encoder_num_layers = 3
  decoder_num_layers = 6

class nmt_test(BaseConfig):
  model_type = 'standard_attention'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001

class gnmt_test_curFalse(BaseConfig):
  model_type = 'gnmt'
  encoder_type = 'bi'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001
  GNMT_current_attention = False

class gnmt_test_curTrue(BaseConfig): # better
  model_type = 'gnmt'
  encoder_type = 'bi'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001
  GNMT_current_attention = True

class C004_attention_scaled_luong_maskedlogits(BaseConfig):
  model_type = 'standard_attention'
  attention = 'scaled_luong'
  optimizer = 'adam'
  learning_rate = 0.001

class TransformerTest(BaseConfig):
  batch_size = 256
  encoder_num_layers = 3
  decoder_num_layers = 3
  model_type = 'transformer'
  optimizer = 'adam'
  learning_rate = 0.001
  before_logits_is_tgt_embedding = False



PARAM = TransformerTest

if __name__ == '__main__':
  pass
