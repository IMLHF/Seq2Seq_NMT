
class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_INFER_KEY = 'infer'
  MODEL_VALIDATE_KEY = 'val'

class BaseConfig(StaticKey):
  # root_dir = '/mnt/d/OneDrive/workspace/tf_recipe/Seq2Seq_NMT'
  # root_dir = '/mnt/f/OneDrive/workspace/tf_recipe/Seq2Seq_NMT'
  root_dir = '/home/room/worklhf/nmt_seq2seq_first/'
  config_name = 'base'
  min_TF_version = "1.12.0"
  num_keep_ckpts = 30
  '''
  # dir to store log, model and results files:
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/summary: tensorboard summary
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/decode: decode results
  $root_dir/exp/$config_name/hparams
  '''

  # dataset
  output_buffer_size = None
  reshuffle_each_iteration = True
  num_parallel_calls = 8

  # region network
  encoder_num_units = 32
  decoder_num_units = 32
  encoder_num_layers = 2  # Encoder depth, equal to num_layers if None.
  decoder_num_layers = 2  # Decoder depth, equal to num_layers if None.
  encoder_layer_start_residual = 3
  decoder_layer_start_residual = 3

  model_type = 'vanilla'
  '''
  vanilla | standard_attention | gnmt | gnmt_current
  vanilla: vanilla seq2seq model
  standard: use top layer to compute attention.
  gnmt: GNMT style of computing attention, use pervious bottom layer to compute attention.
  gnmt_current: similar to gnmt, but use current bottom layer to compute attention.
  '''

  attention = 'luong'
  '''
  luong | scaled_luong | bahdanau | normed_bahdanau or set to '' for no attention
  '''

  encoder_type = 'uni'
  '''
  uni | bi
  For bi, we build num_encoder_layers/2 bi-directional layers.
  For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1) uni-directional layers.
  '''

  residual = False # ?
  time_major= False # Whether to use time-major mode for dynamic RNN.
  # num_embeddings_partitions = 0 # ?

  standard_output_attention = True # Only used in standard attention_architecture. Whether use attention as the cell output at each timestep.
  pass_hidden_state = True # Whether to pass encoder's hidden state to decoder when using an attention based model.
  # endregion

  # regiion optimizer & lr halving & stop criterion
  start_halving_impr = 0.001
  lr_halving_rate = 0.5
  max_lr_halving_time = 4

  optimizer = 'adam' # 'sgd' or 'adam'
  loss = 'cross_entropy' #
  learning_rate = 0.01 # Adam: 0.001 or 0.0001; SGD: 1.0.

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
  src_max_len = 50
  tgt_max_len = 50
  src_max_len_infer = None
  tgt_max_len_infer = None
  tgt_max_len_infer_factor = 2.0 # tgt_max_len_infer = src_seq_max_len*tgt_max_len_infer_factor
  encoder_unit_type = 'lstm'
  decoder_unit_type = 'lstm' # lstm | gru | layer_norm_lstm | nas
  encoder_forget_bias = 1.0
  decoder_forget_bias = 1.0
  encoder_drop_rate = 0.2
  decoder_drop_rate = 0.2
  max_gradient_norm = 5.0 # gradient clip
  batch_size = 1024
  batches_to_logging = 100
  max_train = 0 # Limit on the size of training data (0: no limit).
  num_buckets = 5 # if > 1; Bucket sentence pairs by the length of their source sentence and target sentence.
  num_sampled_softmax = 0 # Use sampled_softmax_loss if > 0, else full softmax loss.
  subword_option = '' # ?

  # Misc
  num_gpus = 1
  metrics = 'bleu,rouge,accuracy'.split(',') # Comma-separated list of evaluations "metrics (bleu,rouge,accuracy)"
  steps_per_external_eval = None
  scope = None # model scope
  avg_ckpts = False # Average the last N checkpoints for external evaluation. N can be controlled by setting --num_keep_ckpts.

  # inference
  infer_mode = 'greedy' # "greedy", "sample", "beam_search"
  beam_width = 0 # beam width for beam search decoder. If 0, use standard decoder with greedy helper.
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
  max_epoch = 50

  #################
  # Extra
  #################
  language_model = False # True to train a language model, ignoring encoder
  warmup_steps = 0 # steps lr warmup
  warmup_scheme = 't2t'
  lr_decay_scheme = None
  '''
  "luong234" | "luong5" | "luong10" | None
  How we decay learning rate. Options include:
      luong234: after 2/3 num train steps, we start halving the learning rate
        for 4 times before finishing.
      luong5: after 1/2 num train steps, we start halving the learning rate
        for 5 times before finishing.
      luong10: after 1/2 num train steps, we start halving the learning rate
        for 10 times before finishing.
      None: no decay.
  '''

  use_char_encode = False # ?


class TEST_C(BaseConfig):
  pass


PARAM = TEST_C


if __name__ == '__main__':
  pass
