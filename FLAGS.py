
class base_config(object):
    config_name = 'base'
    num_units = 32
    num_layers = 2
    num_encoder_layers = None  # Encoder depth, equal to num_layers if None.
    num_decoder_layers = None  # Decoder depth, equal to num_layers if None.

    encoder_type = 'uni'
    '''
    uni | bi | gnmt
    For bi, we build num_encoder_layers/2 bi-directional layers.
    For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1) uni-directional layers.
    '''

    residual = False # ?
    time_major= False
    num_embeddings_partitions = 0 # ?
    attention = ''
    '''
    luong | scaled_luong | bahdanau | normed_bahdanau or set to '' for no attention
    '''

    attention_architecture = 'standard'
    '''
    standard | gnmt | gnmt_v2 | gnmt_current.
    standard: use top layer to compute attention.
    gnmt: GNMT style of computing attention, use pervious bottom layer to compute attention.
    gnmt_v2: similar to gnmt, but use current bottom layer to compute attention.
    gnmt_current: same to gnmt_v2.
    '''

    standard_output_attention = True # Only used in standard attention_architecture. Whether use attention as the cell output at each timestep.
    pass_hidden_state = True # Whether to pass encoder's hidden state to decoder when using an attention based model.
    optimizer = 'adam' # 'sgd' or 'adam'
    learning_rate = 0.001 # Adam: 0.001 or 0.0001; SGD: 1.0.
    warmup_steps = 0 # ?
    warmup_scheme = 't2t' # ?
    lr_decay_scheme = ''
    '''
    luong234 | luong5 | luong10
    How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
          for 5 times before finishing.
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.
    '''

    num_train_steps = 12000
    colocate_gradients_with_ops = True # ?
    init_op = 'uniform' # 'uniform' or 'glorot_normal' or 'glorot_uniform'
    init_weight = 0.1 # for uniform init_op, initialize weights between [-this, this].

    src = None # Source suffix, e.g., en. Must be assigned.
    tgt = None # Target suffix.
    train_prefix = None # Train prefix, expect files with src/tgt suffixes.
    dev_prefix = None # Dev prefix.
    test_prefix = None # Test prefix.
    out_dir = None # Store log, model and results files.
    '''
    $out_dir/$config_name/nnet: ckpt
    $out_dir/$config_name/decode: decode results
    $out_dir/$config_name/log: logs
    '''

    vocab_prefix = None # Vocab prefix, expect files with src/tgt suffixes.
    embed_prefix = None
    '''
    Pretrained embedding prefix, expect files with src/tgt suffixes.
    The embedding files should be Glove formated txt files.
    '''

    sos = '<s>' # Start-of-sentence symbol.
    eos = '</s>' # End- of-sentence symbol.
    share_vocab = True # ?
    chech_special_token = True # ?
    src_max_len = 50
    tgt_max_len = 50
    src_max_len_infer = None
    tgt_max_len_infer = None
    unit_type = 'lstm' # lstm | gru | layer_norm_lstm | nas
    forget_bias = 1.0
    drop_rate = 0.2
    max_gradient_norm = 5.0 # gradient clip
    batch_size = 128
    steps_to_logging = 100
    max_train = 0 # Limit on the size of training data (0: no limit).
    num_buckets = 5 # ?
    num_sampled_softmax = 0 # Use sampled_softmax_loss if > 0, else full softmax loss.
    subword_option = '' # ?

    # Misc
    use_char_encoder = False # ?
    num_gpus = 1
    log_device_placement = False # ?
    metrics = 'bleu' # Comma-separated list of evaluations "metrics (bleu,rouge,accuracy)"
    steps_per_external_eval = None
    scope = None # ?
    hparams_path = None # Path to standard hparams json file that overrides hparams values from FLAGS.
    override_loaded_hparams = False # ?
    random_seed = None
    avg_ckpts = False # Average the last N checkpoints for external evaluation. N can be controlled by setting --num_keep_ckpts.
    language_model = False # ? True to train a language model, ignoring encoder

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
    jobid = 0 # Task id of the worker.
    num_workers = 1 # Number of workers (inference only).
    num_inter_threads = 0 # ?number of inter_op_parallelism_threads.
    num_intra_threads = 0 # ?number of intra_op_parallelism_threads.
