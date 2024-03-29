from distutils import version
import collections
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ..FLAGS import PARAM

def add_rootdir(_dir):
  return os.path.join(PARAM.root_dir,_dir)

def config_name():
  return PARAM.__class__name__(PARAM())

def ini_task(name):
  exp_dir = os.path.join(PARAM.root_dir,'exp',config_name())
  log_dir = os.path.join(exp_dir, 'log')
  ckpt_dir = os.path.join(exp_dir, 'ckpt')
  summary_dir = os.path.join(exp_dir, 'summary')
  log_file = os.path.join(log_dir, '%s_%s.log' % (name,config_name()))
  if not os.path.exists(exp_dir):
    printinfo('Output directory "%s" not exist, creating...' % exp_dir)
    os.makedirs(exp_dir)
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    printinfo('Log directory "%s" not exist. creating...' % log_dir, log_file)
  if not os.path.exists(summary_dir):
    printinfo('Summary directory "%s" not exist. creating...' % summary_dir, log_file)
    os.mkdir(summary_dir)
  if not os.path.exists(ckpt_dir):
    printinfo('CheckPoint directory "%s" not exist. creating...' % ckpt_dir, log_file)
    os.mkdir(ckpt_dir)

  printinfo("# save log at:%s" % log_file, log_file)

  # printinfo("# Visible Devices to TensorFlow %s." % repr(tf.Session().list_devices()),
  #           log_file)

  # print and save hparams
  print_hparams(not PARAM.verbose_print_hparams)
  hparams_file = os.path.join(exp_dir, 'hparam')
  save_hparams(hparams_file)
  # TODO ensure textline file have no empty line before filling into dataset.
  return exp_dir, log_dir, summary_dir, ckpt_dir, log_file


def printinfo(msg, f=None, new_line=True, noPrt=False):
  if new_line:
    msg += '\n'
  if not noPrt:
    print(msg, end='')
  if f:
    f = open(f,'a+')
    f.writelines(msg)
    f.close()
  sys.stdout.flush()


def save_hparams(f):
  f = open(f, 'a+')
  from .. import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  f.writelines('FLAGS.PARAM:\n')
  supper_dict = FLAGS.BaseConfig.__dict__
  for key in sorted(supper_dict.keys()):
    if key in self_dict_keys:
      f.write('%s:%s\n' % (key,self_dict[key]))
    else:
      f.write('%s:%s\n' % (key,supper_dict[key]))
  f.write('--------------------------\n\n')

  f.write('Short hparams:\n')
  [f.write("%s:%s\n" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  f.write('--------------------------\n\n')


def print_hparams(short=True):
  from .. import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  print('\n--------------------------\n')
  print('Short hparams:')
  [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  print('--------------------------\n')
  if not short:
    print('FLAGS.PARAM:')
    supper_dict = FLAGS.BaseConfig.__dict__
    for key in sorted(supper_dict.keys()):
      if key in self_dict_keys:
        print('%s:%s' % (key,self_dict[key]))
      else:
        print('%s:%s' % (key,supper_dict[key]))
    print('--------------------------\n')
    print('Short hparams:')
    [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
    print('--------------------------\n')


def get_session_config_proto():
  config_proto = tf.ConfigProto()
  config_proto.log_device_placement = PARAM.log_device_placement
  config_proto.allow_soft_placement = PARAM.allow_soft_placement
  config_proto.gpu_options.allow_growth = PARAM.gpu_allow_growth

  # CPU threads options
  if PARAM.num_intra_threads>0:
    config_proto.intra_op_parallelism_threads = PARAM.num_intra_threads
  if PARAM.num_inter_threads>0:
    config_proto.inter_op_parallelism_threads = PARAM.num_inter_threads
  return config_proto


def check_tensorflow_version():
  # LINT.IfChange
  min_tf_version = PARAM.min_TF_version
  # LINT.ThenChange(<pwd>/nmt/copy.bara.sky)
  if (version.LooseVersion(tf.__version__) <
          version.LooseVersion(min_tf_version)):
    raise EnvironmentError("Tensorflow version must >= %s" % min_tf_version)


def get_initializer(init_op, seed=None, init_weight=None):
  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(
        -init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(
        seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(
        seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
  return clipped_gradients, gradient_norm_summary, gradient_norm


def show_variables(vars_, graph):
  with graph.as_default():
    slim.model_analyzer.analyze_vars(vars_, print_info=True)
    sys.stdout.flush()


def show_all_variables(graph):
  with graph.as_default():
    model_vars = tf.global_variables()
    # model_vars = tf.trainable_variables()
    show_variables(model_vars, graph)


def add_summary(summary_writer, global_step, tag, value):
  """
  Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  summary_writer.add_summary(summary, global_step)


def get_translation_text_from_samplewords(sample_words, sentence_id, eos, subword_option):
  """
  sample_words: [batch, words]
  """
  if eos:
    eos = eos.encode("utf-8")

  sentence = sample_words[sentence_id].tolist()

  # If there is an eos symbol in outputs, cut them at that position.
  if eos and eos in sentence:
    sentence = sentence[:sentence.index(eos)]
    # sentence[sentence.index(eos)] = "233".encode("utf-8")

  if (not hasattr(sentence, "__len__") and  # for numpy array
          not isinstance(sentence, collections.Iterable)):
    sentence = [sentence]
  return b" ".join(sentence)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


if __name__ == '__main__':
  save_hparams('tes2t')
  # printinfo('testsmse')
  # printinfo('testmsfd','test',False)
  # printinfo('testmsfd','test')
