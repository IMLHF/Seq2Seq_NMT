import sys
import os
from FLAGS import PARAM
import tensorflow as tf
from distutils import version


def ini_task(name):
  exp_dir = os.path.join(PARAM.root_dir,'exp',PARAM.config_name)
  log_dir = os.path.join(exp_dir, 'log')
  ckpt_dir = os.path.join(exp_dir, 'ckpt')
  summary_dir = os.path.join(exp_dir, 'summary')
  log_file = os.path.join(log_dir, '%s_%s.log' % (name,PARAM.config_name))
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

  printinfo("# Visible Devices to TensorFlow %s." % repr(tf.Session().list_devices()),
            log_file)

  # print and save hparams
  print_hparams(not PARAM.verbose_print_hparams)
  hparams_file = os.path.join(exp_dir, 'hparam')
  save_hparams(hparams_file)
  return exp_dir, log_dir, summary_dir, ckpt_dir, log_file

def printinfo(msg, f=None, new_line=True):
  if new_line:
    msg += '\n'
  print(msg, end='')
  if f:
    f = open(f,'a+')
    f.writelines(msg)
    f.close()
  sys.stdout.flush()

def save_hparams(f):
  f = open(f, 'a+')
  import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  f.writelines('FLAGS.PARAM:\n')
  supper_dict = FLAGS.base_config.__dict__
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
  import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  if not short:
    print('FLAGS.PARAM:')
    supper_dict = FLAGS.base_config.__dict__
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
  config_proto = tf.ConfigProto(
    log_device_placement=PARAM.log_device_placement,
    allow_soft_placement=PARAM.allow_soft_placement)
  config_proto.gpu_options.allow_growth = PARAM.gpu_allow_growth

  # CPU threads options
  if PARAM.num_intra_threads>0:
    config_proto.intra_op_parallelism_threads = PARAM.num_intra_threads
  if PARAM.num_inter_threads>0:
    config_proto.inter_op_parallelism_threads = PARAM.num_inter_threads


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


if __name__ == '__main__':
  save_hparams('tes2t')
  # printinfo('testsmse')
  # printinfo('testmsfd','test',False)
  # printinfo('testmsfd','test')
