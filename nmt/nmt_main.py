import tensorflow as tf
import sys
import os

from .utils import misc_utils
from .FLAGS import PARAM
from . import _1_train
from . import _2_test


def main(_):
  exp_dir, log_dir, summary_dir, ckpt_dir, log_file = misc_utils.ini_task('train_and_test')

  _1_train.main(exp_dir=exp_dir,
                log_dir=log_dir,
                summary_dir=summary_dir,
                ckpt_dir=ckpt_dir,
                log_file=log_file)

  _2_test.main(exp_dir=exp_dir,
               log_dir=log_dir,
               summary_dir=summary_dir,
               ckpt_dir=ckpt_dir,
               log_file=log_file)


if __name__ == '__main__':
  config_name = str(sys.argv[0]).split("/")[-2]
  assert config_name == misc_utils.config_name(), (
      'PARAM_class_name != dir_name. PARAM_name:%s, dir_name:%s.' % (config_name, misc_utils.config_name()))
  os.environ['CUDA_VISIBLE_DEVICES'] = PARAM.VISIBLE_GPU
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main, argv=sys.argv)
  # tensorboard --port 22222 --logdir /tmp/nmt_model/
