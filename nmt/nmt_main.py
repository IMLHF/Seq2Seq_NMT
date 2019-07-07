import tensorflow as tf
import sys
# import os

from .utils import misc_utils
# from .FLAGS import PARAM
from . import _1_train
from . import _2_test


def main(_):
  # num_worker = PARAM.num_workers
  exp_dir, log_dir, summary_dir, ckpt_dir, log_file = misc_utils.ini_task('nmt_main')

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
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main, argv=sys.argv)
  # tensorboard --port 22222 --logdir /tmp/nmt_model/
