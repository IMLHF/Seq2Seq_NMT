from FLAGS import PARAM
import tensorflow as tf
import sys
from utils import misc_utils
import os
import _1_train
import _2_test


def main():
  # num_worker = PARAM.num_workers
  misc_utils.printinfo("# Visible Devices to TensorFlow %s." % repr(tf.Session().list_devices()))

  exp_dir = os.path.join(PARAM.root_dir,'exp')
  if not os.path.exists(exp_dir):
    misc_utils.printinfo('Output directory "exp" not exist, creating...')

  # print hparams
  misc_utils.print_hparams(not PARAM.verbose_print_hparams)

  _1_train.main()
  _2_test.main()


if __name__ == '__main__':
  tf.app.run(main=main, argv=sys.argv)
