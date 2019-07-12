import collections
import os
# import sys
import tensorflow as tf
import time

from .FLAGS import PARAM
from .models import model_builder
# from .utils import dataset_utils
from .utils import misc_utils

class TestOneEpochOutputs(
    collections.namedtuple("TestOneEpochOutputs",
                           ("average_bleu", "duration"))):
  pass

def test_one_epoch(log_file, summary_writer, test_sgmd):
  pass

def main(exp_dir,
         log_dir,
         summary_dir,
         ckpt_dir,
         log_file):  # test
  pass


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(*misc_utils.ini_task('test')) # generate log in '"test_"+PARAM.__class__.__name__+".log"'
