import collections
import os
import sys
import tensorflow as tf
import time

from FLAGS import PARAM
from models import model_builder
from utils import dataset_utils
from utils import misc_utils


class EvalOneEpochOutputs(
    collections.namedtuple("EvalOneEpochOutputs",
                           ("average_loss", "duration"))):
  pass

def eval_one_epoch(log_file, val_sgmd):
  val_loss = 0
  data_len = 0
  s_time = time.time()
  val_sgmd.session.run(val_sgmd.dataset.initializer)

  while True:
    try:
      (loss,
       #  current_bs,
       ) = (val_sgmd.session.run([
           val_sgmd.model.loss,
           # val_sgmd.model.batch_size,
       ]))
      val_loss += loss
      data_len + 1
    except tf.errors.OutOfRangeError:
      break

  val_loss /= data_len
  e_time = time.time()
  return EvalOneEpochOutputs(average_loss=val_loss,
                             duration=e_time-s_time)


class TrainOneEpochOutputs(
    collections.namedtuple("TrainOneEpochOutputs",
                           ("average_loss", "duration", "learning_rate"))):
  pass

def train_one_epoch(log_file, train_sgmd):
  tr_loss, i = 0.0, 0
  s_time = time.time()
  train_sgmd.session.run(train_sgmd.dataset.initializer)

  while True:
    try:
      (_, loss, lr,
       ) = (train_sgmd.session.run([
           train_sgmd.model.train_op,
           train_sgmd.model.loss,
           train_sgmd.model.learning_rate,
       ]))
      tr_loss += loss
      msg = 'batchstep, loss:%.4f, lr:%.4f.' % (loss, lr)
      misc_utils.printinfo(msg, log_file)
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss /= i
  e_time = time.time()
  return TrainOneEpochOutputs(average_loss=tr_loss,
                              druation=s_time-e_time,
                              learning_rate=lr)


def main(exp_dir,
         log_dir,
         summary_dir,
         ckpt_dir,
         log_file):  # train
  # gmd : session, graph, model, dataset
  train_sgmd = model_builder.build_train_model(log_file, ckpt_dir, PARAM.scope)
  val_sgmd = model_builder.build_val_model(log_file, ckpt_dir, PARAM.scope)

  # finalize graph
  train_sgmd.graph.finalize()
  val_sgmd.graph.finalize()

  # region validation before training
  evalOneEpochOutputs = eval_one_epoch(log_file, val_sgmd)
  val_msg = "\n\nPRERUN AVG.LOSS %.4F  costime %ds\n" % (
      evalOneEpochOutputs.average_loss,
      evalOneEpochOutputs.duration)
  misc_utils.printinfo(val_msg, log_file)

  # train epochs
  assert PARAM.start_epoch > 0, 'start_epoch > 0 is required.'
  for epoch in range(PARAM.start_epoch, PARAM.max_epoch+1):
    # train
    trainOneEpochOutput = train_one_epoch(log_file, train_sgmd)

    # eval
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    val_sgmd.model.saver.restore(val_sgmd.session,
                                 ckpt.model_checkpoint_path)
    evalOneEpochOutputs = eval_one_epoch(log_file, val_sgmd)

    # prt
    msg = ("\nEpoch : %03d\n"
           "trloss:%.4f, valloss:%.4f, lr%.4f, duration:%ds.") % (
             epoch, trainOneEpochOutput.average_loss,
             evalOneEpochOutputs.average_loss,
             trainOneEpochOutput.learning_rate,
             trainOneEpochOutput.duration+evalOneEpochOutputs.duration,
           )
    misc_utils.printinfo(msg, log_file)

    #
    ckpt_name = PARAM.config_name+'_iter%d_trloss%.4f_valloss%.4f_lr%.4f'
    train_sgmd.model.saver.save(train_sgmd.session,
                                os.path.join(ckpt_dir, ckpt_name))
  msg = '################### Training Done. ###################'
  tf.logging.info(msg)
  misc_utils.printinfo(msg, log_file, noPrt=True)


if __name__ == '__main__':
  main(*misc_utils.ini_task('train'))
