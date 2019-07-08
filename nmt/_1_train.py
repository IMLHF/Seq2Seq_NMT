import collections
import os
# import sys
import tensorflow as tf
import time

from .FLAGS import PARAM
from .models import model_builder
# from .utils import dataset_utils
from .utils import misc_utils


class ValOneEpochOutputs(
    collections.namedtuple("ValOneEpochOutputs",
                           ("average_bleu", "average_ppl",
                            "average_loss", "duration"))):
  pass

def val_one_epoch(log_file, summary_writer, val_sgmd):
  val_loss = 0
  data_len = 0
  s_time = time.time()
  val_sgmd.session.run(val_sgmd.dataset.initializer)

  while True:
    try:
      (loss,
       # val_summary,
       # current_bs,
       ) = (val_sgmd.session.run(
        [
          val_sgmd.model.loss,
          # val_sgmd.model.val_summary,
          # val_sgmd.model.batch_size,
        ]))
      val_loss += loss
      data_len += 1
    except tf.errors.OutOfRangeError:
      break

  val_loss /= data_len
  e_time = time.time()
  # tf.summary.scalar('val_loss')
  return ValOneEpochOutputs(average_loss=val_loss,
                            duration=e_time-s_time)


class TrainOneEpochOutputs(
    collections.namedtuple("TrainOneEpochOutputs",
                           ("average_loss", "duration", "learning_rate"))):
  pass

def train_one_epoch(log_file, summary_writer, train_sgmd):
  tr_loss, i, lr = 0.0, 0, -1
  s_time = time.time()
  train_sgmd.session.run(train_sgmd.dataset.initializer)

  while True:
    try:
      (_, loss, lr, summary_train, global_step,
       ) = (train_sgmd.session.run([
           train_sgmd.model.train_op,
           train_sgmd.model.loss,
           train_sgmd.model.learning_rate,
           train_sgmd.model.train_summary,
           train_sgmd.model.global_step
       ]))
      tr_loss += loss
      summary_writer.add_summary(summary_train, global_step)
      # msg = 'batchstep, loss:%.4f, lr:%.4f.' % (loss, lr)
      # misc_utils.printinfo(msg, log_file)
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss /= i
  e_time = time.time()
  return TrainOneEpochOutputs(average_loss=tr_loss,
                              duration=e_time-s_time,
                              learning_rate=lr)


def main(exp_dir,
         log_dir,
         summary_dir,
         ckpt_dir,
         log_file):  # train
  # sgmd : session, graph, model, dataset
  train_sgmd = model_builder.build_train_model(log_file, ckpt_dir, PARAM.scope)
  val_sgmd = model_builder.build_val_model(log_file, ckpt_dir, PARAM.scope)
  # misc_utils.show_all_variables(train_sgmd.graph)
  misc_utils.show_variables(train_sgmd.model.save_variables, train_sgmd.graph)

  # finalize graph
  train_sgmd.graph.finalize()
  val_sgmd.graph.finalize()

  # Summary writer
  summary_writer = tf.summary.FileWriter(summary_dir, train_sgmd.graph)
  
  # region validation before training
  valOneEpochOutputs_prev = val_one_epoch(log_file, summary_writer, val_sgmd)
  val_msg = "\n\nPRERUN AVG.LOSS %.4F  costime %ds\n" % (
      valOneEpochOutputs_prev.average_loss,
      valOneEpochOutputs_prev.duration)
  misc_utils.printinfo(val_msg, log_file)

  # train epochs
  assert PARAM.start_epoch > 0, 'start_epoch > 0 is required.'
  best_ckpt_name = None
  lr_halving_time = 0
  for epoch in range(PARAM.start_epoch, PARAM.max_epoch+1):
    # train
    trainOneEpochOutput = train_one_epoch(log_file, summary_writer, train_sgmd)
    train_sgmd.model.saver.save(train_sgmd.session,
                                os.path.join(ckpt_dir,'tmp'))

    # validation (loss, ppl, bleu)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    tf.logging.set_verbosity(tf.logging.WARN)
    val_sgmd.model.saver.restore(val_sgmd.session,
                                 ckpt.model_checkpoint_path)
    tf.logging.set_verbosity(tf.logging.INFO)
    valOneEpochOutputs = val_one_epoch(log_file, summary_writer, val_sgmd)
    val_loss_rel_impr = 1.0 - (valOneEpochOutputs.average_loss / valOneEpochOutputs_prev.average_loss)
    

    # save or abandon ckpt
    ckpt_name = PARAM.config_name+('_iter%d_trloss%.4f_valloss%.4f_lr%.4f_duration%ds' % (
        epoch, trainOneEpochOutput.average_loss, valOneEpochOutputs.average_loss,
        trainOneEpochOutput.learning_rate, trainOneEpochOutput.duration+valOneEpochOutputs.duration))
    if valOneEpochOutputs.average_loss < valOneEpochOutputs_prev.average_loss:
      train_sgmd.model.saver.save(train_sgmd.session,
                                  os.path.join(ckpt_dir, ckpt_name))
      valOneEpochOutputs_prev = valOneEpochOutputs
      best_ckpt_name = ckpt_name
      msg = ("\nEpoch : %03d\n"
             "        trloss:%.4f, valloss:%.4f, lr%e, duration:%ds.\n"
             "        %s saved.") % (
          epoch, trainOneEpochOutput.average_loss,
          valOneEpochOutputs.average_loss,
          trainOneEpochOutput.learning_rate,
          trainOneEpochOutput.duration+valOneEpochOutputs.duration,
          best_ckpt_name,
      )
    else:
      train_sgmd.model.saver.restore(train_sgmd.session,
                                     os.path.join(ckpt_dir, best_ckpt_name))
      msg = ("\nEpoch : %03d\n"
             "        trloss:%.4f, valloss:%.4f, lr%e, duration:%ds."
             "        %s abandoned.") % (
              epoch, trainOneEpochOutput.average_loss,
              valOneEpochOutputs.average_loss,
              trainOneEpochOutput.learning_rate,
              trainOneEpochOutput.duration + valOneEpochOutputs.duration,
              best_ckpt_name,
            )
    # prt
    misc_utils.printinfo(msg, log_file)

    # start lr halving
    if val_loss_rel_impr < PARAM.start_halving_impr:
      new_lr = trainOneEpochOutput.learning_rate * PARAM.lr_halving_rate
      lr_halving_time += 1
      train_sgmd.model.change_lr(train_sgmd.session, new_lr)

    # stop criterion
    if epoch >= PARAM.max_epoch or lr_halving_time > PARAM.max_lr_halving_time:
      msg = "finished, too small learning rate %e." % trainOneEpochOutput.learning_rate
      tf.logging.info(msg)
      misc_utils.printinfo(msg, log_file, noPrt=True)
      break

  train_sgmd.session.close()
  val_sgmd.session.close()
  msg = '################### Training Done. ###################'
  tf.logging.info(msg)
  misc_utils.printinfo(msg, log_file, noPrt=True)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(*misc_utils.ini_task('train')) # generate log in '"train_"+PARAM.config_name+".log"'
