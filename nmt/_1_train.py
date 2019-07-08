import collections
import os
# import sys
import tensorflow as tf
import time
import numpy as np

from .FLAGS import PARAM
from .models import model_builder
# from .utils import dataset_utils
from .utils import eval_utils
from .utils import misc_utils


class ValOneEpochOutputs(
    collections.namedtuple("ValOneEpochOutputs",
                           ("average_bleu", "average_ppl",
                            "average_loss", "duration"))):
  pass

def val_one_epoch(log_file, src_textline_file, tgt_textline_file,
                  summary_writer, epoch, val_sgmd):
  val_loss = 0 # total loss
  data_len = 0 # batch_num*batch_size:dataset records num
  # total_predict_len, loss_sum_batchandtime = 0, 0.0 # for ppl2 at github:tensorflow/nmt
  total_ppl = 0
  s_time = time.time()
  val_sgmd.session.run(val_sgmd.dataset.initializer,
                       feed_dict={val_sgmd.dataset.src_textline_file_ph: src_textline_file,
                                  val_sgmd.dataset.tgt_textline_file_ph:tgt_textline_file})

  while True:
    try:
      (loss, # reduce_mean batch&time
       #  predict_len, # for ppl2
       #  mat_loss, # for ppl2
       batch_sum_ppl, # reduce_sum batch && reduce_mean time
       # val_summary,
       current_bs,
       ) = (val_sgmd.session.run(
           [
            val_sgmd.model.loss,
            # val_sgmd.model.predict_count, # for ppl2
            # val_sgmd.model.mat_loss, # for ppl2
            val_sgmd.model.batch_sum_ppl,
            # val_sgmd.model.val_summary,
            val_sgmd.model.batch_size,
            ]))
      val_loss += loss
      data_len += current_bs
      total_ppl += batch_sum_ppl

      # for ppl2
      # print(np.shape(mat_loss))
      # total_predict_len += predict_len
      # loss_sum_batchandtime += np.sum(mat_loss)

    except tf.errors.OutOfRangeError:
      break

  # ppl
  avg_ppl = total_ppl/data_len # reduce_mean batch&time

  # ppl2
  # avg_ppl2 = eval_utils.calc_ppl(loss_sum_batchandtime, total_predict_len)
  # print('debug——ppl2: %d' % avg_ppl2)

  # avg_loss
  avg_val_loss = val_loss / data_len

  # bleu
  # TODO

  e_time = time.time()
  misc_utils.add_summary(summary_writer, epoch, "epoch_val_loss", avg_val_loss)
  if epoch: # if epoch=0, pre_run not show msg.
    misc_utils.printinfo('        validation done, duration %ds' % (e_time-s_time), log_file)
  return ValOneEpochOutputs(average_loss=avg_val_loss,
                            duration=e_time-s_time,
                            average_bleu=None,
                            average_ppl=avg_ppl)


class TrainOneEpochOutputs(
    collections.namedtuple("TrainOneEpochOutputs",
                           ("average_loss", "duration", "learning_rate"))):
  pass

def train_one_epoch(log_file, src_textline_file, tgt_textline_file,
                    summary_writer, epoch, train_sgmd):
  tr_loss, i, data_len, lr = 0.0, 0, 0, -1
  s_time = time.time()
  minbatch_time = time.time()
  train_sgmd.session.run(train_sgmd.dataset.initializer,
                         feed_dict={train_sgmd.dataset.src_textline_file_ph: src_textline_file,
                                    train_sgmd.dataset.tgt_textline_file_ph:tgt_textline_file})

  while True:
    try:
      (_, loss, lr, summary_train, global_step, current_bs,
       #  mat_loss,
       ) = (train_sgmd.session.run([
           train_sgmd.model.train_op,
           train_sgmd.model.loss,
           train_sgmd.model.learning_rate,
           train_sgmd.model.train_summary,
           train_sgmd.model.global_step,
           train_sgmd.model.batch_size,
           #  train_sgmd.model.mat_loss,
       ]))
      # print(np.shape(mat_loss))
      tr_loss += loss
      data_len += current_bs
      summary_writer.add_summary(summary_train, global_step)
      # msg = 'batchstep, loss:%.4f, lr:%.4f.' % (loss, lr)
      # misc_utils.printinfo(msg, log_file)
      i += 1
      if i % PARAM.batches_to_logging == 0:
        msg = "        Minbatch %04d: loss:%.4f, duration:%ds." % (
                i, tr_loss/data_len, time.time()-minbatch_time,
              )
        minbatch_time = time.time()
        misc_utils.printinfo(msg, log_file)

    except tf.errors.OutOfRangeError:
      break
  tr_loss /= data_len
  e_time = time.time()
  misc_utils.add_summary(summary_writer, epoch, "epoch_train_loss", tr_loss)
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

  # dataset textline files
  train_set_textlinefile_src = "%s.%s" % (PARAM.train_prefix, PARAM.src)
  train_set_textlinefile_tgt = "%s.%s" % (PARAM.train_prefix, PARAM.tgt)
  val_set_textlinefile_src = "%s.%s" % (PARAM.val_prefix, PARAM.src)
  val_set_textlinefile_tgt = "%s.%s" % (PARAM.val_prefix, PARAM.tgt)
  train_set_textlinefile_src = misc_utils.add_rootdir(train_set_textlinefile_src)
  train_set_textlinefile_tgt = misc_utils.add_rootdir(train_set_textlinefile_tgt)
  val_set_textlinefile_src = misc_utils.add_rootdir(val_set_textlinefile_src)
  val_set_textlinefile_tgt = misc_utils.add_rootdir(val_set_textlinefile_tgt)

  # region validation before training
  valOneEpochOutputs_prev = val_one_epoch(log_file,
                                          val_set_textlinefile_src,
                                          val_set_textlinefile_tgt,
                                          summary_writer, None, val_sgmd)
  val_msg = "\n\nPRERUN AVG.LOSS %.4F, AVG.PPL %.4F, costime %ds\n" % (
      valOneEpochOutputs_prev.average_loss,
      valOneEpochOutputs_prev.average_ppl,
      valOneEpochOutputs_prev.duration)
  misc_utils.printinfo(val_msg, log_file)

  # add initial epoch_train_loss
  misc_utils.add_summary(summary_writer, 0, "epoch_train_loss", valOneEpochOutputs_prev.average_loss)

  # train epochs
  assert PARAM.start_epoch > 0, 'start_epoch > 0 is required.'
  best_ckpt_name = None
  lr_halving_time = 0
  for epoch in range(PARAM.start_epoch, PARAM.max_epoch+1):
    misc_utils.printinfo("Epoch : %03d" % epoch, log_file)
    # train
    trainOneEpochOutput = train_one_epoch(log_file,
                                          train_set_textlinefile_src,
                                          train_set_textlinefile_tgt,
                                          summary_writer, epoch, train_sgmd)
    train_sgmd.model.saver.save(train_sgmd.session,
                                os.path.join(ckpt_dir,'tmp'))

    # validation (loss, ppl, bleu)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    tf.logging.set_verbosity(tf.logging.WARN)
    val_sgmd.model.saver.restore(val_sgmd.session,
                                 ckpt.model_checkpoint_path)
    tf.logging.set_verbosity(tf.logging.INFO)
    valOneEpochOutputs = val_one_epoch(log_file,
                                       val_set_textlinefile_src, val_set_textlinefile_tgt,
                                       summary_writer, epoch, val_sgmd)
    val_loss_rel_impr = 1.0 - (valOneEpochOutputs.average_loss / valOneEpochOutputs_prev.average_loss)


    # save or abandon ckpt
    ckpt_name = PARAM.config_name+('_iter%d_trloss%.4f_valloss%.4f_valppl%.4f_lr%.2e_duration%ds' % (
        epoch, trainOneEpochOutput.average_loss, valOneEpochOutputs.average_loss, valOneEpochOutputs.average_ppl,
        trainOneEpochOutput.learning_rate, trainOneEpochOutput.duration+valOneEpochOutputs.duration))
    if valOneEpochOutputs.average_loss < valOneEpochOutputs_prev.average_loss:
      train_sgmd.model.saver.save(train_sgmd.session,
                                  os.path.join(ckpt_dir, ckpt_name))
      valOneEpochOutputs_prev = valOneEpochOutputs
      best_ckpt_name = ckpt_name
      msg = ("        trloss:%.4f, valloss:%.4f, valppl:%.4f, lr:%.2e, duration:%ds.\n"
             "        ckpt(%s) saved.\n") % (
          trainOneEpochOutput.average_loss,
          valOneEpochOutputs.average_loss,
          valOneEpochOutputs.average_ppl,
          trainOneEpochOutput.learning_rate,
          trainOneEpochOutput.duration+valOneEpochOutputs.duration,
          best_ckpt_name,
      )
    else:
      train_sgmd.model.saver.restore(train_sgmd.session,
                                     os.path.join(ckpt_dir, best_ckpt_name))
      msg = ("        trloss:%.4f, valloss:%.4f, valppl:%.4f, lr:%.2e, duration:%ds.\n"
             "        ckpt(%s) abandoned.\n") % (
              trainOneEpochOutput.average_loss,
              valOneEpochOutputs.average_loss,
              valOneEpochOutputs.average_ppl,
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
