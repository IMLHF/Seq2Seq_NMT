import collections
import codecs
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


class ValOrTestOutputs(
    collections.namedtuple("ValOneEpochOutputs",
                           ("val_scores", "average_ppl",
                            "average_loss", "duration"))):
  pass


def val_or_test(exp_dir, log_file, src_textline_file, tgt_textline_file,
                summary_writer, epoch, val_sgmd, infer_sgmd):
  """
  Args:
    exp_dir : $PARAM.root_dir/exp/$PARAM.config_name
    val_sgmd : for validation&test. get loss, ppl.
    infer_sgmd : for validation&test. get bleu, rouge, accracy.
  """
  s_time = time.time()

  # region val_model to calculate loss, ppl
  val_loss = 0 # total loss
  total_ppl = 0
  data_len = 0 # batch_num*batch_size:dataset records num
  # total_predict_len, loss_sum_batchandtime = 0, 0.0 # for ppl2 at github:tensorflow/nmt

  val_sgmd.session.run(val_sgmd.dataset.initializer,
                       feed_dict={val_sgmd.dataset.src_textline_file_ph:src_textline_file,
                                  val_sgmd.dataset.tgt_textline_file_ph:tgt_textline_file})

  while True:
    try:
      (loss, # reduce_mean batch&time
       batch_sum_ppl, # reduce_sum batch && reduce_mean time
       current_bs,
       #  predict_len, # for ppl2
       #  mat_loss, # for ppl2
       # val_summary,
       #  mat_loss,
       ) = (val_sgmd.session.run(
           [
            val_sgmd.model.loss,
            val_sgmd.model.batch_sum_ppl,
            val_sgmd.model.batch_size,
            # val_sgmd.model.predict_count, # for ppl2
            # val_sgmd.model.mat_loss, # for ppl2
            # val_sgmd.model.val_summary,
            # val_sgmd.model.logits,
            ]))
      val_loss += loss
      total_ppl += batch_sum_ppl
      data_len += current_bs

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
  # endregion val_model

  # region infer_mode to calculate bleu, rouge, accuracy
  trans_file = os.path.join(exp_dir, 'val_set_translate_result_iter%04d.txt' % epoch)
  trans_f = codecs.getwriter("utf-8")(tf.gfile.GFile(trans_file, mode="wb"))
  trans_f.write("")  # Write empty string to ensure file is created.

  infer_sgmd.session.run(infer_sgmd.dataset.initializer,
                         feed_dict={infer_sgmd.dataset.src_textline_file_ph: src_textline_file,
                                    infer_sgmd.dataset.tgt_textline_file_ph: tgt_textline_file})
  while True:
    try:
      (sample_words, # words list, text, dim:[beam_width, batch_size, words] if beam_search else [batch_size, words]
       current_bs,
       ) = (infer_sgmd.session.run(
           [
            infer_sgmd.model.sample_words,
            infer_sgmd.model.batch_size,
            ]))

      # translated text
      if PARAM.infer_mode == 'beam_search':
        sample_words = np.array(sample_words[0]) # [batch_size, words]
      # print(current_bs, sample_words.shape[0])
      assert current_bs == sample_words.shape[0], 'batch_size exception.'
      for sentence_id in range(current_bs):
        translation = misc_utils.get_translation_text_from_samplewords(sample_words,
                                                                       sentence_id,
                                                                       eos=PARAM.eos,
                                                                       subword_option=PARAM.subword_option)
        trans_f.write((translation+b"\n").decode("utf-8"))
    except tf.errors.OutOfRangeError:
      break

  trans_f.close()

  # evaluation scores like bleu, rouge, accuracy etc.
  eval_scores = {}
  for metric in PARAM.metrics:
    eval_scores[metric] = eval_utils.evalute(
      ref_textline_file=tgt_textline_file,
      trans_textline_file=trans_file,
      metric=metric,
      subword_option=PARAM.subword_option
    )
  # endregion infer_model

  # summary
  misc_utils.add_summary(summary_writer, epoch, "epoch_val_loss", avg_val_loss)

  e_time = time.time()
  return ValOrTestOutputs(average_loss=avg_val_loss,
                          duration=e_time-s_time,
                          val_scores=eval_scores,
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
        msg = "    Minbatch %04d: loss:%.4f, duration:%ds." % (
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
  # train_mode : for training. get loss.
  train_sgmd = model_builder.build_train_model(log_file, ckpt_dir, PARAM.scope)
  misc_utils.show_variables(train_sgmd.model.save_variables, train_sgmd.graph)

  # val_model : for validation&test. get loss, ppl.
  val_sgmd = model_builder.build_val_model(log_file, ckpt_dir, PARAM.scope)

  # infer_model : for validation&test. get bleu, rouge, accracy.
  infer_sgmd = model_builder.build_infer_model(log_file, ckpt_dir, PARAM.scope)
  # misc_utils.show_all_variables(train_sgmd.graph)

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
  test_set_textlinefile_src = "%s.%s" % (PARAM.test_prefix, PARAM.src)
  test_set_textlinefile_tgt = "%s.%s" % (PARAM.test_prefix, PARAM.tgt)
  train_set_textlinefile_src = misc_utils.add_rootdir(train_set_textlinefile_src)
  train_set_textlinefile_tgt = misc_utils.add_rootdir(train_set_textlinefile_tgt)
  val_set_textlinefile_src = misc_utils.add_rootdir(val_set_textlinefile_src)
  val_set_textlinefile_tgt = misc_utils.add_rootdir(val_set_textlinefile_tgt)
  test_set_textlinefile_src = misc_utils.add_rootdir(test_set_textlinefile_src)
  test_set_textlinefile_tgt = misc_utils.add_rootdir(test_set_textlinefile_tgt)


  # validation before training
  valOneEpochOutputs_prev = val_or_test(exp_dir, log_file,
                                        val_set_textlinefile_src,
                                        val_set_textlinefile_tgt,
                                        summary_writer, 0,
                                        val_sgmd, infer_sgmd)
  # score_smg: str(" BLEU:XX.XXXX, ROUGE:X.XXXX,")
  scores_msg = eval_utils.scores_msg(valOneEpochOutputs_prev.val_scores, upper_case=True)
  val_msg = "\n\nPRERUN.val> LOSS:%.4F, PPL:%.4F," % (valOneEpochOutputs_prev.average_loss,
                                                      valOneEpochOutputs_prev.average_ppl) + \
      scores_msg + " cost_time %ds" % valOneEpochOutputs_prev.duration
  misc_utils.printinfo(val_msg, log_file)

  # test before training
  testOneEpochOutputs = val_or_test(exp_dir, log_file,
                                    test_set_textlinefile_src,
                                    test_set_textlinefile_tgt,
                                    summary_writer, 0,
                                    val_sgmd, infer_sgmd)
  scores_msg = eval_utils.scores_msg(testOneEpochOutputs.val_scores, upper_case=True)
  val_msg = "PRERUN.test> LOSS:%.4F, PPL:%.4F," % (testOneEpochOutputs.average_loss,
                                                   testOneEpochOutputs.average_ppl) + \
      scores_msg + " cost_time %ds\n" % testOneEpochOutputs.duration
  misc_utils.printinfo(val_msg, log_file)

  # add initial epoch_train_loss
  misc_utils.add_summary(summary_writer, 0, "epoch_train_loss",
                         valOneEpochOutputs_prev.average_loss+testOneEpochOutputs.average_loss)

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
    misc_utils.printinfo("    Train> loss:%.4f, lr:%.2e, duration:%ds.\n" % (
        trainOneEpochOutput.average_loss,
        trainOneEpochOutput.learning_rate,
        trainOneEpochOutput.duration),
        log_file)

    # validation (loss, ppl, scores(bleu...))
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    tf.logging.set_verbosity(tf.logging.WARN)
    val_sgmd.model.saver.restore(val_sgmd.session,
                                 ckpt.model_checkpoint_path)
    infer_sgmd.model.saver.restore(infer_sgmd.session,
                                   ckpt.model_checkpoint_path)
    tf.logging.set_verbosity(tf.logging.INFO)
    valOneEpochOutputs = val_or_test(exp_dir, log_file,
                                     val_set_textlinefile_src, val_set_textlinefile_tgt,
                                     summary_writer, epoch, val_sgmd, infer_sgmd)
    val_loss_rel_impr = 1.0 - (valOneEpochOutputs.average_loss / valOneEpochOutputs_prev.average_loss)
    misc_utils.printinfo("    Val  > loss:%.4f, ppl:%.4f, bleu:%.4f, rouge:%.4f, accuracy:%.4f, duration %ds\n" % (
        valOneEpochOutputs.average_loss,
        valOneEpochOutputs.average_ppl,
        valOneEpochOutputs.val_scores["bleu"],
        valOneEpochOutputs.val_scores["rouge"],
        valOneEpochOutputs.val_scores["accuracy"],
        valOneEpochOutputs.duration),
        log_file)

    # test (loss, ppl, scores(bleu...))
    testOneEpochOutputs = val_or_test(exp_dir, log_file,
                                      test_set_textlinefile_src, test_set_textlinefile_tgt,
                                      summary_writer, epoch, val_sgmd, infer_sgmd)
    misc_utils.printinfo("    Test > loss:%.4f, ppl:%.4f, bleu:%.4f, rouge:%.4f, accuracy:%.4f, duration %ds\n" % (
        testOneEpochOutputs.average_loss,
        testOneEpochOutputs.average_ppl,
        testOneEpochOutputs.val_scores["bleu"],
        testOneEpochOutputs.val_scores["rouge"],
        testOneEpochOutputs.val_scores["accuracy"],
        testOneEpochOutputs.duration),
        log_file)

    # save or abandon ckpt
    ckpt_name = PARAM.config_name+('_iter%d_trloss%.4f_valloss%.4f_valppl%.4f_lr%.2e_duration%ds' % (
        epoch, trainOneEpochOutput.average_loss, valOneEpochOutputs.average_loss, valOneEpochOutputs.average_ppl,
        trainOneEpochOutput.learning_rate, trainOneEpochOutput.duration+valOneEpochOutputs.duration))
    if valOneEpochOutputs.average_loss < valOneEpochOutputs_prev.average_loss:
      train_sgmd.model.saver.save(train_sgmd.session,
                                  os.path.join(ckpt_dir, ckpt_name))
      valOneEpochOutputs_prev = valOneEpochOutputs
      best_ckpt_name = ckpt_name
      msg = "    ckpt(%s) saved.\n" % ckpt_name
    else:
      tf.logging.set_verbosity(tf.logging.WARN)
      train_sgmd.model.saver.restore(train_sgmd.session,
                                     os.path.join(ckpt_dir, best_ckpt_name))
      tf.logging.set_verbosity(tf.logging.INFO)
      msg = "    ckpt(%s) abandoned.\n" % ckpt_name
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
  infer_sgmd.session.close()
  msg = '################### Training Done. ###################'
  tf.logging.info(msg)
  misc_utils.printinfo(msg, log_file, noPrt=True)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(*misc_utils.ini_task('train')) # generate log in '"train_"+PARAM.config_name+".log"'
