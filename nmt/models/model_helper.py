import collections
import tensorflow as tf
import tensorflow.contrib as contrib

from FLAGS import PARAM
from utils import dataset_utils
from utils import vocab_utils


def _single_rnn_cell(unit_type, num_units, forget_bias, droprate, mode,
                     residual_connection=False, device_str=None, residual_fn=None,
                     activation_fun='tanh'):
  droprate = 0.0 if mode == PARAM.MODEL_INFER_KEY else droprate

  if unit_type == "lstm":
    single_cell = contrib.rnn.LSTMCell(num_units=num_units,
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       # state_is_tuple=True,
                                       activation=activation_fun,
                                       # use_peepholes=True,
                                       forget_bias=forget_bias)
  elif unit_type == "gru":
    single_cell = contrib.rnn.GRUCell(num_units=num_units,
                                      activation=activation_fun)
  elif unit_type == "layer_norm_lstm":
    single_cell = contrib.rnn.LayerNormLSTMCell(num_units=num_units,
                                                forget_bias=forget_bias,
                                                layer_norm=True)
  elif unit_type == "nas":
    single_cell = contrib.rnn.NASCell(num_units=num_units)
  else:
    raise ValueError("Unknow units type %s" % unit_type)

  # Dropout
  if droprate > 0.0:
    single_cell = contrib.rnn.DropoutWrapper(
      cell=single_cell,input_keep_prob=(1.0-droprate))

  # Residual
  if residual_connection:
    single_cell = contrib.rnn.ResidualWrapper(
      single_cell, residual_fn=residual_fn)

  # Device
  if device_str:
    single_cell = contrib.rnn.DeviceWrapper(single_cell, device_str)

  return single_cell

def multiRNNCell(unit_type, num_units, num_layers, layer_start_residual,
                 forget_bias, droprate, mode, num_gpus):
  cell_list = []
  for i in range(1,num_layers+1):
    single_cell = _single_rnn_cell(unit_type,num_units,forget_bias,droprate,mode,
                                   residual_connection=(i>=layer_start_residual),
                                   device_str="/gpu:%d" % ((i-1)%num_gpus))
    cell_list.append(single_cell)
  if len(cell_list)==1:
    return cell_list[0]
  else:
    return contrib.rnn.MultiRNNCell(cell_list)




