
from FLAGS import PARAM
from tensorflow.python.ops import lookup_ops
import os

UNK = PARAM.unk
SOS = PARAM.sos
EOS = PARAM.eos
UNK_ID = 0

# char ids 0-255 come from utf-8 encoding bytes
# assign 256-300 to special chars
BOS_CHAR_ID = 256  # <begin sentence>
EOS_CHAR_ID = 257  # <end sentence>
BOW_CHAR_ID = 258  # <begin word>
EOW_CHAR_ID = 259  # <end word>
PAD_CHAR_ID = 260  # <padding>

DEFAULT_CHAR_MAXLEN = 50  # max number of chars for each word.


def create_vocal_tables():
  src_vocab_file = os.path.join(PARAM.vocab_prefix,PARAM.src)
  tgt_vocab_file = os.path.join(PARAM.vocab_prefix,PARAM.tgt)
  src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file,default_value=UNK_ID)
  if PARAM.share_vocab:
    tgt_vocab_table = src_vocab_table
  else:
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file,default_value=UNK_ID)
  return src_vocab_table, tgt_vocab_table

