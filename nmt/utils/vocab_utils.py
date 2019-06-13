
from FLAGS import PARAM
from tensorflow.python.ops import lookup_ops
import os
import codecs
import tensorflow as tf
from utils import misc_utils
import numpy as np

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


def _load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size

def _check_vocab(log_file, vocab_file, check_special_token=True):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  if tf.gfile.Exists(vocab_file):
    misc_utils.printinfo("# Vocab file %s exists" % vocab_file, log_file)
    vocab, vocab_size = _load_vocab(vocab_file)
    if check_special_token:
      # Verify if the vocab starts with unk, sos, eos
      # If not, prepend those tokens & generate a new vocab file
      unk = UNK
      sos = SOS
      eos = EOS
      assert len(vocab) >= 3
      if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
        misc_utils.printinfo("The first 3 vocab words [%s, %s, %s]"
                             " are not [%s, %s, %s]" %
                             (vocab[0], vocab[1], vocab[2], unk, sos, eos),
                             log_file)
        vocab = [unk, sos, eos] + vocab
        vocab_size += 3
        os.rename(vocab_file, vocab_file+'.bak')
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  vocab_size = len(vocab)
  return vocab_size

def create_vocab_tables(log_file):
  src_vocab_file = "%s.%s" % (PARAM.vocab_prefix,PARAM.src)
  tgt_vocab_file = "%s.%s" % (PARAM.vocab_prefix,PARAM.tgt)
  src_vocab_size = _check_vocab(log_file, src_vocab_file)
  tgt_vocab_size = _check_vocab(log_file, tgt_vocab_file)
  src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file,default_value=UNK_ID)
  if PARAM.share_vocab:
    tgt_vocab_table = src_vocab_table
  else:
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file,default_value=UNK_ID)
  return src_vocab_table, tgt_vocab_table, src_vocab_size, tgt_vocab_size

def _load_embed_txt(log_file, embed_file):
  """Load embed_file into a python dictionary.

  Note: the embed_file should be a Glove/word2vec formatted txt file. Assuming
  Here is an exampe assuming embed_size=5:

  the -0.071549 0.093459 0.023738 -0.090339 0.056123
  to 0.57346 0.5417 -0.23477 -0.3624 0.4037
  and 0.20327 0.47348 0.050877 0.002103 0.060547

  For word2vec format, the first line will be: <num_words> <emb_size>.

  Args:
    embed_file: file path to the embedding file.
  Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
  """
  emb_dict = dict()
  emb_size = None

  is_first_line = True
  with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, "rb")) as f:
    for line in f:
      tokens = line.rstrip().split(" ")
      if is_first_line:
        is_first_line = False
        if len(tokens) == 2:  # header line
          emb_size = int(tokens[1])
          continue
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        if emb_size != len(vec):
          misc_utils.printinfo(
              "Ignoring %s since embeding size is inconsistent." % word,
              log_file)
          del emb_dict[word]
      else:
        emb_size = len(vec)
  return emb_dict, emb_size


def create_pretrained_emb_from_txt(
    log_file, vocab_file, embed_file, num_trainable_tokens=3, dtype=tf.float32,
    scope=None):
  """Load pretrain embeding from embed_file, and return an embedding matrix.

  Args:
    embed_file: Path to a Glove formated embedding txt file.
    num_trainable_tokens: Make the first n tokens in the vocab file as trainable
      variables. Default is 3, which is "<unk>", "<s>" and "</s>".
  """
  vocab, _ = _load_vocab(vocab_file)
  trainable_tokens = vocab[:num_trainable_tokens]

  misc_utils.printinfo("# Using pretrained embedding: %s." % embed_file, log_file)
  misc_utils.printinfo("  with trainable tokens: ", log_file)

  emb_dict, emb_size = _load_embed_txt(log_file, embed_file)
  for token in trainable_tokens:
    misc_utils.printinfo("    %s" % token, log_file)
    if token not in emb_dict:
      emb_dict[token] = [0.0] * emb_size

  emb_mat = np.array(
      [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
  emb_mat = tf.constant(emb_mat)
  emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
  with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
    with tf.device(PARAM.embedding_on_device):
      emb_mat_var = tf.get_variable(
          "emb_mat_var", [num_trainable_tokens, emb_size])
  return tf.concat([emb_mat_var, emb_mat_const], 0)

def new_or_pretrain_embed(log_file, embed_name, vocab_file, embed_file,
                          vocab_size, embed_size, dtype):
  """Create a new or load an existing embedding matrix."""
  if vocab_file and embed_file:
    embedding = create_pretrained_emb_from_txt(log_file, vocab_file, embed_file)
  else:
    with tf.device(PARAM.embedding_on_device):
      embedding = tf.get_variable(
          embed_name, [vocab_size, embed_size], dtype)
  return embedding