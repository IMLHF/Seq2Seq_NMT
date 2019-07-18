
import codecs
import numpy as np
import os
from tensorflow.python.ops import lookup_ops
import tensorflow as tf

from ..FLAGS import PARAM
from ..utils import misc_utils

PAD = PARAM.pad
UNK = PARAM.unk
SOS = PARAM.sos
EOS = PARAM.eos
UNK_ID = 1

# char ids 0-255 come from utf-8 encoding bytes
# assign 256-300 to special chars
BOS_CHAR_ID = 256  # <begin sentence>
EOS_CHAR_ID = 257  # <end sentence>
BOW_CHAR_ID = 258  # <begin word>
EOW_CHAR_ID = 259  # <end word>
PAD_CHAR_ID = 260  # <padding>

DEFAULT_CHAR_MAXLEN = 50  # max number of chars for each word.

# __all__ = ['create_vocab_tables', 'new_or_pretrain_embed', 'tokens_to_bytes']

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
      pad = PAD
      unk = UNK
      sos = SOS
      eos = EOS
      assert len(vocab) >= 4
      if vocab[0] != pad or vocab[1] != unk or vocab[2] != sos or vocab[3] != eos:
        misc_utils.printinfo("The first 4 vocab words [%s, %s, %s, %s]"
                             " are not [%s, %s, %s, %s]" %
                             (vocab[0], vocab[1], vocab[2], vocab[3], pad, unk, sos, eos),
                             log_file)
        if pad in vocab:
          vocab.pop(vocab.index(pad))
          vocab_size -=1
        if unk in vocab:
          vocab.pop(vocab.index(unk))
          vocab_size -=1
        if sos in vocab:
          vocab.pop(vocab.index(sos))
          vocab_size -=1
        if eos in vocab:
          vocab.pop(vocab.index(eos))
          vocab_size -=1
        vocab = [pad, unk, sos, eos] + vocab
        vocab_size += 4
        os.rename(vocab_file, vocab_file+'.bak')
        with codecs.getwriter("utf-8")(
                tf.gfile.GFile(vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  vocab_size = len(vocab)
  return vocab_size

def create_vocab_word2id_tables(log_file):
  '''
  Returns:
    src_vocab_table : word -> id
    tgt_vocab_table : word -> id
    src_vocab_size:
    tgt_vocab_size:
  '''
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

def _create_pretrained_emb_from_txt(
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
                          vocab_size, embed_size, _dtype):
  """Create a new or load an existing embedding matrix."""
  if vocab_file and embed_file:
    embedding = _create_pretrained_emb_from_txt(log_file, vocab_file, embed_file)
  else:
    with tf.device(PARAM.embedding_on_device):
      embedding = tf.get_variable(
          name=embed_name, shape=[vocab_size, embed_size], dtype=_dtype)
  return embedding

def _string_to_bytes(text, max_length):
  """Given string and length, convert to byte seq of at most max_length.

  This process mimics docqa/elmo's preprocessing:
  https://github.com/allenai/document-qa/blob/master/docqa/elmo/data.py

  Note that we make use of BOS_CHAR_ID and EOS_CHAR_ID in iterator_utils.py &
  our usage differs from docqa/elmo.

  Args:
    text: tf.string tensor of shape []
    max_length: max number of chars for each word.

  Returns:
    A tf.int32 tensor of the byte encoded text.
  """
  byte_ids = tf.to_int32(tf.decode_raw(text, tf.uint8))
  byte_ids = byte_ids[:max_length - 2]
  padding = tf.fill([max_length - tf.shape(byte_ids)[0] - 2], PAD_CHAR_ID)
  byte_ids = tf.concat(
      [[BOW_CHAR_ID], byte_ids, [EOW_CHAR_ID], padding], axis=0)
  tf.logging.info(byte_ids)

  byte_ids = tf.reshape(byte_ids, [max_length])
  tf.logging.info(byte_ids.get_shape().as_list())
  return byte_ids + 1


def tokens_to_bytes(tokens):
  """Given a sequence of strings, map to sequence of bytes.

  Args:
    tokens: A tf.string tensor

  Returns:
    A tensor of shape words.shape + [bytes_per_word] containing byte versions
    of each word.
  """
  bytes_per_word = DEFAULT_CHAR_MAXLEN
  with tf.device("/cpu:0"):
    tf.assert_rank(tokens, 1)
    shape = tf.shape(tokens)
    tf.logging.info(tokens)
    tokens_flat = tf.reshape(tokens, [-1])
    as_bytes_flat = tf.map_fn(
        fn=lambda x: _string_to_bytes(x, max_length=bytes_per_word),
        elems=tokens_flat,
        dtype=tf.int32,
        back_prop=False)
    tf.logging.info(as_bytes_flat)
    as_bytes = tf.reshape(as_bytes_flat, [shape[0], bytes_per_word])
  return as_bytes
