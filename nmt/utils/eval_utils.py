import tensorflow as tf
import time
import math
from ..models import model_builder

def _safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def calc_ppl(total_loss, total_predict_count): # ppl at github:tensorflow/nmt, not used.
  perplexity = _safe_exp(total_loss / total_predict_count)
  return perplexity

