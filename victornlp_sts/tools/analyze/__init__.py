sentiment_analysis_fn = {}
def register_analysis_fn(name):
  def decorator(fn):
    sentiment_analysis_fn[name] = fn
    return fn
  return decorator

from .accuracy import *
