"""
@module loss_sparse_dist

Implements sparse distribution loss introduced with TreeLSTM(Kai, 2016).
"""

import torch
import torch.nn as nn

from . import register_loss_fn

@register_loss_fn('sparse-dist')
def loss_sparse_dist(model, inputs_a, inputs_b, inputs_pairinfo, **kwargs):
  """
  Sparse distribution loss implemented.

  @param model STS Model designed for spase distribution. Must have `r_size` attribute.
  @param inputs_a List of dictionaries. Refer to 'dataset.py' for more details.
  @param inputs_b List of dictionaries(. Refer to 'dataset.py' for more details.
  @param
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs_a)

  scores = model.run(inputs_a, inputs_b)

  r_size = model.r_size

  # Generate sparse distribution for input batch
  mask = torch.zeros(batch_size, r_size, device=device).detach()
  sparse = torch.zeros(batch_size, r_size, device=device).detach()
  # FIXME: assertion that score is 0~5 range
  MAX_SCORE = 5
  for i, score in enumerate(inputs_pairinfo):
    if score == MAX_SCORE:
      mask[i, -1] = 1
      sparse[i, -1] = 1
    else:
      floor = int(score / MAX_SCORE * (r_size-1)) 
      ceil = floor + 1
      mask[i, floor] = 1
      mask[i, ceil] = 1
      sparse[i, ceil] = score - floor * MAX_SCORE / (r_size-1)
      sparse[i, floor] = 1 - sparse[i, ceil]
  
  # Calculate nll loss for sparce distribution
  loss = torch.sum(torch.abs(scores * mask - sparse)) / batch_size