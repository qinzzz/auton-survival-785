# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Utility functions to train the Deep Survival Machines models"""

from .dsm_torch import DeepSurvivalMachinesTorch
from .losses import unconditional_loss, conditional_loss

from sklearn.utils import shuffle

from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np

import gc
import logging


def get_optimizer(model, lr):

  if model.optimizer == 'Adam':
    return torch.optim.Adam(model.parameters(), lr=lr)
  elif model.optimizer == 'SGD':
    return torch.optim.SGD(model.parameters(), lr=lr)
  elif model.optimizer == 'RMSProp':
    return torch.optim.RMSprop(model.parameters(), lr=lr)
  else:
    raise NotImplementedError('Optimizer '+model.optimizer+
                              ' is not implemented')

def pretrain_dsm(model, t_train, e_train, t_valid, e_valid,
                 n_iter=10000, lr=1e-2, thres=1e-4):

  premodel = DeepSurvivalMachinesTorch(1, 1,
                                       dist=model.dist,
                                       risks=model.risks,
                                       optimizer=model.optimizer)
  premodel.double()

  optimizer = get_optimizer(premodel, lr)

  oldcost = float('inf')
  patience = 0
  costs = []
  for _ in tqdm(range(n_iter)):

    optimizer.zero_grad()
    loss = 0
    for r in range(model.risks):
      loss += unconditional_loss(premodel, t_train, e_train, str(r+1))
    loss.backward()
    optimizer.step()

    valid_loss = 0
    for r in range(model.risks):
      valid_loss += unconditional_loss(premodel, t_valid, e_valid, str(r+1))
    valid_loss = valid_loss.detach().cpu().numpy()
    costs.append(valid_loss)
    #print(valid_loss)
    if np.abs(costs[-1] - oldcost) < thres:
      patience += 1
      if patience == 3:
        break
    oldcost = costs[-1]

  return premodel

def _reshape_tensor_with_nans(data):
  """Helper function to unroll padded RNN inputs."""
  data = data.reshape(-1)
  return data[~torch.isnan(data)]

def _get_padded_features(x):
  """Helper function to pad variable length RNN inputs with nans."""
  d = max([len(x_) for x_ in x])
  padx = []
  for i in range(len(x)):
    pads = np.nan*np.ones((d - len(x[i]),) + x[i].shape[1:])
    padx.append(np.concatenate([x[i], pads]))
  return np.array(padx)

def _get_padded_targets(t):
  """Helper function to pad variable length RNN inputs with nans."""
  d = max([len(t_) for t_ in t])
  padt = []
  for i in range(len(t)):
    pads = np.nan*np.ones(d - len(t[i]))
    padt.append(np.concatenate([t[i], pads]))
  return np.array(padt)[:, :, np.newaxis]

def train_dsm(model,
              x_train, t_train, e_train, x_train_normalized,
              x_valid, t_valid, e_valid, x_valid_normalized,
              n_iter=10000, lr=1e-3, elbo=True,
              bs=100, random_seed=0):
  """Function to train the torch instance of the model."""

  # print("x_train: ", x_train[:3])
  # print("x_val: ", x_valid[:3])
  # print("x_train_normalized: ", x_train_normalized[:3])
  # print("x_val_normalized: ", x_valid_normalized[:3])


  torch.manual_seed(random_seed)
  np.random.seed(random_seed)

  logging.info('Pretraining the Underlying Distributions...')
  # For padded variable length sequences we first unroll the input and
  # mask out the padded nans.
  t_train_ = _reshape_tensor_with_nans(t_train)
  e_train_ = _reshape_tensor_with_nans(e_train)
  t_valid_ = _reshape_tensor_with_nans(t_valid)
  e_valid_ = _reshape_tensor_with_nans(e_valid)

  premodel = pretrain_dsm(model,
                          t_train_,
                          e_train_,
                          t_valid_,
                          e_valid_,
                          n_iter=10000,
                          lr=1e-2,
                          thres=1e-4)

  for r in range(model.risks):
    model.shape[str(r+1)].data.fill_(float(premodel.shape[str(r+1)]))
    model.scale[str(r+1)].data.fill_(float(premodel.scale[str(r+1)]))

  model.double()
  optimizer = get_optimizer(model, lr)

  patience = 0
  oldcost = float('inf')

  nbatches = int(x_train.shape[0]/bs)+1

  dics = []
  costs = []

  # pretrain vae
  print("Start pretraining VAE ...")
  for r in range(model.risks):
    model.vade[str(r+1)].vae_pretrain(x_train, nbatches, 50)

  print("Start training DSM + VAE ...")
  i = 0
  for i in tqdm(range(n_iter)):
    epoch_train_dsm_loss = 0
    epoch_train_vae_loss = 0

    x_train, t_train, e_train, x_train_normalized = shuffle(x_train, t_train, e_train, x_train_normalized, random_state=i)

    for j in range(nbatches):

      xb = x_train[j*bs:(j+1)*bs]
      tb = t_train[j*bs:(j+1)*bs]
      eb = e_train[j*bs:(j+1)*bs]
      xb_nm = x_train_normalized[j*bs:(j+1)*bs]

      if xb.shape[0] == 0:
        continue

      optimizer.zero_grad()
      dsm_loss = 0
      vae_loss = 0
      for r in range(model.risks):
        shape, scale, logits, z_mu, z_sigma2_log = model.forward(xb, xb_nm, str(r+1))
        dsm_loss += conditional_loss(model.dist,
                                 model.discount,
                                 shape,
                                 scale,
                                 logits,
                                 model.k,
                                 _reshape_tensor_with_nans(tb),
                                 _reshape_tensor_with_nans(eb),
                                 elbo=elbo,
                                 risk=str(r+1))

        # vae_loss += loss_function(model.vae[str(r+1)].vae_decode(z), xb_nm, qy, model.k)
        vae_loss += 0.1 * model.vade[str(r+1)].vae_loss(xb_nm, z_mu, z_sigma2_log)

      # print("train_dsm_loss: ", dsm_loss.item())
      # print("train_vae_loss: ", vae_loss.item())

      loss = dsm_loss + vae_loss
      loss.backward()
      optimizer.step()

      epoch_train_dsm_loss += dsm_loss
      epoch_train_vae_loss += vae_loss

    # print("Epoch {}: Train DSM Loss: {:.04f}, Train VAE Loss {:.04f}".format(i, epoch_train_dsm_loss.item()/nbatches, epoch_train_vae_loss.item()/nbatches))
    

    valid_dsm_loss = 0
    valid_vae_loss = 0
    for r in range(model.risks):
      shape, scale, logits, z_mu, z_sigma2_log = model.forward(x_valid, x_valid_normalized, str(r+1))
      valid_dsm_loss += conditional_loss(model.dist,
                                     model.discount,
                                     shape,
                                     scale,
                                     logits,
                                     model.k,
                                     t_valid_,
                                     e_valid_,
                                     elbo=False,
                                     risk=str(r+1))
      # valid_vae_loss += loss_function(model.vae[str(r+1)].vae_decode(z), x_valid_normalized, qy, model.k)
      valid_vae_loss += 0.1 * model.vade[str(r+1)].vae_loss(x_valid_normalized, z_mu, z_sigma2_log)


    valid_dsm_loss = valid_dsm_loss.detach().cpu().numpy()
    valid_vae_loss = valid_vae_loss.detach().cpu().numpy()

    # print("Epoch {}: Valid DSM Loss: {:.04f}, Valid VAE Loss {:.04f}".format(i, valid_dsm_loss, valid_vae_loss))



    # print("valid_dsm_loss: ", valid_dsm_loss.item())
    # print("valid_vae_loss: ", valid_vae_loss.item())

    costs.append(float(valid_dsm_loss))
    dics.append(deepcopy(model.state_dict()))

    print("valid loss", valid_loss)

    if costs[-1] >= oldcost:
      if patience == 2:
        minm = np.argmin(costs)
        model.load_state_dict(dics[minm])

        del dics
        gc.collect()

        return model, i
      else:
        patience += 1
    else:
      patience = 0

    oldcost = costs[-1]

  minm = np.argmin(costs)
  model.load_state_dict(dics[minm])

  del dics
  gc.collect()

  return model, i