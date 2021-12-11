from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import numpy as np
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
      output, loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def tv_loss(self, img, alpha=2):
    """
    Compute total variation loss.

    Args:
    -- img: PyTorch Variable of shape (N, 3, H, W) holding an input image.
    -- alpha: alpha norm.

    Returns:
    -- loss: PyTorch Variable holding a scalar giving the total variation loss for img.
    """
    N = img.shape[0]
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], alpha))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], alpha))
    loss = (h_variance + w_variance) / N
    return loss
  
  def calculate_clipping(self, scale=1/255):
    """
    Helper function for calculating lo and hi.

    Args:
    -- cfg: configuration file for model
    -- scale: scale for lo, hi. If lo, hi in [0,255], scale is 1. If lo, hi in [0,1], scale is 1/255

    Returns:
    -- LO, HI: list, lower bound and upper bound of each channel
    """
    LO = []
    HI = []
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    for c in range(3):
        lo = float(-mean[c] / std[c])*scale
        hi = float((255.0 - mean[c]) / std[c])*scale
        LO.append(lo)
        HI.append(hi)
    return LO, HI

  def clipping(self, X, lo, hi):
    """
    Helper function for clipping.

    Args:
    -- X: Pytorch tensor of shape (N, C, H, W)
    -- lo: list, lower bound for each channel
    -- hi: list, upper bound for each channel

    Returns:
    -- X: Pytorch tensor of shape (N, C, H, W)
    """
    for c in range(3):
        X.data[:, c].clamp_(min=lo[c], max=hi[c])
    return X

  def image_init(self, images):
    lo, hi = self.calculate_clipping()
    # initialize input tensor
    with torch.no_grad():
        x = torch.randn(images.shape).to(images.device)
        x = x * 0.2
        x = self.clipping(x, lo, hi)
    return x
  
  def single_batch(self, batch, device):
    model_with_loss = self.model_with_loss
    model_with_loss.eval()
    model_with_loss.model.eval()
    model_with_loss = model_with_loss.to(device)
    for k in batch:
      if k != 'meta':
        batch[k] = batch[k].to(device=device)
    output, loss, loss_stats = model_with_loss(batch)

    print(output)

  def blur_image(self, X, sigma=1):
    """
    Helper function to blur an image.

    Args:
    -- X: Pytorch tensor of shape (N, C, H, W).
    -- sigma: sigma of gaussian blur.

    Returns
    -- A new pytorch tensor of shape (N, C, H, W).
    """
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

  def invert(self, batch, device, iters=1000):
    lo, hi = self.calculate_clipping()
    model_with_loss = self.model_with_loss
    model_with_loss.eval()
    model_with_loss.model.eval()
    model_with_loss = model_with_loss.to(device)
    opt = self.opt
    print_every = 50
    print(device)
    for k in batch:
      if k != 'meta':
        batch[k] = batch[k].to(device=device)
    batch['input'] = self.image_init(batch['input'])
    init_image = deepcopy(batch['input'])
    batch['input'].requires_grad_(True)
    optimizer = torch.optim.SGD([batch['input']], lr=2, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)
    tv_alpha =  2
    lamb_tv = 1.0e-5

    blur_every =  5
    blur_start = 5.0
    blur_decay = 0.985
    blur_end = 0.5

    if_jitter = False
    jitter_every = 30
    jitter_x = 4
    jitter_y = 4
    sigma = blur_start
    for iter in range(iters):
      output, loss, loss_stats = model_with_loss(batch)
      # TODO: Add the TV loss here
      
      loss = loss.mean()
      loss += 3 * lamb_tv * self.tv_loss(batch['input'], tv_alpha)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()
      if iter%print_every == 0:
        print(f"iter = {iter}, loss={loss.item()}")
        print(loss)
        print(loss_stats)
        if torch.equal(init_image, batch['input']):
          print("Not changing")
      
      with torch.no_grad():
        batch['input'] = self.clipping(batch['input'], lo, hi)
        if iter % blur_every == 0:
            sigma = max(sigma * blur_decay, blur_end)
            self.blur_image(batch['input'], sigma=sigma)
      if iter % 20 == 0:
        img = batch['input'].squeeze().detach().cpu()
        img_disp = img.permute(1,2,0)
        #plt.imshow((img_disp+1)*0.5)
        img_path = "/home/paritosh/Desktop/SEM2/16824/code/vlr-project/animation_images/18/" + str(iter) + ".png"
        try:
          plt.imsave(img_path,((img_disp+1)*0.5).detach().cpu().numpy(), dpi=1200)
        except:
          print(iter)
        # if if_jitter is True:
        #     if i % jitter_every == 0:
        #         jitter(batch['input'], jitter_x, jitter_y)
  def invert_with_output(self, batch, device, iters=1000):
    lo, hi = self.calculate_clipping()
    model_with_loss = self.model_with_loss
    model_with_loss.eval()
    model_with_loss.model.eval()
    model_with_loss = model_with_loss.to(device)
    opt = self.opt
    print_every = 50
    print(device)
    for k in batch:
      if k != 'meta':
        batch[k] = batch[k].to(device=device)
    output, loss, loss_stats = model_with_loss(batch)
    #print(output.keys())
    for key in output:
      if key in batch:
        #batch[key] = output[key]
        print(key, batch[key].shape, output[key].shape)
      #batch[key] = output[key]
    assert(False)
    batch['input'] = self.image_init(batch['input'])

    init_image = deepcopy(batch['input'])
    batch['input'].requires_grad_(True)
    optimizer = torch.optim.SGD([batch['input']], lr=2, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)
    tv_alpha =  2
    lamb_tv = 1.0e-5

    blur_every =  5
    blur_start = 5.0
    blur_decay = 0.985
    blur_end = 0.5

    if_jitter = False
    jitter_every = 30
    jitter_x = 4
    jitter_y = 4
    sigma = blur_start
    for iter in range(iters):
      output, loss, loss_stats = model_with_loss(batch)
      # TODO: Add the TV loss here
      loss = loss.mean()
      loss += 3 * lamb_tv * self.tv_loss(batch['input'], tv_alpha)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()
      if iter%print_every == 0:
        print(f"iter = {iter}, loss={loss.item()}")
        print(loss)
        print(loss_stats)
        if torch.equal(init_image, batch['input']):
          print("Not changing")
      
      with torch.no_grad():
        batch['input'] = self.clipping(batch['input'], lo, hi)
        if iter % blur_every == 0:
            sigma = max(sigma * blur_decay, blur_end)
            self.blur_image(batch['input'], sigma=sigma)
        # if if_jitter is True:
        #     if i % jitter_every == 0:
        #         jitter(batch['input'], jitter_x, jitter_y)     
    

  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)

  def infer(self, batch, device, iters):
    return self.invert(batch, device, iters)
  
  def infer_with_model_out(self, batch, device, iters):
    return self.invert_with_output(batch, device, iters)