import os
import ocnn
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from solver import Solver, Dataset, AverageTracker, parse_args, get_config
from datasets import get_scannet_dataset

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_current_consistency_weight(epoch, consistency, rampup):
  if rampup == 0:
    rampup =  1.0
  else:
    current = np.clip(epoch, 0.0, rampup)
    phase = 1.0 - current / rampup
    rampup = float(np.exp(-5.0 * phase * phase))
  return consistency * rampup

def loss_function(logit, label):
  criterion = torch.nn.CrossEntropyLoss()
  loss = criterion(logit, label.long())
  return loss

def loss_function_consist(logit, logit_stdnt):
  input_soft = F.softmax(logit)
  target_soft = F.softmax(logit_stdnt)
  loss_u = F.mse_loss(input_soft, target_soft, size_average=True)
  return loss_u

def loss_function_mini_enpty(logit_u):
  loss_u = - torch.sum(F.softmax(logit_u) * F.log_softmax(logit_u), 1)
  loss_u = 0.5 * torch.mean(loss_u)
  return loss_u

def accuracy(logit, label):
  pred = logit.argmax(dim=1)
  accu = pred.eq(label).float().mean()
  return accu

def IoU_per_shape(logit, label, class_num):
  pred = logit.argmax(dim=1)

  IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
  intsc, union = [None] * class_num, [None] * class_num
  for k in range(class_num):
    pk, lk = pred.eq(k), label.eq(k)
    intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
    union[k] = torch.sum(torch.logical_or(pk, lk).float())

    valid = torch.sum(lk.any()) > 0
    valid_part_num += valid.item()
    IoU += valid * intsc[k] / (union[k] + esp)

  # Calculate the shape IoU for ShapeNet
  IoU /= valid_part_num + esp
  return IoU, intsc, union


class SegSolver(Solver):
  def get_model(self, flags):
    if flags.name.lower() == 'segnet':
      model = ocnn.SegNet(flags.depth, flags.channel, flags.nout, flags.interp)
    elif flags.name.lower() == 'unet':
      model = ocnn.UNet(flags.depth, flags.channel, flags.nout, flags.nempty,
                        flags.interp, flags.use_checkpoint)
    else:
      raise ValueError
    
    if 'pretrain_file' in flags:
      print('Load Pretrained pth file: ' + flags.pretrain_file)
      model_dict = torch.load(flags.pretrain_file)
      for k, v in model_dict.items() : 
        if 'header' not in k :
          param = model_dict[k].data
          model.state_dict()[k].copy_(param)
    return model
  
  def get_pseudo_model(self, flags):
    if flags.name.lower() == 'segnet':
      model = ocnn.SegNet(flags.depth, flags.channel, flags.nout, flags.interp)
    elif flags.name.lower() == 'unet':
      model = ocnn.UNet(flags.depth, flags.channel, flags.nout, flags.nempty,
                        flags.interp, flags.use_checkpoint)
    else:
      raise ValueError
    
    print('Load Pretrained pth file: ' + flags.pseudo_file)
    model_dict = torch.load(flags.pseudo_file)
    for k, v in model_dict.items() : 
        param = model_dict[k].data
        model.state_dict()[k].copy_(param)
    return model

  def config_model(self):
    model = self.get_model(self.FLAGS.MODEL)
    model.cuda(device=self.device)

    if self.world_size > 1:
      if self.FLAGS.MODEL.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
      model = torch.nn.parallel.DistributedDataParallel(
          module=model, device_ids=[self.device],
          output_device=self.device, broadcast_buffers=False,
          find_unused_parameters=False)
    # if self.is_master:
    #   print(model)
    self.model = model

    if self.FLAGS.DATA.train.semi:
      model_pred = self.get_model(self.FLAGS.MODEL)
      model_pseudo = self.get_pseudo_model(self.FLAGS.MODEL)
      model_pred.cuda(device=self.device)
      model_pseudo.cuda(device=self.device)
      if self.world_size > 1:
        if self.FLAGS.MODEL.sync_bn:
          model_pred = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_pred)
          model_pseudo = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_pred)
        model_pred = torch.nn.parallel.DistributedDataParallel(
            module=model_pred, device_ids=[self.device],
            output_device=self.device, broadcast_buffers=False,
            find_unused_parameters=False)
        model_pseudo = torch.nn.parallel.DistributedDataParallel(
            module=model_pseudo, device_ids=[self.device],
            output_device=self.device, broadcast_buffers=False,
            find_unused_parameters=False)
      self.model_pred = model_pred
      self.model_pseudo = model_pseudo
  
  def save_checkpoint(self, epoch):
    if not self.is_master:
      return

    # clean up
    ckpts = sorted(os.listdir(self.ckpt_dir))
    ckpts = [ck for ck in ckpts if ck.endswith('.pth') or ck.endswith('.tar')]
    if len(ckpts) > self.FLAGS.SOLVER.ckpt_num:
      for ckpt in ckpts[:-self.FLAGS.SOLVER.ckpt_num]:
        os.remove(os.path.join(self.ckpt_dir, ckpt))

    # save ckpt
    model_dict = self.model.module.state_dict() \
                 if self.world_size > 1 else self.model.state_dict()
    ckpt_name = os.path.join(self.ckpt_dir, '%05d' % epoch)
    torch.save(model_dict, ckpt_name + '.model.pth')

    if self.FLAGS.DATA.train.semi:
      model_pred_dict = self.model_pred.module.state_dict() \
                 if self.world_size > 1 else self.model_pred.state_dict()
      torch.save({'model_dict': model_dict, 'model_pred_dict': model_pred_dict, 'epoch': epoch, 'global_step': self.global_step,
                'optimizer_dict': self.optimizer.state_dict(),
                'scheduler_dict': self.scheduler.state_dict()},
               ckpt_name + '.solver.tar')
    else:
      torch.save({'model_dict': model_dict, 'epoch': epoch, 'global_step': self.global_step,
                'optimizer_dict': self.optimizer.state_dict(),
                'scheduler_dict': self.scheduler.state_dict()},
               ckpt_name + '.solver.tar')
  
  def load_checkpoint(self):
    ckpt = self.FLAGS.SOLVER.ckpt
    if not ckpt:
      # If ckpt is empty, then get the latest checkpoint from ckpt_dir
      if not os.path.exists(self.ckpt_dir):  return
      ckpts = sorted(os.listdir(self.ckpt_dir))
      ckpts = [ck for ck in ckpts if ck.endswith('solver.tar')]
      if len(ckpts) > 0:
        ckpt = os.path.join(self.ckpt_dir, ckpts[-1])
    if not ckpt:  return  # return if ckpt is still empty

    # load trained model
    # check: map_location = {'cuda:0' : 'cuda:%d' % self.rank}
    trained_dict = torch.load(ckpt, map_location='cuda')
    if ckpt.endswith('.solver.tar'):
      model_dict = trained_dict['model_dict']
      self.start_epoch = trained_dict['epoch'] + 1  # !!! add 1
      self.global_step = trained_dict['global_step']
      if self.optimizer:
        self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
      if self.scheduler:
        self.scheduler.load_state_dict(trained_dict['scheduler_dict'])
    else:
      model_dict = trained_dict
    model = self.model.module if self.world_size > 1 else self.model
    model.load_state_dict(model_dict)
    
    if self.FLAGS.DATA.train.semi:
      model_pred_dict = trained_dict['model_pred_dict']
      model_pred = self.model_pred.module if self.world_size > 1 else self.model_pred
      model_pred.load_state_dict(model_pred_dict) 
      for param in model_pred.parameters():
          param.detach_()
    # print messages
    if self.is_master:
      tqdm.write('Load the checkpoint: %s' % ckpt)
      tqdm.write('The start_epoch is %d' % self.start_epoch)

  def get_dataset(self, flags):
    if 'limited_reconst' in flags:
      with open(flags.limited_reconst) as f:
        self.LR = [name.split('\n')[0] for name in f.readlines()]
      if not flags.semi:
        flags.defrost()
        flags.filelist = flags.limited_reconst

    if flags.name.lower() == 'scannet':
      return get_scannet_dataset(flags)
    else:
      transform = ocnn.TransformCompose(flags)
      dataset = Dataset(flags.location, flags.filelist, transform, 
                        in_memory=flags.in_memory)
      return dataset, ocnn.collate_octrees

  def parse_batch(self, batch):
    octree = batch['octree'].cuda()
    if self.model.training:
      points = batch['points']
      pts = ocnn.points_batch_property(points, 'xyzi').cuda()
      label = ocnn.points_batch_property(points, 'label').squeeze().cuda()
    
    else:
      points = batch['points']
      pts = ocnn.points_batch_property(points, 'xyzi').cuda()
      label = ocnn.points_batch_property(points, 'label').squeeze().cuda()
    
    return octree, pts, label

  def model_forward(self, batch):
    octree, pts, label = self.parse_batch(batch)
    logit = self.model(octree, pts)

    # Limited Annotation 
    if self.model.training:
      if self.FLAGS.DATA.train.semi:
        octree_stdnt, pts_stdnt = torch.autograd.Variable(octree, volatile=True), torch.autograd.Variable(pts, volatile=True)
        logit_stdnt = self.model_pred(octree_stdnt, pts_stdnt)
        with torch.no_grad():
          logit_pseudo = self.model_pseudo(octree_stdnt, pts_stdnt)
      pred_l, targ = [], []
      pred_u, targ_lu, pseu, pseu_lu = [], [], [], [] 
      for i in range(self.FLAGS.DATA.train.batch_size):
        ## labeled data
        if batch['filename'][i] in self.LR:
          pred_l.append(logit[pts[:, 3]==i])
          targ.append(label[pts[:, 3]==i])
        
        ## Unlabeled data
        else:
          genre, ind, count = F.softmax(logit_pseudo, 1)[pts[:, 3]==i].argmax(1).unique(return_inverse=True, return_counts=True)
          # Calculate the confidence of pseudo labels in each class
          for g, c in zip(genre, count):
            topoints = int(c*self.FLAGS.LOSS.pseudo_rate)
            softmax_logit = F.softmax(logit_pseudo, 1)
            pts_mask = softmax_logit[pts[:, 3]==i].argmax(1) == g
            pts_logit = logit[pts[:, 3]==i][pts_mask]
            pts_logit_pseudo = logit_pseudo[pts[:, 3]==i][pts_mask]
            pts_logit_stdnt = logit_stdnt[pts[:, 3]==i][pts_mask]
            pts_label = label[pts[:, 3]==i][pts_mask]
            class_logit = softmax_logit[pts[:, 3]==i][pts_mask]
            topk = torch.topk(class_logit.amax(1), topoints)
            map = torch.zeros(class_logit.shape[0])
            map[topk.indices] = 1
            mask_l = (map==1).squeeze().bool()
            mask_u = (map==0).squeeze().bool()
            
            if mask_l.sum() : pred_l.append(pts_logit[mask_l])
            if mask_l.sum() : targ.append(pts_logit_pseudo.argmax(1)[mask_l])
            if topoints : pred_u.append(pts_logit[mask_u])
            if topoints : pseu.append(pts_logit_stdnt[mask_u])
    
      logit_l = torch.cat(pred_l, 0) if pred_l else torch.Tensor([0]).cuda()
      logit_u = torch.cat(pred_u, 0) if pred_u else torch.Tensor([0]).cuda()
      logit_u_stdnt = torch.cat(pseu, 0) if pred_u else torch.Tensor([0]).cuda()
      label = torch.cat(targ, 0) if pred_l else torch.Tensor([0]).cuda()
      label_mask = label > self.FLAGS.LOSS.mask  # filter labels

      if self.FLAGS.DATA.train.semi:
        return logit_l[label_mask], logit_u, logit_u_stdnt, label[label_mask]
      else:
        return logit_l[label_mask], label[label_mask]

    else:
      label_mask = label > self.FLAGS.LOSS.mask  # filter labels
      return logit[label_mask], label[label_mask]

  def train(self):
    self.global_step = 0 
    self.config_model()
    self.config_dataloader()
    self.configure_optimizer()
    self.configure_log()
    self.load_checkpoint()

    rng = range(self.start_epoch, self.FLAGS.SOLVER.max_epoch+1)
    for epoch in tqdm(rng, ncols=80, disable=self.disable_tqdm):
      # training epoch
      self.train_epoch(epoch)

      # update learning rate
      self.scheduler.step()
      if self.is_master:
        lr = self.scheduler.get_last_lr()  # lr is a list
        self.summry_writer.add_scalar('train/lr', lr[0], epoch)

      # testing or not
      if epoch % self.FLAGS.SOLVER.test_every_epoch != 0:
        continue

      # testing epoch
      self.test_epoch(epoch)

      # checkpoint
      self.save_checkpoint(epoch)

    # sync and exit
    if self.world_size > 1:
      torch.distributed.barrier()

  def train_epoch(self, epoch):
    self.model.train()
    if self.FLAGS.DATA.train.semi:
      self.model_pred.train()
      self.model_pseudo.eval()
    if self.world_size > 1:
      self.train_loader.sampler.set_epoch(epoch)

    train_tracker = AverageTracker()
    rng = range(len(self.train_loader))
    for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
      self.optimizer.zero_grad()

      # forward
      batch = self.train_iter.next()
      batch['iter_num'] = it
      batch['epoch'] = epoch
      output = self.train_step(batch)

      # backward
      output['train/loss'].backward()
      self.optimizer.step()

      # track the averaged tensors
      train_tracker.update(output)
      if self.FLAGS.DATA.train.semi:
        update_ema_variables(self.model, self.model_pred, self.FLAGS.LOSS.alpha, self.global_step)

    # save logs
    if self.world_size > 1:
      train_tracker.average_all_gather()
    if self.is_master:
      train_tracker.log(epoch, self.summry_writer)

  def train_step(self, batch):
    if self.FLAGS.DATA.train.semi:
      logit, unlbl_logit, unlbl_logit_stdnt, label = self.model_forward(batch)
      unlbl_logit_stdnt = torch.autograd.Variable(unlbl_logit_stdnt.detach().data, requires_grad=False)
      loss = {'label': [], 'unlabel_consist': [], 'unlabel_mini_entpy': []}
      loss_total = 0 
      
      loss['label'] = 1 * loss_function(logit, label) if logit.shape[0] > 1  else 0 
      consist_weight = get_current_consistency_weight(batch['epoch'], self.FLAGS.LOSS.consistency, self.FLAGS.LOSS.consist_rampup)
      loss['unlabel_consist'] = consist_weight * loss_function_consist(unlbl_logit, unlbl_logit_stdnt) if unlbl_logit.shape[0] > 1  else 0 
      loss['unlabel_mini_entpy'] = self.FLAGS.LOSS.weight_sum * loss_function_mini_enpty(unlbl_logit) if unlbl_logit.shape[0] > 1  else 0 
      self.global_step += 1

      for k, v in loss.items():
        loss_total += v

    else:
      logit, label = self.model_forward(batch)
      loss_total = loss_function(logit, label)

    return {'train/loss': loss_total}

  def test_step(self, batch):
    logit, label = self.model_forward(batch)
    if logit.shape[0] == 0:
      return None  # degenerated case
    loss = loss_function(logit, label)
    accu = accuracy(logit, label)
    num_class = self.FLAGS.LOSS.num_class
    IoU, insc, union = IoU_per_shape(logit, label, num_class)

    names = ['test/loss', 'test/accu', 'test/mIoU'] + \
            ['test/intsc_%d' % i for i in range(num_class)] + \
            ['test/union_%d' % i for i in range(num_class)]
    tensors = [loss, accu, IoU] + insc + union
    return dict(zip(names, tensors))

  def eval_step(self, batch):
    octree = batch['octree'].cuda()
    pts = ocnn.points_batch_property(batch['points'], 'xyzi').cuda()
    logit = self.model(octree, pts)
    prob = torch.nn.functional.softmax(logit, dim=1)
    label = prob.argmax(dim=1)

    assert len(batch['inbox_mask']) == 1, 'The batch_size must be 1'
    filename = '%02d.%04d.npz' % (batch['epoch'], batch['iter_num'])
    np.savez(os.path.join(self.logdir, filename), 
             prob=prob.cpu().numpy(),
             label=label.cpu().numpy(),
             inbox_mask=batch['inbox_mask'][0].numpy().astype(bool))

  def result_callback(self, avg_tracker, epoch):
    ''' Calculate the part mIoU for PartNet and ScanNet'''
    avg = avg_tracker.average()

    iou_part = 0.0
    # Labels smaller than mask is ignored. The points with the label 0 in
    # PartNet are background points, i.e., unlabeled points
    mask = self.FLAGS.LOSS.mask + 1
    num_class = self.FLAGS.LOSS.num_class
    for i in range(mask, num_class):
      instc_i = avg['test/intsc_%d' % i]
      union_i = avg['test/union_%d' % i]
      iou_part += instc_i / (union_i + 1.0e-10)
    iou_part = iou_part / (num_class - mask)
    if self.summry_writer:
      self.summry_writer.add_scalar('test/mIoU_part', iou_part, epoch)
    else:
      print('Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))


def main(TheSolver):
  get_config().LOSS.mask = -1           # mask the invalid labels
  get_config().LOSS.point_wise = False  # point-wise loss or voxel-wise loss

  FLAGS = parse_args()
  Solver.main(FLAGS, TheSolver)


if __name__ == "__main__":
  main(SegSolver)