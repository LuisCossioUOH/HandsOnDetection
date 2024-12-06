from argparse import Namespace
import logging
import math

import random
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp

from tqdm import tqdm

import evaluation # import test.py to get mAP after each epoch

from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, LoadImagesAndLabels, letterbox
from utils.general import labels_to_class_weights, init_seeds, fitness, \
    check_file, check_img_size, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossAuxOTA
from utils.torch_utils import ModelEMA, intersect_dicts, torch_distributed_zero_first

logger = logging.getLogger(__name__)



class training_session:
    def __init__(self, hyp, opt, device):
        logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        save_dir, epochs, batch_size, weights = Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights

        # Directories
        wdir = save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.results_file = save_dir / 'results.txt'
        self.last = wdir / 'last.pt'
        self.best = wdir / 'best.pt'
        self.save_dir = save_dir

        self.wdir = wdir

        init_seeds(opt.seed)
        with open(opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

        if opt.update_dataset:
            data_dict['train'] = opt.data_train
            data_dict['val'] = opt.data_train
            data_dict['nc'] = opt.nc
            data_dict['names'] = opt.names_classes
        self.is_coco = data_dict['nc'] == 80

        # Logging- Doing this before checking the dataset. Might update data_dict
        self.loggers = {'wandb': None}  # loggers dict

        opt.hyp = hyp  # add hyperparameters
        weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

        # self.wandb_logger = wandb_logger
        self.epochs = opt.epochs
        self.batch_size = batch_size
        self.nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
        self.names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        self.data_dict = data_dict
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (
            len(self.names), self.nc, opt.data)  # check

        # Model
        pretrained = weights.endswith('.pt')
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            self.model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=self.nc, anchors=hyp.get('anchors')).to(
                device)  # create
            exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            logger.info(
                'Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), weights))  # report
        else:
            self.model = Model(opt.cfg, ch=3, nc=self.nc, anchors=hyp.get('anchors')).to(device)  # create

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        # Optimizer
        logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):
                    pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imc'):
                if hasattr(v.imc, 'implicit'):
                    pg0.append(v.imc.implicit)
                else:
                    for iv in v.imc:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imb'):
                if hasattr(v.imb, 'implicit'):
                    pg0.append(v.imb.implicit)
                else:
                    for iv in v.imb:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imo'):
                if hasattr(v.imo, 'implicit'):
                    pg0.append(v.imo.implicit)
                else:
                    for iv in v.imo:
                        pg0.append(iv.implicit)
            if hasattr(v, 'ia'):
                if hasattr(v.ia, 'implicit'):
                    pg0.append(v.ia.implicit)
                else:
                    for iv in v.ia:
                        pg0.append(iv.implicit)
            if hasattr(v, 'attn'):
                if hasattr(v.attn, 'logit_scale'):
                    pg0.append(v.attn.logit_scale)
                if hasattr(v.attn, 'q_bias'):
                    pg0.append(v.attn.q_bias)
                if hasattr(v.attn, 'v_bias'):
                    pg0.append(v.attn.v_bias)
                if hasattr(v.attn, 'relative_position_bias_table'):
                    pg0.append(v.attn.relative_position_bias_table)
            if hasattr(v, 'rbr_dense'):
                if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                    pg0.append(v.rbr_dense.weight_rbr_origin)
                if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                    pg0.append(v.rbr_dense.weight_rbr_avg_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                    pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                    pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                    pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
                if hasattr(v.rbr_dense, 'vector'):
                    pg0.append(v.rbr_dense.vector)

        if opt.adam:
            self.optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group(
            {'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
        self.hyp = hyp
        lf = one_cycle(1, self.hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        self.ema = ModelEMA(self.model)
        self.start_epoch = 0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']
            # EMA
            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                self.ema.updates = ckpt['updates']

            # Results
            if ckpt.get('training_results') is not None:
                self.results_file.write_text(ckpt['training_results'])  # write results.txt

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if opt.resume:
                assert self.start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (
                weights, epochs)
            if opt.epochs < self.start_epoch:
                logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                            (weights, ckpt['epoch'], epochs))
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, state_dict

        self.device = device

        self.max_stride = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.number_layers = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        self.model.half().float()  # pre-reduce anchor precision
        self.best_fitness = 0

    def setup_dataset(self, opt, dataset_train, dataloader_train, dataloader_test, imgsz=640):
        # Anchors
        self.imgsz = imgsz
        self.imgsz_test = imgsz
        if not opt.noautoanchor:
            check_anchors(dataset_train, model=self.model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)

        self.nb = len(dataloader_train)  # number of batches
        self.loader_train = dataloader_train
        self.dataset_train = dataset_train
        self.loader_test = dataloader_test

    def setup_training(self):
        # Model parameters
        self.hyp['box'] *= 3. / self.number_layers  # scale to layers
        self.hyp['cls'] *= self.nc / 80. * 3. / self.number_layers  # scale to classes and layers
        self.hyp['obj'] *= (self.imgsz / 640) ** 2 * 3. / self.number_layers  # scale to image size and layers
        self.hyp['label_smoothing'] = self.opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.class_weights = labels_to_class_weights(self.dataset_train.labels, self.nc).to(
            self.device) * self.nc  # attach class weights
        self.model.names = self.names

        # Start training
        self.t0 = time.time()
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb),1000)  # number of warmup iterations, max(3 epochs, 1k iterations)

        # Resume
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        logger.info(f'Image sizes {self.imgsz} train, {self.imgsz_test} test\n'
                    f'Using {self.loader_train.num_workers} dataloader workers\n'
                    f'Logging results to {self.save_dir}\n'
                    f'Starting training for {self.epochs} epochs...')

        self.cuda = self.device.type != 'cpu'
        self.scaler = amp.GradScaler(enabled=self.cuda)
        self.compute_loss_ota = ComputeLossAuxOTA(self.model)  # init loss class
        self.compute_loss = ComputeLoss(self.model)  # init loss class

    def warmup_learning_rate(self, ni, epoch, lamba_function):
        xi = [0, self.nw]  # x interp

        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(ni, xi,
                                [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lamba_function(epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

    def train(self, opt):
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        lf = one_cycle(1, self.hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']

        for epoch in range(self.start_epoch, self.epochs):  # epoch ------------------------------------------------------------------
            self.model.train()
            ############## SETUP PROGRESS BAR VARIABLES ####################
            mloss = torch.zeros(4, device=device)  # mean losses
            pbar = enumerate(self.loader_train)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
            pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch --------------------------------------------------------
                ni = i + self.nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= self.nw:
                    self.warmup_learning_rate(ni, epoch, lf)

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / self.gs) * self.gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=self.cuda):
                    pred = self.model(imgs)  # forward
                    loss, loss_items = self.compute_loss_ota(pred, targets.to(device),
                                                             imgs)  # loss scaled by batch_size

                # Backward/load gradients
                self.scaler.scale(loss).backward()

                # Optimize
                if epoch % opt.accumulate == 0:
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.ema.update(self.model)

                # Print
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, self.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)
                # end batch ------------------------------------------------------------------------------------------------
            # end epoch ----------------------------------------------------------------------------------------------------
            self.scheduler.step()

            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == self.epochs
            if final_epoch:  # Calculate mAP
                results, maps, times = evaluation.test(self.data_dict,
                                                 batch_size=self.batch_size,
                                                 imgsz=self.imgsz_test,
                                                 model=self.model,
                                                 single_cls=opt.single_cls,
                                                 dataloader=self.loader_test,
                                                 save_dir=self.save_dir,
                                                 verbose=True,
                                                 plots=True,
                                                 # wandb_logger=self.wandb_logger,
                                                 compute_loss=self.compute_loss,
                                                 is_coco=self.is_coco,
                                                 v5_metric=False)
            else:
                if not opt.notest:  # Calculate mAP
                    results, maps, times = evaluation.test(self.data_dict,
                                                     batch_size=self.batch_size,
                                                     imgsz=self.imgsz_test,
                                                     model=self.model,
                                                     single_cls=opt.single_cls,
                                                     dataloader=self.loader_test,
                                                     save_dir=self.save_dir,
                                                     verbose=True,
                                                     plots=False,
                                                     # wandb_logger=wandb_logger,
                                                     compute_loss=self.compute_loss,
                                                     is_coco=self.is_coco,
                                                     v5_metric=opt.v5_metric)

            # Write
            with open(self.results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

            # Log
            # tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
            #         'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            #         'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
            #         'x/lr0', 'x/lr1', 'x/lr2']  # params
            # for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            #     if self.wandb_logger.wandb:
            #         self.wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > self.best_fitness:
                self.best_fitness = fi

            # self.wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': self.best_fitness,
                        'training_results': self.results_file.read_text(),
                        'model': deepcopy(self.model).half(),
                        'ema': deepcopy(self.ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                if (epoch % opt.save_interval) == 0:
                    torch.save(ckpt, self.wdir / 'epoch_{:03d}.pt'.format(epoch))
                del ckpt
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        logger.info('%g epochs completed in %.3f hours.\n' % (epoch, (time.time() - self.t0) / 3600))
        # self.wandb_logger.finish_run()
        torch.cuda.empty_cache()
        return results


def get_files_datasets(hyp, opt, data_dict, stride_max):
    train_path = data_dict['train']
    test_path = data_dict['val']
    # Image sizes
    batch_size = opt.batch_size

    nc = data_dict['nc']
    gs = max(stride_max, 32)  # grid size (max stride)

    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=-1,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=False, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class

    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,  # testloader
                                   hyp=hyp, augment=False, cache=opt.cache_images and not opt.notest, rect=opt.rect,
                                   rank=-1,
                                   world_size=opt.world_size, workers=opt.workers,
                                   pad=0.5, prefix=colorstr('val: '))[0]

    return dataset, dataloader, testloader, imgsz


def create_images_circles(image, coordinates, color, thickness=10):
    dims = np.array([dim for dim in reversed(image.shape[:2])])
    center_point = (coordinates[:2] * dims).astype(np.int32)
    radius = (coordinates[2:4] * dims / 2)
    axesLength = (int(radius[0]), int(radius[1]))
    angle = 0

    start_angle = 0
    end_angle = 360

    image = cv2.ellipse(image, center=center_point, axes=axesLength, angle=angle,
                        startAngle=start_angle, endAngle=end_angle, color=color, thickness=thickness)
    return image


def generate_coordinates(n_objects, image_shape, distance_border_x=0.1, distance_border_y=0.1, min_dim=20):
    coordinates = np.empty([n_objects, 4], dtype=np.int32)
    deltah = int(image_shape[1] * distance_border_x)
    deltaw = int(image_shape[0] * distance_border_y)

    coordinates[:, :2] = np.random.randint([deltaw, deltah], [image_shape[1] - deltaw, image_shape[0] - deltah],
                                           [n_objects, 2])
    upper_width = max(deltaw, min_dim + 10)
    upper_height = max(deltah, min_dim + 10)
    coordinates[:, 2:] = coordinates[:, :2] + np.random.randint([min_dim, min_dim], [upper_width, upper_height],
                                                                [n_objects, 2], dtype=np.int32)
    coordinates[:, 0:4:2] = np.clip(coordinates[:, 0:4:2], 0, image_shape[0])
    coordinates[:, 1:4:2] = np.clip(coordinates[:, 1:4:2], 0, image_shape[1])
    coordinates2 = coordinates.copy()

    coordinates2[:, :2] = (coordinates[:, :2] + coordinates[:, 2:4]) / 2
    coordinates2[:, 2:] = (coordinates[:, 2:4] - coordinates[:, :2])

    return coordinates2


class EllipseGenerator:
    def __init__(self, img_sizes, n_ellipses, dataset_size, n_colors=3):
        self.dataset_size = dataset_size
        self.img_sizes = img_sizes
        self.n_ellipses = n_ellipses
        self.colors = np.random.randint(0, 255, [n_colors, 3])
        self.n_colors = n_colors
        self.estimated_shape = []
        self.min_circle_size = 50
        for size in img_sizes:
            new_size = np.ceil(size / 32) * 32
            self.estimated_shape += [int(new_size)]

        self.shapes = np.array([img_sizes] * dataset_size, dtype=np.int32)
        self.labels = self.generate_labels()

    def generate_labels(self):
        labels = []
        dims = np.array([dim for dim in reversed(self.img_sizes)])
        for i in range(self.dataset_size):
            n_ellipses = np.random.randint(0, self.n_ellipses + 1)
            if n_ellipses > 0:
                coordinates = generate_coordinates(n_ellipses, self.img_sizes, min_dim=self.min_circle_size)
                classes_id = np.random.randint(0, self.n_colors, [n_ellipses])

                labels_out = np.zeros([n_ellipses, 5])
                labels_out[:, 0] = classes_id
                labels_out[:, 1:5] = coordinates
                labels_out[:, 1:3] /= dims[None]
                labels_out[:, 3:5] /= dims[None]

            else:
                labels_out = np.zeros([0, 5])
            labels += [labels_out]


        return labels

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        assert idx < self.dataset_size, "ERROR index {:d} too big for dataset".format(idx)

        img = np.zeros([self.img_sizes[0], self.img_sizes[1]] + [3], dtype=np.uint8)
        labels = self.labels[idx]
        n_ellipses = len(labels)
        for i in range(n_ellipses):
            colors = self.colors[int(labels[i, 0])]
            img = create_images_circles(img, labels[i, 1:], colors.tolist(), thickness=5)

        h, w = self.estimated_shape
        h0, w0 = self.img_sizes
        img, ratio, pad = letterbox(img, self.estimated_shape,
                                    auto=False, scaleup=True)

        labels_out = torch.zeros([n_ellipses, 6])
        labels_out[:, 1:] = torch.from_numpy(self.labels[idx])
        # ratioh_padded = (h - pad[1] * 2) / h0
        # ratiow_padded = (w - pad[0] * 2) / w0
        ratioh = h / h0
        ratiow = w / w0
        shapes = (h0, w0), ((ratioh, ratiow), pad)  # for COCO mAP rescaling
        # labels_out[:, 2:6:2] = (labels_out[:, 2:6:2]) * ratio[0] + pad[0]
        # labels_out[:, 3:6:2] = (labels_out[:, 3:6:2]) * ratio[1] + pad[1]
        # labels_out[:, 2:6:2] = labels_out[:, 2:6:2] / self.estimated_shape[1]
        # labels_out[:, 3:6:2] = labels_out[:, 3:6:2] / self.estimated_shape[0]
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), labels_out, 'image_{:d}.jpg'.format(idx), shapes


def create_ellipse_dataloaders(img_size, batch_size, max_stride=64, workers=8, n_images=200, n_ellipses=3):
    dataset = EllipseGenerator(img_size, n_ellipses, n_images)

    batch_size = min(batch_size, len(dataset))
    sampler_train = torch.utils.data.RandomSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             sampler=sampler_train,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)
    sampler_test = None
    dataloader_test = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             sampler=sampler_test,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)

    imgsz, imgsz_test = [check_img_size(x, max_stride) for x in img_size]  # verify imgsz are gs-multiples
    return dataset, dataloader, dataloader_test, imgsz

def main(hyp, opt, device):
    ts = training_session(hyp, opt, device)
    max_stride = ts.max_stride
    if opt.use_ellipse:
        dataset_train, dataloader_train, dataloader_test, imgsz = create_ellipse_dataloaders(opt.img_size,
                                                                                             opt.batch_size,
                                                                                             max_stride,
                                                                                             workers=opt.workers)
    else:
        dataset_train, dataloader_train, dataloader_test, imgsz = get_files_datasets(hyp, opt, ts.data_dict,
                                                                                        max_stride)
    ts.setup_dataset(opt, dataset_train, dataloader_train, dataloader_test, imgsz)
    ts.setup_training()
    ts.train(opt)


if __name__ == '__main__':
    opt = Namespace()
    ##### PATH SETTINGS ################
    opt.data = 'data/vehicle.yaml'
    opt.cfg = 'cfg/training/yolov7-vehicle.yaml'
    opt.hyp = 'data/hyp.scratch.custom.yaml'
    opt.project = 'runs/train'
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.name = 'yolov7-6e'
    opt.save_dir = 'runs_ellipse'
    ########################################################
    ##### Dataset info ####################################
    ########################################################
    opt.update_dataset = True
    opt.data_train = '/home/luis/2024/Hands_on/archive/data/images'
    opt.nc = 3
    opt.names_classes = ['ellipse{:d}'.format(i) for i in range(opt.nc)] # ['vehicle'] #['ripe','unripe','green']
    opt.single_cls = opt.nc==1
    torch.cuda.empty_cache()
    ########################################################
    ##### HYPER-PARAMS OPTIMIZATION ########################
    ########################################################
    opt.weights = 'runs_ellipse/weights/last.pt'  # 'runs_ellipse/weights/last.pt'
    opt.batch_size = 8
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.multi_scale = False
    opt.epochs = 66
    opt.img_size = [192, 192]
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.adam = True
    ########################################################
    #### DEFAULT SETTING (DON'T CHANGE)
    ########################################################
    opt.seed = 999
    opt.save_interval = 10
    opt.world_size = 1
    opt.rect = True
    opt.global_rank = -1
    set_logging(opt.global_rank)
    opt.resume = False
    opt.workers = 0
    opt.accumulate = 2
    opt.cache_images = True
    opt.label_smoothing = 0.0
    opt.bbox_interval = -1
    opt.save_period = -1
    opt.noautoanchor = False
    opt.nosave = False
    opt.notest = False
    opt.v5_metric = False
    opt.use_ellipse = False

    ########################################################

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    logger.info(opt)
    # Train
    main(hyp, opt, device)
