import argparse
import gc
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, SoftmaxWeightedLoss
from utils import ContrastiveLoss
from torchvision import transforms

def trim_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

def trainer_Myops(args, model, snapshot_path):
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            # M_MMAP_THRESHOLD = -3, M_TRIM_THRESHOLD = -1
            libc.mallopt(ctypes.c_int(-3), ctypes.c_int(65536))
            libc.mallopt(ctypes.c_int(-1), ctypes.c_int(65536))
        except Exception:
            pass
    from datasets.dataset_Myops import Myops_dataset, RandomGenerator
    log_file_path = os.path.join(snapshot_path, "log.txt")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    # Ensure FileHandler exists and writes to log.txt
    if not any(isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(log_file_path) for h in logger.handlers):
        fh = logging.FileHandler(log_file_path, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Ensure StreamHandler prints to terminal
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == sys.stdout for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    start_epoch = getattr(args, 'start_epoch', 0)
    db_train = Myops_dataset(base_dir=args.root_path, base_dir1=args.root_path1, base_dir2=args.root_path2, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    num_workers = getattr(args, 'num_workers', 0)
    pin_memory = bool(getattr(args, 'pin_memory', 0))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    con_loss = ContrastiveLoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader) 
    iter_num = start_epoch * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        do_contrast = epoch_num > args.start_contrast_epoch
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].cuda(non_blocking=pin_memory)
            image1_batch = sampled_batch['image1'].cuda(non_blocking=pin_memory)
            image2_batch = sampled_batch['image2'].cuda(non_blocking=pin_memory)
            label_batch = sampled_batch['label'].cuda(non_blocking=pin_memory)
            del sampled_batch

            out_pre, dec_seg, features_embedding_list, text_embedding_list= model(image_batch, image1_batch, image2_batch, do_contrast)
            ignores = ([2,3],[0],[0])
            loss_all = 0
            if do_contrast:
                for i in range(len(features_embedding_list)):
                    feature_list = features_embedding_list[i]
                    ignore = ignores[i]
                    loss_con = con_loss(feature_list,
                                        label_batch,
                                        text_embedding_list,
                                        ignore,
                                        sample_num = args.contrast_sample_num,
                                        )
                    loss_all += loss_con
                loss_all = loss_all /len(features_embedding_list)
            else:
                loss_all = 0
            out_cross_loss = ce_loss(out_pre, label_batch)
            out_dice_loss = dice_loss(out_pre, label_batch, softmax=True)
            out_loss = 0.2* out_cross_loss + 0.8* out_dice_loss
            dec_cross_loss = torch.zeros(1).cuda().float()
            dec_dice_loss = torch.zeros(1).cuda().float()
            for dec_pred in dec_seg:
                dec_cross_loss += ce_loss(dec_pred, label_batch)
                dec_dice_loss += dice_loss(dec_pred, label_batch, softmax=True)
            dec_loss = 0.2* dec_cross_loss + 0.8* dec_dice_loss
            
            if epoch_num < args.region_fusion_start_epoch:
                loss = out_loss * 0.0 + dec_loss+ loss_all * args.contrast_w
            else:
                loss = out_loss + 0.5 * dec_loss+ 0.5 * loss_all * args.contrast_w

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', out_dice_loss.item(), iter_num)

            if iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f, loss_fuse: %f' % (iter_num, loss.item(), out_dice_loss.item()))
                sample_idx = 0 if image_batch.shape[0] == 1 else 1
                image = image_batch[sample_idx, 0:1, :, :].detach().cpu()
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                writer.add_image('train/Image', image, iter_num)
                with torch.no_grad():
                    out_pre_img = torch.argmax(torch.softmax(out_pre.detach(), dim=1), dim=1, keepdim=True).cpu()
                writer.add_image('train/Prediction', out_pre_img[sample_idx, ...] * 50, iter_num)
                labs = label_batch[sample_idx, ...].unsqueeze(0).detach().cpu() * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                writer.flush()

            del out_pre, dec_seg, features_embedding_list, text_embedding_list, loss
            del image_batch, image1_batch, image2_batch, label_batch

        writer.flush()
        trim_memory()
        save_interval = 20  
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 5:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        if epoch_num >= max_epoch:    
            break

    iterator.close()
    writer.close()
    return "Training Finished!"
