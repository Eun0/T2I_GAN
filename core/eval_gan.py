import os
import sys


PROJ_DIR = os.path.abspath(os.path.realpath(__file__).split('core/'+os.path.basename(__file__))[0])
sys.path.append(PROJ_DIR)

import argparse
import random
import glob

import wandb
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image 
import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pytorch_fid.fid_score import calculate_fid_given_paths

from config import cfg, cfg_from_file
from dataset import WordTextDataset, SentTextDataset, index_to_sent
from model.encoder import RNN_ENCODER, SBERT_ENCODER, SBERT_FT_ENCODER

from model.df_gan import DF_GEN, DF_DISC
from model.xmc_gan import XMC_DISC

from utils.logger import setup_logger
from utils.miscc import count_params, truncated_z_sample

import multiprocessing
multiprocessing.set_start_method('spawn', True)

_TEXT_DATASET = {"WORD":WordTextDataset, "SENT":SentTextDataset, }
_TEXT_ARCH = {"RNN":RNN_ENCODER, "SBERT":SBERT_ENCODER, "SBERT_FT":SBERT_FT_ENCODER}
_GEN_ARCH = {"DF_GEN":DF_GEN, }
_DISC_ARCH = {"DF_DISC":DF_DISC, "XMC_DISC":XMC_DISC}


def parse_args():
    parser = argparse.ArgumentParser(description='Eval T2I-GAN')
    parser.add_argument('--cfg',type=str,default='cfg/df_gan/df_gan_sbert_ft_img_match_global_sent_withGD.yml')
    parser.add_argument('--gpu',dest = 'gpu_id', type=int, default=0)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--resume_epoch',type=int,default=80)
    parser.add_argument('--max_epoch',type=int,default=-1)
    parser.add_argument('--log_type',type=str,default='wdb')
    parser.add_argument('--bs',type=int,default=100)
    parser.add_argument('--imsize',type=int,default=128)
    parser.add_argument('--mode',type=str,default='all')
    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(cfg, args, loader, state_epoch, text_encoder, netG, logger, img_dir, num_samples = 6000, writer = None):

    netG.eval()

    cnt = 0
    save_dir = f'{img_dir}/test'
    org_dir = f'{img_dir}/org'

    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(org_dir,exist_ok=True)

    save_org = True if len(os.listdir(org_dir)) != num_samples else False
    #save_org = True

    for data in loader:
        imgs,texts_lst,keys = data 
        texts = texts_lst[0]
        batch_size = imgs.size(0)
        
        caps = texts[0]
        cap_lens = texts[1]

        sents = caps if cfg.TEXT.TYPE=='SENT' else index_to_sent(test_set.i2w,caps)

        if not cfg.TEXT.JOINT_FT:
            words_embs, sent_embs, mask = text_encoder(caps, cap_lens)
        else:
            words_embs, sent_embs, mask, bert_embs = text_encoder(caps, cap_lens)
        words_embs, sent_embs = words_embs.detach(), sent_embs.detach()
        noise = truncated_z_sample(batch_size, cfg.TRAIN.NOISE_DIM, seed=args.seed)
        noise = torch.from_numpy(noise).float().cuda()
        #noise = torch.randn(sent_embs.size(0), cfg.TRAIN.NOISE_DIM).cuda()
        fake_imgs = netG(noise = noise, sent_embs = sent_embs, words_embs = words_embs, mask = mask)

        for j in range(batch_size):
            im = fake_imgs[j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1,2,0))
            im = Image.fromarray(im)
            fullpath = f'{save_dir}/{sents[j]}_{keys[j]}.png'
            im.save(fullpath)
            if save_org:
                org_im = imgs[j].data.cpu().numpy()
                org_im = (org_im + 1.0)*127.5
                org_im = org_im.astype(np.uint8)
                org_im = np.transpose(org_im, (1,2,0))
                org_im = Image.fromarray(org_im)
                fullpath = f'{org_dir}/{sents[j]}_{keys[j]}.png'
                org_im.save(fullpath)

        cnt += batch_size

        if cnt >= num_samples:
            break 
    
    fid_score = calculate_fid_given_paths([org_dir,save_dir], batch_size = 100, device = torch.device('cuda'), dims = 2048)
    logger.info(f' epoch {state_epoch}, FID : {fid_score}')

    if args.log_type =='wdb':
        wandb.log({"FID":fid_score,"epoch":state_epoch})
    else:
        writer.add_scalar('FID', fid_score, state_epoch)

if __name__ == '__main__':

    args = parse_args()
    cfg_from_file(args.cfg)

    if args.imsize != -1:
        cfg.IMG.SIZE = args.imsize
    if args.bs != -1:
        cfg.TRAIN.BATCH_SIZE = args.bs
    if args.max_epoch != -1:
        cfg.TRAIN.MAX_EPOCH = args.max_epoch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    state_epoch = args.resume_epoch

    output_dir = f'{PROJ_DIR}/output/{cfg.DATASET_NAME}{cfg.IMG.SIZE}_{cfg.CONFIG_NAME}_{args.seed}'
    
    img_dir = output_dir + '/test'
    log_dir = img_dir + '/log'
    model_dir = output_dir + '/model'

    #os.makedirs(output_dir,exist_ok=True)    
    os.makedirs(img_dir,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    #os.makedirs(model_dir,exist_ok=True)

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    writer = None
    if args.log_type == 'wdb':
        wandb.init(project = f'Finals', config = cfg)
        wandb.run.name = cfg.CONFIG_NAME
    else:
        writer = SummaryWriter(log_dir = log_dir)
    
    logger = setup_logger(name = cfg.CONFIG_NAME, save_dir = log_dir, distributed_rank = 0)
    logger.info('Using config:')
    logger.info(cfg)
    logger.info(f'seed now is : {args.seed}')

    ##### dataset
    img_size = cfg.IMG.SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE


    data_dir = f'{PROJ_DIR}/data/{cfg.DATASET_NAME}'
    data_arch = _TEXT_DATASET[cfg.TEXT.TYPE]
    
    test_set = data_arch(data_dir = data_dir, mode = 'test', transform = transforms.Resize((img_size,img_size)), cfg = cfg)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, drop_last = True, shuffle = False, num_workers = int(cfg.TRAIN.NUM_WORKERS))

    text_arch = _TEXT_ARCH[cfg.TEXT.ENCODER_NAME]
    text_encoder = text_arch(cfg = cfg)
    text_encoder = text_encoder.cuda()


    if cfg.TEXT.ENCODER_DIR != '':
        text_encoder.load_state_dict(torch.load(f'{PROJ_DIR}/{cfg.TEXT.ENCODER_DIR}', map_location='cuda'), strict=False)
    elif cfg.TEXT.JOINT_FT and state_epoch != 0:
        text_encoder.load_state_dict(torch.load(f'{model_dir}/text_encoder{state_epoch:03d}.pth', map_location='cuda'), strict=False)
        logger.info('Load fine-tuned text encoder, epoch {state_epoch}')
        
    text_encoder.eval()

    g_arch = _GEN_ARCH[cfg.GEN.ENCODER_NAME]
    
    cond_dim = cfg.TEXT.EMBEDDING_DIM if not cfg.TEXT.JOINT_FT else cfg.TRAIN.NEF

    netG = g_arch(cfg, cond_dim = cond_dim).cuda()
    netG.load_state_dict(torch.load(f'{model_dir}/netG_{state_epoch:03d}.pth',map_location='cuda'))

    logger.info(f'netG # of parameters: {count_params(netG)}')

    if args.mode =='all':
        for epoch in range(state_epoch, cfg.TRAIN.MAX_EPOCH + 1):
            netG.load_state_dict(torch.load(f'{model_dir}/netG_{epoch:03d}.pth',map_location='cuda'))
            if cfg.TEXT.JOINT_FT:
                text_encoder.load_state_dict(torch.load(f'{model_dir}/text_encoder{epoch:03d}.pth', map_location='cuda'), strict=False)
            eval(cfg = cfg, args = args, loader = test_loader, state_epoch = epoch, text_encoder = text_encoder, netG = netG, logger = logger, img_dir = img_dir, num_samples = 30000, writer = writer)
    else:
        eval(cfg = cfg, args = args, loader = test_loader, state_epoch = state_epoch, text_encoder = text_encoder, netG = netG, logger = logger, img_dir = img_dir, num_samples = 30000, writer = writer)
        

    

    

    