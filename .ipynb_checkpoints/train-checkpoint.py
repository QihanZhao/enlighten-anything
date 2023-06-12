import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from model import *
from multi_read_data import MemoryFriendlyLoader

# 该脚本命令行参数 可选项
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=1000, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='exp/', help='location of the data corpus')
parser.add_argument('--pretrain', type=str, default=None, help='pretrained weights directory')
parser.add_argument('--frozen', type=bool, default=False, help='froze the original weights')
parser.add_argument('--train_dir', type=str, default='data/LOL/train480/low', help='training data directory')
parser.add_argument('--val_dir', type=str, default='data/LOL/val5/low', help='training data directory')
parser.add_argument('--m', type=str, default=None, help='comment')
args = parser.parse_args()

# 根据命令行参数进行设置
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

snapshot_dir = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H:%M:%S"))
utils.create_exp_dir(snapshot_dir, scripts_to_save=glob.glob('*.py'))
model_path = snapshot_dir + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = snapshot_dir + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(snapshot_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def model_init(model):
    if(args.pretrain==None):
        # model.enhance.in_conv.apply(model.weights_init)
        # model.enhance.conv.apply(model.weights_init)
        # model.enhance.out_conv.apply(model.weights_init)
        # model.calibrate.in_conv.apply(model.weights_init)
        # model.calibrate.convs.apply(model.weights_init)
        # model.calibrate.out_conv.apply(model.weights_init)
        
        # model.enhance.apply(model.weights_init)
        # model.calibrate.apply(model.weights_init)
 
        model.apply(model.weights_init)

    else:
        pretrained_dict = torch.load(args.pretrain)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        if(args.frozen == True):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.enhance.fusion.parameters():
            # for param in model.enhance.parameters():
                param.requires_grad = True

def main():
    logging.info("train file name = %s", os.path.split(__file__))
    logging.info("args = %s", args)
    
    if not torch.cuda.is_available(): #默认使用GPU,且强制使用
        logging.info('no gpu device available')
        sys.exit(1)
    else:
        logging.info('gpu device = %s' % args.gpu)

    # GPU训练的准备1: 数据加载器的采样
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    # 模型
    # model = Network(stage=args.stage)
    model = Network_woCalibrate()
    model_init(model)
        # GPU训练的准备2: 模型放到GPU
    model = model.cuda()
        # 打一个日志记录模型大小
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    
    # 优化器
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=args.lr*100, betas=(0.9, 0.999), weight_decay=3e-4)

    # 数据集 
    TrainDataset = MemoryFriendlyLoader(img_dir=args.train_dir,        #'../LOL/train480/semantic'
                                        sem_dir = os.path.join(os.path.split(args.train_dir)[0], 'low_semantic')) 
    ValDataset = MemoryFriendlyLoader(img_dir=args.val_dir, 
                                      sem_dir = os.path.join(os.path.split(args.val_dir)[0], 'high_semantic'))
    # from torch.utils.data import RandomSampler
    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        shuffle=True,
        # sampler=RandomSampler(TrainDataset, generator=torch.Generator(device='cuda'))
        pin_memory=True, 
    )
    val_queue = torch.utils.data.DataLoader(
        ValDataset, batch_size=1, shuffle=False,
        pin_memory=True
    )


    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_idx, (in_, sem_, imgname_, semname_ ) in enumerate(train_queue):
            # GPU训练的准备3: 数据放到GPU
            in_ = in_.cuda() #从dataset的设计来看, requires_grad默认是False
            sem_ = sem_.cuda()
            
            # 向前传播；计算损失
            loss = model._loss(in_, sem_)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            # 更新梯度
            optimizer.step()

            losses.append(loss.item())
            logging.info('train: epoch %3d: batch %3d: loss %f', epoch, batch_idx, loss)


        logging.info('train: epoch %3d: average_loss %f', epoch, np.average(losses))
        logging.info('----------')
        utils.save(model, os.path.join(model_path, f'weights_{epoch}.pt'))


        # model.eval()
        # with torch.no_grad():
        #     for batch_idx, (in_, sem_, imgname_, semname_ ) in enumerate(val_queue):
        #         in_ = in_.cuda()
        #         sem_ = sem_.cuda()
        #         image_name = os.path.splitext(imgname_[0])[0]
        #         illu_list, ref_list, input_list, atten= model(in_, sem_)
        #         u_name = f'{image_name}_{epoch}.png' 
        #         u_path = image_path + '/' + u_name
        #         save_images(ref_list[0], u_path)
                
        with torch.no_grad():
            for batch_idx, (in_, sem_, imgname_, semname_ ) in enumerate(val_queue):
                in_ = in_.cuda()
                sem_ = sem_.cuda()
                image_name = os.path.splitext(imgname_[0])[0]
                i, r = model(in_, sem_)
                u_name = '%s.png' % (image_name)
                print('validation processing {}'.format(u_name))
                u_path = image_path + '/' + u_name
                save_images(r, u_path)
        # break

if __name__ == '__main__':
    
    main()
