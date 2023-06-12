import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Network_woCalibrate

from dataset import ImageLowSemDataset

parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--weights', type=str, default="weights/pretrained_SCI/medium.pt", help='weights after training with semantic')
parser.add_argument('--test_dir', type=str, default='data/LOL/test15/low', help='testing data directory')
parser.add_argument('--test_output_dir', type=str, default='test_output', help='testing output directory')
args = parser.parse_args()

save_path = args.test_output_dir
os.makedirs(save_path, exist_ok=True)

import subprocess
print("sam is working...")
# subprocess.call(['python', 'sam.py', '--source_dir', args.test_dir])
print("sam is done...")

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def model_init(model):
    weights_dict = torch.load(args.weights)
    model_dict = model.state_dict()
    weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
    model_dict.update(weights_dict)
    model.load_state_dict(model_dict)

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 
    
    TestDataset = ImageLowSemDataset(img_dir = args.test_dir,
                                       sem_dir = os.path.join(os.path.split(args.test_dir)[0], 'low_semantic'))
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1, shuffle = False,
        pin_memory=True
    )
    
    model = Network_woCalibrate()
    model_init(model)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        for batch_idx, (in_, sem_, imgname_, semname_ ) in enumerate(test_queue):
            in_ = in_.cuda()
            sem_ = sem_.cuda()
            image_name = os.path.splitext(imgname_[0])[0]
            i, r = model(in_, sem_)
            u_name = '%s.png' % (image_name)
            print('test processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)


if __name__ == '__main__':
    main()
