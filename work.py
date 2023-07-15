import argparse
import os
from torch.utils import data

from clothing_mask import image_mask

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--clothing_dir', type=str, default='./datasets/test/')
    parser.add_argument('--clothing_mode', type=str, default='cloth')
    parser.add_argument('--clothing_list', type=str, default='clothing.txt')


    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()
    print(opt)

    im = image_mask(opt)
    im.mask(opt)
    im.save_mask(opt)
    

if __name__ == '__main__':
    main()