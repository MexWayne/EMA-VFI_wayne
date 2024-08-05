import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
from imageio import mimsave

if __name__ == "__main__":

    '''==========import from our code=========='''
    sys.path.append('.')
    import config as cfg
    from Trainer import Model
    from benchmark.utils.padder import InputPadder

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ours_t', type=str)
    parser.add_argument('--n', default=16, type=int)
    args = parser.parse_args()
    assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'


    '''==========Model setting=========='''
    TTA = True
    if args.model == 'ours_small_t':
        TTA = False
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F = 16,
            depth = [2, 2, 2, 2, 2]
        )
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F = 32,
            depth = [2, 2, 2, 4, 4]
        )
    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()

    input_path = r"/home/mex/Desktop/learn_videointerp/mexwayne_ECCV2022-RIFE/img/memc/"

    img_list = os.listdir(input_path)
    img_list.sort(key=lambda x: int(x.split('.')[0]))  # sort

    count = 0
    for i in range(len(img_list)-1):
        print(f'=========================Start Generating=========================')

        I0 = cv2.imread(input_path + str(i) + '.jpg')
        I2 = cv2.imread(input_path + str(i + 1) + '.jpg')

        #I0 = cv2.resize(I0,dsize=None,fx=0.25,fy=0.25,interpolation=cv2.INTER_LINEAR)
        #I2 = cv2.resize(I2,dsize=None,fx=0.25,fy=0.25,interpolation=cv2.INTER_LINEAR)


        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

        padder = InputPadder(I0_.shape, divisor=32)
        I0_, I2_ = padder.pad(I0_, I2_)

        images = [I0[:, :, ::-1]]
        preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i+1)*(1./args.n) for i in range(args.n - 1)], fast_TTA=TTA)
        for pred in preds:
            images.append((padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])
        images.append(I2[:, :, ::-1])


        #mimsave('example/out_Nx.gif', images, fps=args.n)
        for i in range(len(images)):
            #cv2.imwrite('./memc_result/img{}.jpg'.format(count), images[i][:, :, ::-1])
            cv2.imwrite('./memc_result/{}.jpg'.format(str(count).zfill(4)), images[i][:, :, ::-1])
            count += 1


        print(f'=========================Done=========================')