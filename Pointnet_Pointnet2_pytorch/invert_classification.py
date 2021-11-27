"""
Author: Abhinav
Adapted from : Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from torch import nn

import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Inverting')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()

def vispcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().detach().numpy()[0].T)
    o3d.visualization.draw_geometries([pcd])

# Abstracted as a class so that we can add regularizers and custom loss
class InversionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, predicted, ground_truth):
        return self.criterion(predicted, ground_truth)

# Abstracted as a class so that we can add experiment with different initialisations
class PointInitializer(nn.Module):
    def __init__(self, type):
        super().__init__()

        assert(type is not None)
        if type == 'uniform_cube':
            self.xyz = torch.zeros((1, 3, 1024)).uniform_(-1,1).cuda()
        else:
            raise NotImplementedError

        self.xyz.requires_grad_(True)

def invert(model, num_class=40, class_to_invert=0, num_steps=1000):
    assert(class_to_invert < num_class)

    classifier = model.eval()

    points = PointInitializer(type='uniform_cube').xyz
        
    #visualize random points
    # vispcd(points)

    target = torch.Tensor([class_to_invert]).long().cuda()
    criterion = InversionLoss()
    optimizer = torch.optim.SGD([points], lr=1e-4, momentum=0.9)

    for j in tqdm(range(num_steps)):
        # if not args.use_cpu:
        #     points, target = points.cuda(), target.cuda()

        # points = points.transpose(2, 1)

        pred, l3_points_feat, l2_points_pos, l1_points_pos = classifier(points)

        loss = criterion(pred, target)
        print("Loss: ", loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        pred_choice = pred.data.max(1)[1]
        print("Predicted: ", pred_choice)
        print("Mean Max Min x: ", points.mean(), points.max(), points.min())

        if j % 100 == 0 and pred_choice.item() == target.item():
            vispcd(points)
            vispcd(l2_points_pos)
            vispcd(l1_points_pos)

    return points.cpu().detach().numpy()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # with torch.no_grad():
    inverted_points = invert(classifier.eval(), num_class=num_class, class_to_invert=0)
    print(inverted_points)

    # Add code for visualization for points

if __name__ == '__main__':
    args = parse_args()
    main(args)
