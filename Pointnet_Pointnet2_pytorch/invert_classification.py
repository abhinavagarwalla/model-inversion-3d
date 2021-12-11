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
import open3d.ml.torch as ml3d

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
    parser.add_argument('--class_to_invert', default=0, type=int, help='Specify the class to invert')
    parser.add_argument('--output_dir', default=None, required=True, type=str, help='output dir to save file')
    parser.add_argument('--regularize', action='store_true', default=False, help='use regularisation')
    return parser.parse_args()

def vispcd(points, iter, output_dir):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().detach().numpy()[0].T)
    pcd.paint_uniform_color([1, 0, 0])

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    # o3d.visualization.draw_geometries([pcd])
    # viewer.run()
    # viewer.capture_screen_image(f'./output/{iter}.png')
    # viewer.close()

    hull, sel_points = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((0, 0, 0))
    viewer.add_geometry(hull_ls)
    # viewer.run()

    '''alpha = 0.03
    print(f"alpha={alpha:.3f}")
    pcd_hull = o3d.geometry.PointCloud()
    pcd_hull.points = pcd.points[sel_points]
    pcd_hull.paint_uniform_color([1, 0, 0])
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_hull, alpha)
    mesh.compute_vertex_normals()
    viewer.add_geometry(mesh)'''


    viewer.poll_events()
    viewer.update_renderer()
    viewer.capture_screen_image(f'./{output_dir}/{iter}.png')
    viewer.destroy_window()

# Abstracted as a class so that we can add regularizers and custom loss
class InversionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, predicted, ground_truth):
        return self.criterion(predicted, ground_truth)

# Abstracted as a class so that we can add experiment with different initialisations
class PointInitializer(nn.Module):
    def __init__(self, type, num_points=1024):
        super().__init__()

        assert(type is not None)
        if type == 'uniform_cube':
            self.xyz = torch.zeros((1, 3, num_points)).uniform_(-0.2,0.2).cuda()
        elif type == 'zero':
            # self.xyz = torch.zeros((1, 3, num_points)).cuda()
            self.xyz = torch.zeros((1, 3, num_points)).uniform_(-0.0001,0.0001).cuda()
        elif type == 'spherical_boundary':
            raise NotImplementedError
        elif type == 'spherical_volume':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.xyz.requires_grad_(True)

def invert(model, num_class=40, class_to_invert=0, num_steps=1000):
    assert(class_to_invert < num_class)

    classifier = model.eval()

    # points = PointInitializer(type='uniform_cube', num_points=10000).xyz
    points = PointInitializer(type='zero', num_points=10000).xyz #works better
        
    #visualize random points
    output_dir = os.path.join(args.output_dir, str(args.class_to_invert))
    os.makedirs(output_dir, exist_ok=True)
    vispcd(points, iter=0, output_dir=output_dir)

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
        print("Mean Max Min x: ", points.mean().item(), points.max().item(), points.min().item())

        if args.regularize and j % 10 == 0:
            #average using neighbouring
            alpha = 0.03
            with torch.no_grad():
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.cpu().detach().numpy()[0].T)

                '''mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                mesh.filter_smooth_taubin()
                points_np = np.asarray(mesh.vertices)
                points.data = torch.from_numpy(points_np)'''

                #distances = pcd.compute_mahalanobis_distance()
                #print(distances.shape)

                '''ans = ml3d.ops.knn_search(points[0].t().cpu(),
                    points[0].t().cpu(), k=2,
                    points_row_splits=torch.LongTensor([0,len(points)]),
                    queries_row_splits=torch.LongTensor([0,len(points)]),
                    #ignore_query_point=True,
                    return_distances=True)
                print(ans)'''

                _, sel_points = pcd.compute_convex_hull()

                distances = torch.cdist(points.transpose(1, 2), points.transpose(1, 2))
                #print(distances.shape)

                values, neighbors = torch.topk(distances[0], k=2, largest=False)
                new_points = torch.zeros_like(points)
                for i in range(new_points.shape[-1]):
                    if False:#i in sel_points:
                        new_points[:, :, i] = points[:, :, i]
                    else:
                        new_points[:, :, i] = points[:, :, neighbors[i]].mean(dim=-1)

                points.data = new_points.data


        
        if j % 50 == 0 and pred_choice.item() == target.item():
            vispcd(points, iter=j, output_dir=output_dir)
            # vispcd(l2_points_pos, iter=j)
            # vispcd(l1_points_pos, iter=j)

        
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

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    inverted_points = invert(classifier.eval(), num_class=num_class, class_to_invert=args.class_to_invert)
    print(inverted_points)


if __name__ == '__main__':
    args = parse_args()
    main(args)
