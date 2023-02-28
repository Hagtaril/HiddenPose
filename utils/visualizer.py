import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import tqdm
import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import wandb


def volume_log(volume, res_path, name, index):
    # table.add_data("pred_joints", "MJPJE", "view_front", "view_left", "view_top", "refine_result_tt")
    # os.makedirs(res_path + '/volume', exist_ok=True)
    os.makedirs(res_path + '/figure/projection_of_vol', exist_ok=True)
    os.makedirs(res_path + '/projection_of_vol', exist_ok=True)
    # volume_path = res_path + '/volume' + f'/{name}_{index}.pt'
    figure_path = res_path + '/projection_of_vol' + \
        f'/{name}_{index}.jpg'
    # torch.save(volume[0, 0], volume_path)
    # im = volume[0, 0].sum(0).detach().cpu().numpy()
    # plt.figure(1)
    # plt.imshow(im)
    # wandb.log({f"projection of {name}": plt}, commit=False)
    # plt.savefig(figure_path)

    debug = True
    if debug:
        plt.figure(1)
        # plt.savefig(f'./debug/{index}_dh_vol.jpg')
        im = volume[0, 0].sum(0).detach().cpu().numpy()
        j = np.loadtxt('./1.txt')
        for i in range(24):
            h = int(j[i,1]) 
            w = int(j[i,2])
            im[h:h+2, w:w+2] += im.max() / 2
        plt.imshow(im)
        plt.savefig(res_path + '/projection_of_vol' + \
                    f'/{name}_{index}_front_joints.jpg')
        # wandb.log({f"projection of {name}": plt}, commit=False) TODO

        im = volume[0, 0].sum(1).detach().cpu().numpy()
        for i in range(24):
            d = int(j[i,0])
            w = int(j[i,2])
            im[d:d+2, w:w+2] += im.max() / 2 
        plt.imshow(im)
        plt.savefig(res_path + '/figure/projection_of_vol' + \
                    f'/{name}_{index}_top_joints.jpg')

        im = volume[0, 0].sum(2).detach().cpu().numpy()
        for i in range(24):
            d = int(j[i,0]) 
            h = int(j[i,1])
            im[d:d+2, h:h+2] += im.max() / 2
        plt.imshow(im)
        plt.savefig(res_path + '/figure/projection_of_vol' + \
                    f'/{name}_{index}_left_joints.jpg')


    plt.clf()


def joints_log(joints, res_path, joint_name, index=0):
    os.makedirs(res_path, exist_ok=True)
    joints_txt_path = os.path.join(res_path, "txt")
    os.makedirs(joints_txt_path, exist_ok=True)
    np.savetxt(joints_txt_path + '/' + joint_name + f"{index}.txt", joints)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(x_major_locator)
    ax.zaxis.set_major_locator(x_major_locator)

    # joints = (joints / 256.0 - 0.5) * 2.5  # TODO 将具体数字改成congfig中变量

    # xs = []
    # ys = []
    # zs = []

    ds = []
    hs = []
    ws = []

    for i in range(joints.shape[0]):
        # xs.append(joints[i, 0])
        # ys.append(joints[i, 1])
        # zs.append(-joints[i, 2])

        ds.append(joints[i, 0])
        hs.append(joints[i, 1])
        ws.append(joints[i, 2])


    # fig = plt.figure()
    # t = ys
    # ys = zs
    # zs = t
    # ys = ys
    ds, hs, ws = ws, ds, hs
    ax.scatter(ds, hs, ws)
    renderBones(ax, ds, hs, ws)

    ax.set_xlabel('D')
    ax.set_ylabel('H')
    ax.set_zlabel('W')
    ax.set_title(joint_name)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    ax.invert_zaxis()
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    ax.figure.savefig(res_path + '/' + joint_name + f"{index}")
    # wandb.log({f"joints of {joint_name}": wandb.Image(ax.figure)}, commit=False) TODO
    plt.clf()


def renderBones(ax, xs, ys, zs):
    link = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 9],
        [7, 10],
        [8, 11],
        [9, 12],
        [9, 13],
        [9, 14],
        [12, 15],
        [13, 16],
        [14, 17],
        [16, 18],
        [17, 19],
        [18, 20],
        [19, 21],
        [20, 22],
        [21, 23],
    ]
    for l in link:
        index1, index2 = l[0], l[1]
        ax.plot([xs[index1], xs[index2]], [ys[index1], ys[index2]],
                [zs[index1], zs[index2]], linewidth=1, label=r"$x=y=z$")


def threeviews_log(re, path, name, index=0):
    volumn_MxNxN = re.detach().cpu().numpy()[0, -1]

    # get rid of bad points
    # if name == 'output':
    #     zdim = volumn_MxNxN.shape[0]
    # elif name == 'volume':
    zdim = volumn_MxNxN.shape[0]
    volumn_MxNxN = volumn_MxNxN[:zdim]
    # print('volumn min, %f' % volumn_MxNxN.min())
    # print('volumn max, %f' % volumn_MxNxN.max())
    # volumn_MxNxN[:5] = 0
    # volumn_MxNxN[-5:] = 0

    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
    plt.imshow(front_view / np.max(front_view))
    # path = Path(path)
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + f'/{name}_front_view_{index}.jpg')
    # wandb.log({f"{name}_front_view" : wandb.Image(plt)}, commit=False) TODO

    top_view = np.max(volumn_MxNxN, axis=1)
    plt.imshow(np.rot90(top_view / np.max(top_view), 2))
    plt.savefig(path + f'/{name}_top_view_{index}.jpg')
    # wandb.log({f"{name}_top_view" : wandb.Image(plt)}, commit=False) TODO

    left_view = np.max(volumn_MxNxN, axis=2)
    plt.imshow(np.rot90(left_view / np.max(left_view), 3))
    plt.savefig(path + f'/{name}_left_view_{index}.jpg')
    # wandb.log({f"{name}_left_view" : wandb.Image(plt)}, commit=False) TODO


if __name__ == "__main__":

    run = wandb.init(project='test', name='test_visualizer_1')
    table = wandb.Table(columns=["pred_joints", "MJPJE"])
    
    test_volume_plot = True
    if test_volume_plot:
        input = torch.rand((2, 1, 256, 256, 256))
        res_path = './results'
        name = 'test'
        volume_log(input, res_path, name, 0)
        

    test_joint_plot = False
    if test_joint_plot:
        input = np.loadtxt(
            "/home/liuping/NlosAutoEncoder/meas2pose_voxel_l1_17joint_1219/joint/gt_6400.txt")
        res_path = './results/figure/joints/'
        joint_name = "test"
        joints_log(input, res_path, joint_name)

    out_joints_fig_from_dir = False
    if out_joints_fig_from_dir:
        res_path = Path("joint")
        file_list = []
        for fileName in res_path.iterdir():
            file_list.append(fileName)
            joints = np.loadtxt(fileName)
            joint_name = fileName.stem

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            x_major_locator = MultipleLocator(0.1)
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(x_major_locator)
            ax.zaxis.set_major_locator(x_major_locator)

            joints = (joints / 64.0 - 0.5) * 2.5

            xs = []
            ys = []
            zs = []

            for i in range(joints.shape[0]):
                xs.append(joints[i, 0])
                ys.append(joints[i, 1])
                zs.append(-joints[i, 2])

            fig = plt.figure()
            t = ys
            ys = zs
            zs = t
            # ys = ys
            renderBones(ax, xs, ys, zs)
            ax.scatter(xs, ys, zs)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(joint_name)
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(-0.75, 0.75)
            ax.set_zlim(-0.75, 0.75)
            # ax.invert_zaxis()
            ax.invert_yaxis()
            # ax.invert_xaxis()
            ax.figure.savefig('./results/joint_fig/' + joint_name)
    print("QwQ")
    wandb.finish()
