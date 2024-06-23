import os
import sys
import argparse
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
from utils.metrics_helper import rgb_metrics, depth_metrics, pose_metrics
    
def metric_single(gt, render):
    """Calculates the PSNR between color and depth folders of images.

    Args:
        gt: The base folder containing 'color' and 'depth' subfolders.
        render: The comparison folder containing 'color' and 'depth' subfolders.

    Returns:
        None.
    """

    if os.path.exists(os.path.join(render, 'eval')):
        render = os.path.join(render, 'eval')

    psnr, ssim, lpips, pl, sl, ll = rgb_metrics(gt, render)
    rmse, rl = depth_metrics(gt, render)
    gp_path = os.path.join(render, 'gt_train_w2c.txt')
    ep_path = os.path.join(render, 'est_w2c.txt')
    # align_gt = 'cmp/Fusion/results/{}/eval/gt_train_w2c.txt'.format(render.split('/')[-2])
    ate, gt_pts, est_pts = pose_metrics(gp_path, ep_path, None)
    
    return {'rgb': [psnr, ssim, lpips], 'depth': [rmse], 'pose': [ate], 'lists': [pl, sl, ll, rl], 'traj': [gt_pts, est_pts]}


def plot_sequences(seq1, seq2, seq3, seq4, labels, save_path):
    """
    Plots four sequences in a single line plot and saves the plot to a specified path.

    Args:
        seq1, seq2, seq3, seq4: Lists or arrays containing the sequences to be plotted.
        labels: A list of strings containing the labels for each sequence.
        save_path: The file path where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(seq1, label=labels[0])
    plt.plot(seq2, label=labels[1])
    plt.plot(seq3, label=labels[2])
    plt.plot(seq4, label=labels[3])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Sequence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_3d_trajectories(gt, est, ate, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt[0], gt[1], gt[2], color='blue', label='GT')
    ax.plot(est[0], est[1], est[2], color='red', label='Estimate')
    
    ax.legend()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'ATE: {ate:.3f}')
    plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", help="The base folder containing 'color' and 'depth' subfolders.")
    parser.add_argument("--render", help="The comparison folder containing 'color' and 'depth' subfolders.")
    parser.add_argument("--test_single", help="Test single metric function.", action='store_true')
    parser.add_argument("--name", help="Name of record folder", default='')
    args = parser.parse_args()


    if args.test_single:
        metrics = metric_single(args.gt, args.render)
        print("PSNR: ", metrics['rgb'][0])
        print("SSIM: ", metrics['rgb'][1])
        print("LPIPS: ", metrics['rgb'][2])
        print("RMSE: ", metrics['depth'][0])
        print("ATE: ", metrics['pose'][0])
        plot_sequences(np.array(metrics['lists'][0])/10, metrics['lists'][1], metrics['lists'][2], metrics['lists'][3], ['PSNR/10', 'SSIM', 'LPIPS', 'RMSE'], 'metrics.png')
        plot_3d_trajectories(metrics['traj'][0], metrics['traj'][1], metrics['pose'][0], 'traj.png')
    else:
        record_dir = './records/' + time.strftime("%Y-%m-%d-%H-%M-%S ", time.localtime()) + args.name
        os.makedirs(record_dir, exist_ok=True)
        
        results = {"PSNR": [], "SSIM": [], "LPIPS": [], "RMSE": [], "ATE": []}
        seqs = os.listdir(args.render)
        if os.path.exists(os.path.join(args.render, seqs[0], 'config.py')):
            shutil.copy(os.path.join(args.render, seqs[0], 'config.py'), os.path.join(record_dir, 'config.py'))
        for seq in seqs:
            if not os.path.isdir(os.path.join(args.render, seq)):
                continue
            print("Processing: ", seq)
            seq_dir = os.path.join(args.render, seq)
            seq_gt = os.path.join(args.gt, seq)
            metrics = metric_single(seq_gt, seq_dir)
            if os.path.exists(os.path.join(args.render, seq, 'keyframes.avi')):
                shutil.copy(os.path.join(args.render, seq, 'keyframes.avi'), os.path.join(record_dir, f'{seq}.avi'))
            if os.path.exists(os.path.join(args.render, seq, 'runtimes.txt')):
                shutil.copy(os.path.join(args.render, seq, 'runtimes.txt'), os.path.join(record_dir, f'{seq}.txt'))
            plot_sequences(np.array(metrics['lists'][0])/10, metrics['lists'][1], metrics['lists'][2], metrics['lists'][3], ['PSNR/10', 'SSIM', 'LPIPS', 'RMSE'], os.path.join(record_dir, f'{seq}_metrics.png'))
            plot_3d_trajectories(metrics['traj'][0], metrics['traj'][1], metrics['pose'][0], os.path.join(record_dir, f'{seq}_traj.png'))

            results["PSNR"].append(metrics['rgb'][0])
            results["SSIM"].append(metrics['rgb'][1])
            results["LPIPS"].append(metrics['rgb'][2])
            results["RMSE"].append(metrics['depth'][0])
            results["ATE"].append(metrics['pose'][0])

        with open(os.path.join(record_dir, 'results.csv'), 'w') as f:
            f.write('Sequence,PSNR,SSIM,LPIPS,RMSE,ATE\n')
            for i, seq in enumerate(seqs):
                f.write(f"{seq},{results['PSNR'][i]},{results['SSIM'][i]},{results['LPIPS'][i]},{results['RMSE'][i]},{results['ATE'][i]}\n")
            f.write(f"Mean,{np.mean(results['PSNR'])},{np.mean(results['SSIM'])},{np.mean(results['LPIPS'])},{np.mean(results['RMSE'])},{np.mean(results['ATE'])}\n")
            f.write(f"Std,{np.std(results['PSNR'])},{np.std(results['SSIM'])},{np.std(results['LPIPS'])},{np.std(results['RMSE'])},{np.std(results['ATE'])}\n")
            
            print("Results saved to: ", os.path.join(record_dir, 'results.csv'))
