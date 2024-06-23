import torch
import lpips
import os
import glob
import cv2
import numpy as np
from pytorch_msssim import ms_ssim
from PIL import Image

lpips_model = lpips.LPIPS(net='alex')
    
def lsFile(folder_path, ext='png'):
    """
    bash like ls command to list files in a folder with a specific extension.
    """
    search_pattern = os.path.join(folder_path, '*.'+ext)
    files = glob.glob(search_pattern)
    sorted_files = sorted(files)
    return sorted_files


def read_pose_file(pose_file):
    """
    Reads a pose file and extracts camera poses. Each pose is expected to be in a comma-separated format, representing a 4x4 transformation matrix.

    Args:
        pose_file: The file path to the pose file. The file should contain lines, each line representing a camera pose as 16 comma-separated floats that can be reshaped into a 4x4 matrix.

    Returns:
        poses: A list of 4x4 numpy arrays, each array representing a camera pose as extracted from the file.
    """
    
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        poses = [np.array([float(x) for x in line.split(',')]).reshape(4, 4) for line in lines]
        if poses[-1][:3, 3].sum() == 0:
            poses = [pose.T for pose in poses]
            # for i in range(len(poses)):
            #     poses[i][:2, 3] *= -1 # this is for the niceslam coord
    return poses


def calculate_psnr(img1, img2):
    """Calculates the PSNR between two images.

    Args:
        img1: The first image: ndarray.
        img2: The second image: ndarray.

    Returns:
        The PSNR between the two images.
    """

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Avoid division by zero
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def calculate_ssim(img1, img2):
    """
    Calculates the MS-SSIM between two images using PyTorch.

    Args:
        img1: The first image: ndarray.
        img2: The second image: ndarray.

    Returns:
        The MS-SSIM between the two images.
    """
    
    if np.max(img1) > 1:
        img1 = img1/255.0
        img2 = img2/255.0
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    ms_ssim_value = ms_ssim(img1_tensor, img2_tensor, data_range=1.0)
    return ms_ssim_value



def calculate_lpips(img1, img2):
    """Calculates the LPIPS between two images.

    Args:
        img1: The first image: ndarray.
        img2: The second image: ndarray.

    Returns:
        The LPIPS between the two images.
    """
    
    if np.max(img1) > 1:
        img1 = img1/255.0
        img2 = img2/255.0
        
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        lpips_distance = lpips_model(img1_tensor, img2_tensor)
    return lpips_distance.item()


def calculate_depth_rmse(depth1, depth2):
    """
    Calculates the RMSE (Root Mean Square Error) between two depth maps.

    Args:
        depth1: The first depth map as a NumPy array.
        depth2: The second depth map as a NumPy array.

    Returns:
        The RMSE value between the two depth maps.
    """
    
    mse = np.mean((depth1 - depth2) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)
    """
    
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return model_aligned, trans_error


def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = np.array(gt_traj_pts).T
    est_traj_pts = np.array(est_traj_pts).T

    gt_aligned, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error, gt_aligned, est_traj_pts
        
        
def rgb_metrics(gt, render):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR), Multi-Scale Structural Similarity Index (MS-SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) metrics for RGB images between the ground truth and rendered images.

    Args:
        gt: The base directory containing 'color' subfolder for ground truth RGB images.
        render: The comparison directory containing 'color' subfolder for rendered RGB images.

    Returns:
        mean_psnr: The mean PSNR value across all compared RGB image pairs.
        mean_ssim: The mean MS-SSIM value across all compared RGB image pairs.
        mean_lpips: The mean LPIPS value across all compared RGB image pairs.
        psnr_list: A list of PSNR values for each pair of compared RGB images.
        ssim_list: A list of MS-SSIM values for each pair of compared RGB images.
        lpips_list: A list of LPIPS values for each pair of compared RGB images.
    """

    color_gt = os.path.join(gt, 'color')
    color_render = os.path.join(render, 'color')
    color_files1 = lsFile(color_gt)[7::8]
    color_files2 = lsFile(color_render)
    if '0000' in color_files2[0]:
        color_files2 = color_files2[7::8]
    color1 = [cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) for image_path in color_files1]
    color2 = [cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) for image_path in color_files2]
    color1 = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in color1]
    color2 = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in color2]
    
    psnr_list = [calculate_psnr(color1[i], color2[i]) for i in range(len(color1))]
    ssim_list = [calculate_ssim(color1[i], color2[i]) for i in range(len(color1))]
    lpips_list = [calculate_lpips(color1[i], color2[i]) for i in range(len(color1))]
    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    mean_lpips = np.mean(lpips_list)  

    return mean_psnr, mean_ssim, mean_lpips, psnr_list, ssim_list, lpips_list
    
def depth_metrics(gt, render):
    """
    Calculates the Root Mean Square Error (RMSE) for depth maps between the ground truth and rendered images.

    Args:
        gt: The base directory containing 'depth' subfolder for ground truth depth maps.
        render: The comparison directory containing 'depth' subfolder for rendered depth maps.

    Returns:
        mean_rmse: The mean RMSE value across all compared depth maps.
        rmse_list: A list of RMSE values for each pair of compared depth maps.
    """

    depth_gt = os.path.join(gt, 'depth')
    depth_render = os.path.join(render, 'depth')
    depth_files1 = lsFile(depth_gt, 'tiff')[7::8]
    depth_files2 = lsFile(depth_render, 'tiff')
    if '0000' in depth_files2[0]:
        depth_files2 = depth_files2[7::8]
    depth1 = [np.array(Image.open(image_path)).astype(np.uint16) / 2.55 for image_path in depth_files1]
    depth2 = [np.array(Image.open(image_path)).astype(np.uint16) / 655.35 for image_path in depth_files2]
    depth2 = [np.clip(image, 0, 100) for image in depth2]
    
    rmse_list = [calculate_depth_rmse(depth1[i], depth2[i]) for i in range(len(depth1))]
    mean_rmse = np.mean(rmse_list)
    
    return mean_rmse, rmse_list

def pose_metrics(gt_w2c_path, est_w2c_path, align_gt_path=None):
    """
    Calculates the Average Trajectory Error (ATE) between ground truth and estimated camera poses.

    Args:
        gt_w2c_path: File path to the ground truth camera poses. Each pose is expected to be in the format of a 4x4 matrix.
        est_w2c_path: File path to the estimated camera poses. Format is the same as for ground truth poses.

    Returns:
        ate: The average translational error (ATE) between the ground truth and estimated poses.
    """
    
    gt_w2c = read_pose_file(gt_w2c_path)
    est_w2c = read_pose_file(est_w2c_path)
    
    if align_gt_path is not None:
        gt_w2c = read_pose_file(align_gt_path)
        
    ate, est, gt = evaluate_ate(est_w2c, gt_w2c)
    
    return ate, np.array(gt), np.array(est)