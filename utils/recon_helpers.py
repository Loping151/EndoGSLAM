import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import numpy as np


def setup_camera(w, h, k, w2c, near=0.01, far=100, bg=[0,0,0], use_simplification=True):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor(bg, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0 if use_simplification else 3,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def calculate_entropy(prob):
    return -np.sum(prob * np.log2(prob + np.finfo(float).eps))

def find_optimal_threshold(gray_image):
    hist, _ = np.histogram(gray_image, bins=np.arange(257), density=True)
    cdf = hist.cumsum()
    max_entropy = -1
    optimal_threshold = 0
    
    for threshold in range(0, 1, 0.05):
        lower_part = hist[:threshold]
        upper_part = hist[threshold:]
        
        lower_prob = lower_part / lower_part.sum()
        upper_prob = upper_part / upper_part.sum()
        
        lower_entropy = calculate_entropy(lower_prob)
        upper_entropy = calculate_entropy(upper_prob)
        
        total_entropy = lower_entropy + upper_entropy
        
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_threshold = threshold
    
    return optimal_threshold

def deg_to_rad(deg):
    return deg * np.pi / 180

def rotation_matrix_x(angle_rad):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

def rotation_matrix_y(angle_rad):
    return np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

def rotation_matrix_z(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

def calculate_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    roll_rad = deg_to_rad(roll_deg)
    pitch_rad = deg_to_rad(pitch_deg)
    yaw_rad = deg_to_rad(yaw_deg)
    
    Rx = rotation_matrix_x(roll_rad)
    Ry = rotation_matrix_y(pitch_rad)
    Rz = rotation_matrix_z(yaw_rad)
    
    R = Rz @ Ry @ Rx
    return R

def energy_mask(color: torch.Tensor, th_1=0.1, th_2=0.9):
    """
    mask out the background(black). set to 0 to mask black only, and other value(0, 1) to filter pixels with certain brightness
    """
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=color.device).view(3, 1, 1)
    gray = torch.sum(color * weights, dim=0).detach() # mask should not have grad
    zero_mask = torch.where((gray >= th_1) & (gray <= th_2), torch.tensor([True], device=color.device), torch.tensor([False], device=color.device))[None]
    # Image.fromarray(np.uint8(zero_mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')

    # return zero_mask
    return torch.ones_like(zero_mask).to(zero_mask.device)
