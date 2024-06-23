import os
import numpy as np
import imageio.v2 as imageio
import cv2
from PIL import Image
import shutil
import argparse
from tqdm import tqdm
import concurrent.futures


def resize_image(path, target_width, target_height):
    color_folder_path = os.path.join(path, 'color')
    file_paths = [f for f in os.listdir(color_folder_path) if f.endswith(".png")]
    for file_path in file_paths:
        image = np.asarray(imageio.imread(os.path.join(color_folder_path, file_path)), dtype=float)
        if image.shape[0] == target_height and image.shape[1] == target_width:
            print(f"Skipping {file_path}")
            continue
        image = cv2.resize(image, (target_width, target_height))
        Image.fromarray(np.uint8(image), 'RGB').save(os.path.join(color_folder_path, file_path))
    depth_folder_path = os.path.join(path, 'depth')
    depth_file_paths = [f for f in os.listdir(depth_folder_path) if f.endswith(".tiff")]
    for depth_file_path in depth_file_paths:
        depth = np.asarray(Image.open(os.path.join(depth_folder_path, depth_file_path)).convert("I"), dtype=np.float64)
        if depth.shape[0] == target_height and depth.shape[1] == target_width:
            print(f"Skipping {depth_file_path}")
            continue
        depth = cv2.resize(depth, (target_width, target_height))
        Image.fromarray(np.uint8(depth), 'L').save(os.path.join(depth_folder_path, depth_file_path))



def process_folders(root_folder, target_width, target_height):
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)
            if os.path.isdir(folder_path) and "video" not in folder_path:    
                tasks.append(executor.submit(resize_image, folder_path, target_width, target_height))
        for _ in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and undisort color and depth images in dataset folders.")
    parser.add_argument("--root_path", default=".", help="Path to the root folder of the dataset.")
    parser.add_argument("--target_width", default=1350//2, type=int, help="Target width for the images.")
    parser.add_argument("--target_height", default=1080//2, type=int, help="Target height for the images.")
    args = parser.parse_args()
    process_folders(args.root_path, args.target_width, args.target_height)