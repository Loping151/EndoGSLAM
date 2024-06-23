import os
import numpy as np
import imageio.v2 as imageio
import cv2
from PIL import Image
import shutil
import argparse
from tqdm import tqdm
import concurrent.futures


class ImageProcessor:
    def __init__(self, calibration_params):
        self.calibration_params = calibration_params

    def undisort(self, color: np.ndarray, depth: np.ndarray):
        K = np.array([[self.calibration_params['fx'], 0, self.calibration_params['cx']],
                      [0, self.calibration_params['fy'], self.calibration_params['cy']],
                      [0, 0, 1]])
        D = np.array([self.calibration_params['k1'], self.calibration_params['k2'], 0, 0])
        color = cv2.undistort(color, K, D)
        depth = cv2.undistort(depth, K, D)

        return color, depth

def process_file(folder_path, filename, processor):
    file_path = os.path.join(folder_path, filename)
    image = np.asarray(imageio.imread(file_path), dtype=float)
    
    depth_folder_path = folder_path.replace('color_raw', 'depth_raw')
    depth_file_path = os.path.join(depth_folder_path, filename.replace('color', 'depth').replace('png', 'tiff'))
    depth = np.asarray(Image.open(depth_file_path).convert("I"), dtype=np.float64)
    
    processed_color, processed_depth = processor.undisort(image, depth)
    os.makedirs(folder_path.replace('color_raw', 'color'), exist_ok=True)
    os.makedirs(depth_folder_path.replace('depth_raw', 'depth'), exist_ok=True)
    Image.fromarray(np.uint8(processed_color), 'RGB').save(file_path.replace('color_raw', 'color'))
    Image.fromarray(np.uint8(processed_depth/65535*255), 'L').save(depth_file_path.replace('depth_raw', 'depth'))
    
def process_image(folder_path, processor):
    folder_path = os.path.join(folder_path, 'color_raw')
    file_paths = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    # NOTE: set max_workers to the number of available logical CPU cores(for example 24 for 13700K)
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        tasks = [executor.submit(process_file, folder_path, filename, processor) for filename in file_paths]
        for _ in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            pass
        
def process_folders(root_folder, calibration_params):
    processor = ImageProcessor(calibration_params)
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for sub_folder_name in ['color', 'depth']:
                sub_folder_path = os.path.join(folder_path, sub_folder_name)
                raw_path = sub_folder_path + '_raw'
                if not os.path.exists(raw_path):
                    print(f"Moving {sub_folder_path} to {raw_path}")
                    shutil.move(sub_folder_path, raw_path)
                else:
                    print(f"Skipping already processed folder: {folder_path}")
            process_image(folder_path, processor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and undisort color and depth images in dataset folders.")
    parser.add_argument("--root_path", default=".", help="Path to the root folder of the dataset.")
    
    args = parser.parse_args()

    camera_params = {
        "fx": 802.319,
        "fy": 801.885,
        "cx": 668.286,
        "cy": 547.733,
        "k1": -0.42234,
        "k2": 0.10654
    }
    process_folders(args.root_path, camera_params)
