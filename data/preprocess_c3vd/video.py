import cv2
import os
import argparse
import concurrent.futures

def makev(image_folder, video_path, fps=30, ext='png'):
    # first_image_path = os.path.join(image_folder, '0000_depth.tiff')
    first_image_path = os.path.join(image_folder, '0000_color.'+ext)
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(0, int(1e10), 1):
        image_path = os.path.join(image_folder, f'{i:04d}_color.'+ext)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            video.write(img)
        else:
            print(f"File {image_path} not found. Assuming end of video.")
            break
    video.release()
    cv2.destroyAllWindows()

def process_folders(root_folder):
    """
    Processes each subfolder within the root folder, applying file classification.
    Skips folders that do not contain files starting with a digit, assuming they've been processed.
    """
    if not os.path.exists('video'):
        os.makedirs('video')
    all_folders = []
    with open ('scene.txt', 'w') as f:
        for folder_name in os.listdir(root_folder):
            if folder_name == 'video':
                continue
            folder_path = os.path.join(root_folder, folder_name)
            if os.path.isdir(folder_path):
                f.write(f"{folder_path.split('./')[-1]}, ")
                all_folders.append(folder_path)
    # NOTE: set max_workers to the number of available logical CPU cores(for example 24 for 13700K)
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        for folder_path in all_folders:
            executor.submit(makev, folder_path+'/color', './video/'+folder_path+'.avi')
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make videos of all datasets")
    parser.add_argument("--root_path", default=".", help="Path to the root folder of the dataset.")
    args = parser.parse_args()
    process_folders(args.root_path)