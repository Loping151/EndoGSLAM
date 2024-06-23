import cv2
import os
import glob


def plot_video(image_folder: str, video_path: str = '', total: int = int(1e10), fps: int = -1):
    if video_path == '':
        video_path = './plot.avi'
    elif not video_path.endswith('.avi'):
        video_path += '.avi'
    print(f"Saving video to {video_path}")
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    if fps == -1:
        fps = 30

    all_image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    
    first_image_path = all_image_paths[0]
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    img = cv2.imread(first_image_path)
    video.write(img)
    for i in range(1, len(all_image_paths)):
        image_path = all_image_paths[i]
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            video.write(img)
        else:
            print(f"File {image_path} not found. Assuming end of sequence.")
            break
    video.release()
    cv2.destroyAllWindows()

