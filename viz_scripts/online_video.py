import cv2
import os
import numpy as np

def create_video_from_images(image_folder, output_video):
    """
    Create a video from images sorted by file name in a folder, considering numerical order for file names.

    Parameters:
    - image_folder: Path to the folder containing images.
    - output_video: Path to save the output video file.
    """

    # Get all files from the folder and filter for image formats
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")]
    
    # Sort the files by numerical order
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    # Read the first image to determine the video size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    size = (width, height)
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        out.write(frame)  # Write out frame to video
    
    out.release()  # Release the VideoWriter object


if __name__ == "__main__":
    image_folder = './online_vis'
    output_video = 'output_video.mp4'
    create_video_from_images(image_folder, output_video)

