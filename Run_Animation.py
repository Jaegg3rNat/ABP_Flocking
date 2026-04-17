import cv2
import os
import numpy as np


def images_to_video(image_folder, output_video_path, fps):
    """
    Convert a sequence of images from a folder into a video file.

    Parameters:
    image_folder (str): Path to the folder containing the images.
    output_video_path (str): Path to the output video file.
    fps (int): Frames per second for the output video.
    """
    # List all image files in the folder with appropriate extensions
    print('Accessing images from folder')
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure images are sorted to maintain the correct order in the video
    images.sort()
    print(f'Creating video Output ...... %...')
    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # Resize the frame if it has different dimensions than the first image
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        # Write the frame to the video file
        video.write(frame)

    # Release the video writer and close any open windows
    cv2.destroyAllWindows()
    video.release()

import os
import ffmpeg
import tempfile

def images_to_video2(image_folder, output_video_path, fps, crf=23, resize=None):
    print("Accessing images from folder...")

    images = sorted(
        img for img in os.listdir(image_folder)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    if not images:
        raise RuntimeError("No images found in the folder.")

    # Create concat list with ABSOLUTE paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        list_file = f.name
        for img in images:
            img_path = os.path.abspath(os.path.join(image_folder, img))
            f.write(f"file '{img_path}'\n")

    stream = ffmpeg.input(
        list_file,
        format='concat',
        safe=0,
        r=fps
    )
    print(os.path.abspath(image_folder_path))
    print(os.listdir(image_folder_path)[:5])

    if resize is not None:
        width, height = resize
        stream = stream.filter('scale', width, height)

    stream = ffmpeg.output(
        stream,
        output_video_path,
        vcodec='libx264',
        pix_fmt='yuv420p',
        crf=crf,
        preset='slow',
        movflags='+faststart'
    )

    ffmpeg.run(stream, overwrite_output=True)
    os.remove(list_file)

    print("Compressed video created successfully.")


if __name__ == "__main__":
    # Define base folder and velocity field name
    v0 = 2e-4
    diff = 2e-6
    Pe = 8
    Mu = 380
    base_folder = f"Data/gif_frames_V0{v0:.4e}_DT{diff:.4e}/"  # Replace with the actual base folder path
    name = f'v0_{v0:.1e}Dt_{diff:.1e}_phase'  # Replace with the actual movie name

    # base_folder = f"Code/run_data_2d/_PE{Pe:.2f}_MU{Mu:.2f}/"  # Replace with the actual base folder path
    # name = f'{Pe}_{Mu}_2d_phase'  # Replace with the actual movie name
    # Define parameters for the video creation


    # Construct the paths for the image folder and the output video file
    image_folder_path = f'{base_folder}'
    output_video_path = f'Data/{name}_video.mov'

    # Set the desired frames per second (fps) for the output video
    fps = 60

    # Call the function to convert images to video
    # images_to_video(image_folder_path, output_video_path, fps)
    images_to_video2(image_folder_path, output_video_path, fps,crf =22, resize=(1280, 720))