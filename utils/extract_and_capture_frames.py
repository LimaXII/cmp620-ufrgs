import cv2
import os
import subprocess

def download_video(url, output_path='Vídeos\\video.mp4'):
    try:
        # Use yt-dlp to download the video
        subprocess.run(['yt-dlp', '-o', output_path, url], check=True)
        print(f"Video downloaded as {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def extract_frames(video_path, num_frames=1000, output_folder='C:\\Users\\thiag\\Downloads\\Skyrim'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    count = 0
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"skyrim_({count}).png")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1
        count += 1
        if frame_id >= num_frames:
            break

    cap.release()
    print(f"Extracted {frame_id} frames to {output_folder}")

# Exemplo de uso
video_url = 'https://youtu.be/cVlAX7TBI3k'
# download_video(video_url)
extract_frames('Vídeos\\video.mp4')
