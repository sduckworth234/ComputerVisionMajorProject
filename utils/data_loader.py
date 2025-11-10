import numpy as np
import cv2
import os


class DataLoader:
    # Load frames from MP4 video or PNG image sequence
    def __init__(self, data_source, data_path, single_frame=False, frame_index=0):

        self.data_source = data_source
        self.data_path = data_path
        self.single_frame = single_frame
        self.frame_index = frame_index

    def load_data(self):
        # Load data based on source type (mp4 or png)
        if self.data_source == 'mp4':
            if self.single_frame:
                return self._load_single_from_mp4()
            return self._load_all_from_mp4()
        elif self.data_source == 'png':
            if self.single_frame:
                return self._load_single_from_png()
            return self._load_all_from_png()
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

    def _load_all_from_mp4(self):
        # Load all frames from MP4 video
        frames = []
        cap = cv2.VideoCapture(self.data_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            print(f"[INFO] Loaded frame {i + 1}/{frame_count}")

        cap.release()
        return np.array(frames)

    def _load_single_from_mp4(self):
        # Load single frame from MP4 at specified index
        cap = cv2.VideoCapture(self.data_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {self.frame_index} from {self.data_path}")
        print(f"[INFO] Loaded single frame {self.frame_index}")
        return frame

    def _load_all_from_png(self):
        # Load all PNG images from directory
        image_files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.png')])
        frames = []

        for i, filename in enumerate(image_files):
            img = cv2.imread(os.path.join(self.data_path, filename), cv2.IMREAD_COLOR)
            frames.append(img)
            print(f"[INFO] Loaded {filename} ({i + 1}/{len(image_files)})")

        return np.array(frames)

    def _load_single_from_png(self):
        # Load single PNG image at specified index
        image_files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.png')])
        if self.frame_index >= len(image_files):
            raise IndexError(f"Frame index {self.frame_index} out of range (found {len(image_files)} images)")

        filename = image_files[self.frame_index]
        img = cv2.imread(os.path.join(self.data_path, filename), cv2.IMREAD_COLOR)
        print(f"[INFO] Loaded single image: {filename}")
        return img

    def convert_to_grayscale(self, frames):
        # Convert BGR frames to grayscale
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            return cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        return np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
