import numpy as np
from PIL import Image
from collections import deque

class AtariPreprocessor:
    """
    Preprocesses Atari frames for RL algorithms.
    Converts RGB to grayscale, resizes to 84x84, and stacks 4 frames.
    Outputs a numpy array of shape (4, 84, 84) in uint8 to save memory.
    """
    def __init__(self, target_size=(84, 84), stack_size=4):
        self.target_width, self.target_height = target_size
        self.stack_size = stack_size
        # Using deque for efficient rolling window of frames
        self.frames = deque(maxlen=self.stack_size)
        
    def _process_frame(self, frame):
        """
        Convert a raw (210, 160, 3) RGB frame to (84, 84) grayscale.
        Uses the ITU-R 601-2 luma transform for grayscale.
        """
        assert frame.shape == (210, 160, 3), f"Expected (210, 160, 3) frame, got {frame.shape}"
        
        # Convert RGB to grayscale using standard luminance weights
        grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        
        # Resize using Pillow with bilinear interpolation
        img = Image.fromarray(grayscale)
        resized_img = img.resize((self.target_width, self.target_height), Image.BILINEAR)
        
        return np.array(resized_img, dtype=np.uint8)

    def reset(self, initial_observation):
        """
        Called at the start of a new episode.
        Initializes the frame stack by repeating the first processed frame 4 times.
        """
        processed_frame = self._process_frame(initial_observation)
        for _ in range(self.stack_size):
            self.frames.append(processed_frame)
            
        return self._get_stacked_state()
        
    def step(self, next_observation):
        """
        Called every time a new observation is received.
        Processes the new frame, adds it to the stack (removing the oldest automatically via deque).
        """
        processed_frame = self._process_frame(next_observation)
        self.frames.append(processed_frame)
        
        return self._get_stacked_state()

    def _get_stacked_state(self):
        """
        Returns the stacked frames as a single numpy array.
        Uses (C, H, W) format which is native to PyTorch.
        Returns uint8 to save memory.
        """
        # np.stack creates shape (4, 84, 84)
        return np.stack(self.frames, axis=0)

