import os
import numpy as np
import cv2
import shutil
import torch
from typing import Tuple, List
from .config import SEQUENCE_LENGTH, UNLABELED_DIR, DATA_DIR, TEST_DIR, ACTIONS

class DataLoader:
    def __init__(self):
        self.UNLABELED_DIR = UNLABELED_DIR
        self.DATA_DIR = DATA_DIR
        self.TEST_DIR = TEST_DIR
        self.ACTIONS = ACTIONS
        self.width, self.height = 640, 480
        
        print("\nDataLoader initialization:")
        print(f"UNLABELED_DIR: {self.UNLABELED_DIR}")
        print(f"DATA_DIR: {self.DATA_DIR}")
        print(f"TEST_DIR: {self.TEST_DIR}")
        print(f"UNLABELED_DIR exists: {os.path.exists(self.UNLABELED_DIR)}")
        print(f"DATA_DIR exists: {os.path.exists(self.DATA_DIR)}")
        print(f"TEST_DIR exists: {os.path.exists(self.TEST_DIR)}")
        
        # Create directories for each action
        for action in self.ACTIONS:
            action_dir = os.path.join(self.DATA_DIR, action)
            print(f"\nChecking action directory: {action_dir}")
            print(f"Exists: {os.path.exists(action_dir)}")
            os.makedirs(action_dir, exist_ok=True)
    
    def load_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load all labeled data from MP_mixed directory for training"""
        return self._load_data_from_directory(self.DATA_DIR)
    
    def load_evaluation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load test data from MP_testset directory for evaluation"""
        return self._load_data_from_directory(self.TEST_DIR)
    
    def _load_data_from_directory(self, base_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper method to load data from a directory"""
        X_data = []
        y_data = []
        
        for action_idx, action in enumerate(self.ACTIONS):
            action_dir = os.path.join(base_dir, action)
            if not os.path.exists(action_dir):
                continue
                
            # Get all sequence folders in action directory
            sequence_folders = [d for d in os.listdir(action_dir) 
                              if os.path.isdir(os.path.join(action_dir, d))]
            
            for seq_folder in sequence_folders:
                seq_path = os.path.join(action_dir, seq_folder)
                sequence = self._load_sequence(seq_path)
                if sequence is not None:
                    sequence_tensor = torch.from_numpy(sequence).float()
                    X_data.append(sequence_tensor)
                    y_data.append(action_idx)
        
        if not X_data:
            return torch.tensor([]), torch.tensor([])
            
        X_data = torch.stack(X_data)
        y_data = torch.tensor(y_data)
        
        print(f"Loaded data from {base_dir}:")
        print(f"X_data shape: {X_data.shape}")
        print(f"y_data shape: {y_data.shape}")
        
        return X_data, y_data
    
    def load_unlabeled_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load all unlabeled sequences"""
        sequences = []
        sequence_names = []
        
        # print(f"\nLoading unlabeled data from: {self.UNLABELED_DIR}")
        # print(f"Directory exists: {os.path.exists(self.UNLABELED_DIR)}")
        
        if not os.path.exists(self.UNLABELED_DIR):
            print("Unlabeled directory does not exist!")
            return [], []
            
        files = os.listdir(self.UNLABELED_DIR)
        # print(f"Files in unlabeled directory: {files}")
        # print(f"Number of files: {len(files)}")
        
        for seq_name in files:
            seq_path = os.path.join(self.UNLABELED_DIR, seq_name)
            # print(f"\nProcessing sequence: {seq_name}")
            # print(f"Sequence path: {seq_path}")
            # print(f"Is directory: {os.path.isdir(seq_path)}")
            
            if not os.path.isdir(seq_path):
                print(f"Skipping {seq_name} - not a directory")
                continue
                
            # Check contents of the sequence directory
            seq_contents = os.listdir(seq_path)
            # print(f"Contents of sequence directory: {seq_contents}")
            # print(f"Number of files in sequence: {len(seq_contents)}")
            
            sequence = self._load_sequence(seq_path)
            if sequence is not None:
                sequences.append(sequence)
                sequence_names.append(seq_name)
                # print(f"Successfully loaded sequence: {seq_name}")
            else:
                print(f"Failed to load sequence: {seq_name}")
        
        print(f"\nTotal loaded sequences: {len(sequences)}")
        return sequences, sequence_names
    
    def _load_sequence(self, seq_path: str) -> np.ndarray:
        """Load a single sequence from directory"""
        # print(f"\nLoading sequence from: {seq_path}")
        
        frame_files = sorted(
            [f for f in os.listdir(seq_path) if f.endswith('.npy')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        # print(f"Found frame files: {frame_files}")
        # print(f"Number of frame files: {len(frame_files)}")
        
        if not frame_files:
            print("No frame files found!")
            return None
            
        sequence_data = []
        for file in frame_files:
            file_path = os.path.join(seq_path, file)
            # print(f"Loading frame: {file_path}")
            try:
                keypoints = np.load(file_path)
                # print(f"Loaded keypoints shape: {keypoints.shape}")
                sequence_data.append(keypoints)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                return None
        
        # Convert to numpy array with shape (sequence_length, features)
        sequence_data = np.array(sequence_data)
        # print(f"Sequence data shape before padding: {sequence_data.shape}")
        
        # If sequence is longer than SEQUENCE_LENGTH, truncate it
        if len(sequence_data) > SEQUENCE_LENGTH:
            sequence_data = sequence_data[:SEQUENCE_LENGTH]
        # If sequence is shorter than SEQUENCE_LENGTH, pad with zeros
        elif len(sequence_data) < SEQUENCE_LENGTH:
            pad_length = SEQUENCE_LENGTH - len(sequence_data)
            sequence_data = np.pad(sequence_data, ((0, pad_length), (0, 0)), mode='constant')
        
        # print(f"Final sequence data shape: {sequence_data.shape}")
        return sequence_data
    
    def visualize_sequence(self, sequence: np.ndarray, seq_name: str, entropy: float):
        """Visualize a sequence with OpenCV"""
        frames = []
        for keypoints in sequence:
            # Create blank canvas
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Split keypoints into different parts
            pose = keypoints[:132].reshape(33, 4)
            face = keypoints[132:132+468*3].reshape(468, 3)
            lh = keypoints[132+468*3:132+468*3+21*3].reshape(21, 3)
            rh = keypoints[132+468*3+21*3:].reshape(21, 3)
            
            # Draw pose landmarks (green)
            for point in pose:
                x, y, z, v = point
                cx, cy = int(x * self.width), int(y * self.height)
                cv2.circle(canvas, (cx, cy), 3, (0, 255, 0), -1)
            
            # Draw face landmarks (blue)
            for point in face:
                x, y, z = point
                cx, cy = int(x * self.width), int(y * self.height)
                cv2.circle(canvas, (cx, cy), 1, (255, 0, 0), -1)
            
            # Draw hand landmarks (red)
            for point in lh:
                x, y, z = point
                cx, cy = int(x * self.width), int(y * self.height)
                cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)
            for point in rh:
                x, y, z = point
                cx, cy = int(x * self.width), int(y * self.height)
                cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)
            
            frames.append(canvas)
        
        # Display sequence
        print(f"Sequence: {seq_name}, Entropy: {entropy:.4f}")
        for frame in frames:
            cv2.imshow(f"Sequence: {seq_name}, Entropy: {entropy:.4f}", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def move_sequence(self, action: str, seq_name: str):
        """Move a sequence from unlabeled to labeled directory with sequential numbering"""
        source_path = os.path.join(self.UNLABELED_DIR, seq_name)
        if not os.path.exists(source_path):
            print(f"Sequence {seq_name} not found in unlabeled directory")
            return False
        
        # Create target action directory if it doesn't exist
        target_dir = os.path.join(self.DATA_DIR, action)
        os.makedirs(target_dir, exist_ok=True)
        
        # Find the next available sequence number
        existing_folders = [d for d in os.listdir(target_dir) 
                          if os.path.isdir(os.path.join(target_dir, d)) and d.isdigit()]
        if existing_folders:
            next_folder = str(max(map(int, existing_folders)) + 1)
        else:
            next_folder = "0"
        
        # Move sequence to target directory with new sequential number
        target_path = os.path.join(target_dir, next_folder)
        shutil.move(source_path, target_path)
        print(f"Sequence {seq_name} moved to {action} directory as {next_folder}")
        return True
    
    @staticmethod
    def to_tensor(data, device='cpu'):
        """Convert numpy array to PyTorch tensor"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device)
        return data.to(device) 