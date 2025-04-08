import cv2
import numpy as np
from .config import WINDOW_WIDTH, WINDOW_HEIGHT, LANDMARK_COLORS

class Visualizer:
    def __init__(self):
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
    
    def draw_landmarks(self, keypoints):
        """Draw landmarks on a blank canvas"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Split keypoints into different parts
        # Pose: 33 landmarks * 4 values = 132
        # Face: 468 landmarks * 3 values = 1404
        # Left hand: 21 landmarks * 3 values = 63
        # Right hand: 21 landmarks * 3 values = 63
        pose = keypoints[:132].reshape(33, 4)
        face = keypoints[132:132+468*3].reshape(468, 3)
        lh = keypoints[132+468*3:132+468*3+21*3].reshape(21, 3)
        rh = keypoints[132+468*3+21*3:].reshape(21, 3)
        
        # Draw pose landmarks (green)
        for point in pose:
            x, y, z, v = point
            cx, cy = int(x * self.width), int(y * self.height)
            cv2.circle(canvas, (cx, cy), 3, LANDMARK_COLORS['pose'], -1)
        
        # Draw face landmarks (blue)
        for point in face:
            x, y, z = point
            cx, cy = int(x * self.width), int(y * self.height)
            cv2.circle(canvas, (cx, cy), 1, LANDMARK_COLORS['face'], -1)
        
        # Draw hand landmarks (red)
        for point in lh:
            x, y, z = point
            cx, cy = int(x * self.width), int(y * self.height)
            cv2.circle(canvas, (cx, cy), 3, LANDMARK_COLORS['hand'], -1)
        
        for point in rh:
            x, y, z = point
            cx, cy = int(x * self.width), int(y * self.height)
            cv2.circle(canvas, (cx, cy), 3, LANDMARK_COLORS['hand'], -1)
        
        return canvas
    
    def visualize_sequence(self, sequence, sequence_name, entropy=None):
        """Visualize a sequence of frames"""
        frames = []
        for frame in sequence:
            canvas = self.draw_landmarks(frame)
            frames.append(canvas)
        
        # Display sequence
        window_name = f"Sequence: {sequence_name}"
        if entropy is not None:
            window_name += f", Entropy: {entropy:.4f}"
        
        for frame in frames:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def show_prediction_distribution(self, probs, actions):
        """Show prediction distribution as a bar chart"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.bar(actions, probs[0])
        plt.title('Prediction Distribution')
        plt.xlabel('Actions')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.show() 