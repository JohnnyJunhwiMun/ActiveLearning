import torch
import numpy as np
from .config import UNCERTAINTY_SAMPLING_SIZE
from typing import List, Tuple
from scipy.stats import entropy

class UncertaintySampler:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def calculate_entropy(self, probs):
        """Calculate entropy for probability distribution"""
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    
    def get_uncertain_samples(self, 
                            sequences: List[np.ndarray], 
                            sequence_names: List[str], 
                            num_samples: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most uncertain samples from unlabeled data
        
        Args:
            sequences: List of unlabeled sequences
            sequence_names: List of sequence names
            num_samples: Number of uncertain samples to return
            
        Returns:
            List of tuples containing (sequence_name, entropy_value)
        """
        if not sequences:
            return []
            
        # Convert sequences to tensor
        sequences_tensor = torch.stack([torch.from_numpy(seq).float() for seq in sequences])
        sequences_tensor = sequences_tensor.to(self.device)
        
        # Get prediction probabilities
        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            probs = torch.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
        
        # Calculate entropy for each sequence
        entropies = []
        for prob in probs_np:
            ent = entropy(prob)
            entropies.append(ent)
        
        # Get indices of top uncertain samples
        top_indices = np.argsort(entropies)[-num_samples:][::-1]
        
        # Return sequence names and their entropy values
        return [(sequence_names[idx], entropies[idx]) for idx in top_indices]
    
    def get_prediction_distribution(self, sequence: np.ndarray) -> np.ndarray:
        """
        Get prediction probability distribution for a single sequence
        
        Args:
            sequence: Input sequence
            
        Returns:
            Probability distribution over classes
        """
        # Convert sequence to tensor
        sequence_tensor = torch.from_numpy(sequence).float().unsqueeze(0)
        sequence_tensor = sequence_tensor.to(self.device)
        
        # Get prediction probabilities
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()[0]
        
        return probs_np 