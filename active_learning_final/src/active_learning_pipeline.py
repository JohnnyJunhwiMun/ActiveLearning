import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from active_learning.model import LSTMModel
from active_learning.data_loader import DataLoader
from active_learning.uncertainty_sampling import UncertaintySampler
from active_learning.visualization import Visualizer
from active_learning.config import (INITIAL_MODEL_PATH, BEST_MODEL_PATH, LEARNING_RATE, 
                    NUM_EPOCHS, STOPPING_CRITERIA, ACTIONS, BATCH_SIZE, MIN_LOSS_THRESHOLD)

class ActiveLearningPipeline:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = LSTMModel().to(device)
        self.data_loader = DataLoader()
        self.uncertainty_sampler = UncertaintySampler(self.model, device)
        self.visualizer = Visualizer()
        self.writer = SummaryWriter('Logs')
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, X_train, y_train, num_epochs=NUM_EPOCHS):
        """Train the model for specified number of epochs"""
        self.model.train()
        total_loss = 0
        best_loss = float('inf')
        
        print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / 100
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
                self.writer.add_scalar('Loss/train', avg_loss, epoch)
                
                # Update best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                total_loss = 0
        
        # If loss is still above threshold after NUM_EPOCHS, continue training
        if best_loss > MIN_LOSS_THRESHOLD:
            print(f"\nLoss ({best_loss:.4f}) is still above threshold ({MIN_LOSS_THRESHOLD}).")
            print("Continuing training...")
            
            while best_loss > MIN_LOSS_THRESHOLD:
                for epoch in range(100):  # Additional 100 epochs
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(X_train)
                    loss = self.criterion(outputs, y_train)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if (epoch + 1) % 100 == 0:
                        avg_loss = total_loss / 100
                        print(f'Additional Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}')
                        self.writer.add_scalar('Loss/train', avg_loss, num_epochs + epoch)
                        
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                        
                        total_loss = 0
                
                if best_loss <= MIN_LOSS_THRESHOLD:
                    print(f"\nLoss ({best_loss:.4f}) reached threshold ({MIN_LOSS_THRESHOLD}).")
                    break
    
    def initial_training(self):
        """Initial training with small labeled dataset"""
        print("Starting initial training...")
        
        # Load initial labeled data from MP_mixed
        X_train, y_train = self.data_loader.load_test_data()
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # Move tensors to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Train the model
        self.train(X_train, y_train, num_epochs=NUM_EPOCHS)
        
        # Save initial model
        self.model.save(INITIAL_MODEL_PATH)
        print(f"Initial training completed and model saved to {INITIAL_MODEL_PATH}")
        
        # Evaluate initial model on test set
        print("\nEvaluating initial model on test set...")
        X_test, y_test = self.data_loader.load_evaluation_data()
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        f1, acc, cm = self.evaluate(X_test, y_test)
        print(f"\nInitial Model Performance:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, preds = torch.max(outputs, 1)
            
            y_true = y_test.cpu().numpy()
            y_pred = preds.cpu().numpy()
            
            f1 = f1_score(y_true, y_pred, average='weighted')
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            
            return f1, acc, cm
    
    def active_learning_loop(self):
        """Main active learning loop"""
        # 0. Initial training with small labeled dataset
        self.initial_training()
        
        iteration = 0
        best_f1 = -1  # Initialize with -1 to ensure first model is saved
        performance_history = []
        
        while True:
            print(f"\nActive Learning Iteration {iteration + 1}")
            
            # 1. Load unlabeled data
            unlabeled_data, sequence_names = self.data_loader.load_unlabeled_data()
            if len(unlabeled_data) == 0:
                print("No more unlabeled data available.")
                break
            
            # 2. Get uncertain samples
            uncertain_sequences = self.uncertainty_sampler.get_uncertain_samples(
                unlabeled_data, sequence_names)
            
            # 3. Visualize and label uncertain samples
            for seq_name, entropy in uncertain_sequences:
                print(f"\nSequence: {seq_name}, Entropy: {entropy:.4f}")
                
                # Load and visualize sequence
                sequence = self.data_loader._load_sequence(
                    os.path.join(self.data_loader.UNLABELED_DIR, seq_name))
                if sequence is None:
                    print(f"Failed to load sequence {seq_name}")
                    continue
                    
                self.visualizer.visualize_sequence(sequence, seq_name, entropy)
                
                # Get prediction distribution
                probs = self.uncertainty_sampler.get_prediction_distribution(sequence)
                self.visualizer.show_prediction_distribution(probs, ACTIONS)
                
                # Get user input for labeling
                print("\nSelect action (1: Like, 2: Fire works, 3: Heart, q: Skip):")
                key = input().strip().lower()
                
                if key == 'q':
                    continue
                
                try:
                    action_idx = int(key) - 1
                    if 0 <= action_idx < len(ACTIONS):
                        target_action = ACTIONS[action_idx]
                        self.data_loader.move_sequence(
                            target_action, 
                            seq_name)
                        print(f"Sequence {seq_name} labeled as {target_action}")
                except ValueError:
                    print("Invalid input. Skipping sequence.")
            
            # 4. Retrain model from scratch with all labeled data
            print("\nRetraining model with all labeled data...")
            # Load all labeled data (including newly labeled ones)
            X_train, y_train = self.data_loader.load_test_data()
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            
            # Reinitialize model and optimizer
            self.model = LSTMModel().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
            
            # Train from scratch with all data
            self.train(X_train, y_train, num_epochs=NUM_EPOCHS)
            
            # 5. Evaluate model using MP_testset data
            X_test, y_test = self.data_loader.load_evaluation_data()
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            
            f1, acc, cm = self.evaluate(X_test, y_test)
            
            # Save performance history
            performance_history.append({
                'iteration': iteration + 1,
                'f1_score': f1,
                'accuracy': acc,
                'confusion_matrix': cm
            })
            
            # Print current iteration's performance
            print(f"\nIteration {iteration + 1} Performance:")
            print(f"F1 Score: {f1:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print("Confusion Matrix:")
            print(cm)
            
            # 6. Check stopping criteria
            if f1 >= STOPPING_CRITERIA:
                print(f"\nStopping criteria met (F1 >= {STOPPING_CRITERIA})")
                break
            
            # 7. Save best model
            if f1 > best_f1:
                best_f1 = f1
                try:
                    # Save entire model instead of just state_dict
                    torch.save(self.model, BEST_MODEL_PATH)
                    print(f"New best model saved to {BEST_MODEL_PATH} with F1 score: {best_f1:.4f}")
                except Exception as e:
                    print(f"Error saving best model: {str(e)}")
            
            iteration += 1
            
            # Print performance history after each iteration
            print("\nPerformance History:")
            print("Iteration | F1 Score | Accuracy")
            print("-" * 35)
            for record in performance_history:
                print(f"{record['iteration']:9d} | {record['f1_score']:.4f} | {record['accuracy']:.4f}")
        
        self.writer.close()
        return best_f1, performance_history

if __name__ == "__main__":
    # Initialize pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = ActiveLearningPipeline(device)
    
    # Start active learning
    best_f1, performance_history = pipeline.active_learning_loop()
    print(f"\nActive Learning completed. Best F1 score: {best_f1:.4f}")
    print(f"Initial model saved to: {INITIAL_MODEL_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}") 