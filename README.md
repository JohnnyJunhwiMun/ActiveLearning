# Active Learning / LSTM / Facetime gesture

This project implements an active learning pipeline for action recognition using LSTM models. The system focuses on recognizing three specific actions: 'Like', 'Fire works', and 'Heart'.
<div align="center">
<img src="https://github.com/user-attachments/assets/05744e50-9622-49d2-9ae4-9d77374623cf" width="600" height="300"/>
</div>


## Project Structure

```
active_learning/
├── src/
│   ├── __init__.py
│   ├── config.py                          # Configuration settings
│   ├── model.py                           # LSTM model definition
│   ├── data_loader.py                     # Data loading and preprocessing
│   ├── uncertainty_sampling.py            # Uncertainty sampling strategies
│   ├── visualization.py                   # Visualization utilities
│   └── active_learning_pipeline.py        # Main pipeline implementation
└── models/
    ├── initial_model.h5                   # Initial trained model
    └── best_model.h5                      # Best performing model

```

## Active Learning Process

1. **Initial Training**  
   Train an LSTM-based action recognition model using a small labeled dataset.

2. **Query Strategy**  
   Use uncertainty sampling to identify the top *k* most uncertain samples from the unlabeled data (based on entropy).

3. **Interactive Labeling**  
   Visualize the selected samples in real time and label them interactively.

4. **Model Retraining**  
   Retrain the model using the updated dataset:  
   `original labeled data + newly labeled uncertain samples`.

5. **Evaluation & Iteration**  
   Evaluate performance (e.g., using the F1 score).  
   Repeat the process until the model reaches a predefined performance threshold (e.g., F1 > 0.9).

## Acknowledgments

- MediaPipe for pose estimation
- PyTorch for deep learning framework 
