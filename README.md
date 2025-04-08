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

## Features

- LSTM-based action recognition model
- Active learning pipeline with uncertainty sampling
- Interactive labeling interface
- Real-time visualization of action sequences
- Performance monitoring and model evaluation

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- TensorBoard

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project can be configured through `config.py`:

- Model parameters (input dimension, number of classes)
- Training settings (learning rate, epochs, batch size)
- Active learning parameters (uncertainty sampling size, stopping criteria)
- Data paths and visualization settings



## Active Learning Process

1. Initial training with a small labeled dataset
2. Query Strategy: Uncertainty sampling from unlabeled data using top k number of entropy
3. Interactive labeling of uncertain samples
4. Model retraining with newly labeled data (original small labeled dataset + labeled top k uncertain samples)
5. Performance evaluation and iteration : untill F1 score > threshold (e.g. 0.9)


## Acknowledgments

- MediaPipe for pose estimation
- PyTorch for deep learning framework 
