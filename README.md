# Active Learning / LSTM / Facetime gesture

In machine learning, the labeling process is both costly and time-consuming. To address this challenge, an active learning technique has been integrated into an LSTM model. Active learning is particularly effective in environments with limited labeled data, as it allows the model to selectively query uncertain data points for labeling. Rather than relying on vast amounts of labeled data for training, this approach achieves optimal performance with a minimal labeling process and high quality data selection.

This project implements an active learning pipeline for action recognition using LSTM models. The system focuses on recognizing three specific actions: "Like," "Fire works," and "Heart."
<div align="center">
  <img src="https://github.com/user-attachments/assets/05744e50-9622-49d2-9ae4-9d77374623cf" alt="Facetime gesture demonstration" width="600" height="300"/>
</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Active Learning Process](#active-learning-process)
  - [Step 1: Initial Training](#step-1-initial-training)
  - [Step 2: Query Strategy](#step-2-query-strategy)
  - [Step 3: Interactive Labeling](#step-3-interactive-labeling)
  - [Step 4: Model Retraining](#step-4-model-retraining)
  - [Step 5: Evaluation and Iteration](#step-5-evaluation-and-iteration)
- [Performance History](#performance-history)
- [Comparison of Performance](#comparison-of-performance)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The project implements an **active learning pipeline** for action recognition using LSTM models. Instead of requiring extensive labeled datasets, this approach identifies and labels the most uncertain samples to enhance model performance efficiently. The recognized gestures include:

- **Like**
- **Fire works**
- **Heart**

---


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
---
## Active Learning Process

### Step 1: Initial Training
   Train an LSTM-based action recognition model using a small labeled dataset.

### Step 2: Query Strategy
Use uncertainty sampling to identify the top *k* most uncertain samples from the unlabeled data (based on entropy). This guarantees high-quality data for retraining, as it selects the most challenging samples for the model to differentiate between each action.


### Step 3: Interactive Labeling
   Visualize the selected samples in real time and label them interactively.

   ![Image](https://github.com/user-attachments/assets/c4527373-55e7-4a7e-b1f2-9bd52d015372)
    
    
  #### Example in Terminal
    
  ```bash
  
  Select action (1: Like, 2: Fire works, 3: Heart, q: Skip):  
  2  (User directly labeling by typing corresponding number as the video above, in this case it's Fire works motion)
  Sequence  f (7) moved to Fire works directory as 43  
  Sequence  f (7) labeled as Fire works
  ```

### Step 4: Model Retraining
   Retrain the model using the updated dataset:  
   `original labeled data + newly labeled uncertain samples`.

### Step 5: Evaluation and Iteration
   Measure the model’s performance (e.g., using the F1 score).  
   Continue to iterate the labeling and training processes until either:
   - The performance exceeds a predefined threshold (e.g., F1 > 0.9), or  
   - There is no remaining unlabeled data.

---
## Performance History

- **Update Strategy:** In each iteration, 10 uncertain samples are added for model retraining.
- **Test Set Composition:** The test set consists of 15 samples for each action.
- **Iterations:** A total of 11 iterations were conducted until no unlabeled data remained.

The following table summarizes the performance metrics across the iterations:

  #### Iteration | F1 Score | Accuracy
  -----------------------------------
          1 | 0.6795 | 0.6957
          2 | 0.7299 | 0.7391
          3 | 0.7768 | 0.7826
          4 | 0.8002 | 0.8043
          5 | 0.7951 | 0.8043
          6 | 0.8202 | 0.8261
          7 | 0.8190 | 0.8261
          8 | 0.8118 | 0.8261
          9 | 0.8190 | 0.8261
         10 | 0.7818 | 0.8043
         11 | 0.8418 | 0.8478

---
   
## Comparison of Performance

The examples below illustrate the evolution of our model's performance throughout the training process. Initially, the model—trained using only a limited amount of labeled data—struggled to differentiate between the "Fire works" gesture and the "Heart" gesture. After iteratively retraining with active learning techniques, the model improved significantly, accurately and rapidly recognizing the intended actions.

- **Initially Trained Model with Limited Labeled Data:**

  ![Initial Model](https://github.com/user-attachments/assets/5c5fe0ef-68c7-4ac0-93ec-6e3707cc092a)

- **Model after Retraining using Active Learning Techniques:**

  ![Best Model](https://github.com/user-attachments/assets/a8062fc4-5894-463a-8fad-3dbf19a17475)


---

## Acknowledgments

- MediaPipe for pose estimation
- PyTorch for deep learning framework 
