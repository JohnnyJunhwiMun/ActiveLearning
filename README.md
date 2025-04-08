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


    <p align="center">
      <video src="https://github.com/user-attachments/assets/671a8827-731c-4dc7-99e8-7533c97b234f" controls width="100" height="200"></video>
    </p>
    
    
    #### Example in Terminal
    
    ```bash
    Select action (1: Like, 2: Fire works, 3: Heart, q: Skip):  
    2  (User directly labeling by typing corresponding number as the video above, in this case it's Fire works motion)
    Sequence  f (7) moved to Fire works directory as 43  
    Sequence  f (7) labeled as Fire works
    ```

5. **Model Retraining**  
   Retrain the model using the updated dataset:  
   `original labeled data + newly labeled uncertain samples`.

6. **Evaluation & Iteration**  
   Measure the model’s performance (e.g., using the F1 score).  
   Continue to iterate the labeling and training processes until either:
   - The performance exceeds a predefined threshold (e.g., F1 > 0.9), or  
   - There is no remaining unlabeled data.

   
    ### Example of Performance History:

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
    Active Learning Iteration 12
 
    No more unlabeled data available.
   
7. **Comparison of Performance**
   
    It's showing simple example. Intially trained model is confused 'Fire work' motion with 'Heart'.
    
    However, retrained model by active leaerning techniques detected accurately and fast.
    
    - **Initially Trained Model with Small Labeled Data:**
    
    ![Initialmodel](https://github.com/user-attachments/assets/5c5fe0ef-68c7-4ac0-93ec-6e3707cc092a)
    
    - **After Retraining using Active Learning Techniques:**
    
    ![Bestmodel](https://github.com/user-attachments/assets/a8062fc4-5894-463a-8fad-3dbf19a17475)
    




## Acknowledgments

- MediaPipe for pose estimation
- PyTorch for deep learning framework 
