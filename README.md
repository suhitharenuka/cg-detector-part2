# CpG Predictor for Variable-Length DNA Sequences

This project builds and trains a neural network using PyTorch to predict the number of CpG dimers ("CG") in synthetic DNA sequences of variable lengths. It includes custom data preparation, model architecture, training, and evaluation workflows.

---

## 📚 Overview

- **Task**: Predict the number of "CG" dimers in a given DNA sequence.
- **Input**: Variable-length DNA sequences (encoded as integers).
- **Model**: Bi-directional LSTM with embedding, fully connected layers, and regularization techniques.
- **Training**: Using Mean Squared Error loss with AdamW optimizer and dynamic learning rate adjustment.

---

## 🛠️ Key Components

### 1. Data Preparation
- **Synthetic DNA Sequences**: Generated randomly with specified minimum and maximum lengths.
- **Label Generation**: Labels represent the count of "CG" dimers in each sequence.
- **Encoding**: DNA bases (`A`, `C`, `G`, `T`, `N`) are mapped to integers.

### 2. Dataset and Dataloader
- **Custom `MyDataset` class**: Handles input sequences and corresponding labels.
- **Padding with `PadSequence`**: Ensures that sequences in a batch are padded to the same length.

### 3. Model: `VariableLengthCpGPredictor`
- **Embedding Layer**: Converts integer-encoded bases to dense vectors.
- **Bi-directional LSTM**: Captures context from both directions of the sequence.
- **Fully Connected Layers**: Predicts the CpG count from the LSTM output.
- **Regularization**: Dropout and LayerNorm to prevent overfitting.

### 4. Training
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: AdamW with weight decay.
- **Scheduler**: Reduce learning rate when validation loss plateaus.
- **Gradient Clipping**: Helps prevent exploding gradients.
- **Checkpointing**: Saves the best model based on validation loss.

### 5. Testing
- **Test Prediction Function**: Makes predictions for individual sequences.

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install torch
   ```

2. **Prepare data**:
   - Training set: 2048 samples
   - Testing set: 512 samples
   - Sequences have lengths between 64 and 128 bases.

3. **Initialize and Train the model**:
   ```python
   model = VariableLengthCpGPredictor()
   train_variable_length_model(model, train_loader, test_loader, num_epochs=30)
   ```

4. **Save Best Model**:
   - Best model is automatically saved as `best_variable_length_cpg_predictor.pt`.


## 📈 Results
- The model's training and validation loss is printed after each epoch.
- The learning rate is adjusted dynamically if the validation loss plateaus.

---
