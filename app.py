import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Tuple, Sequence
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
import random

# Importing necessary PyTorch modules and utilities for building and training the model
import torch.nn as nn # For defining neural network layers
import torch.nn.functional as F # For using activation functions and operations
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence # For handling variable-length sequences
from torch.utils.data import DataLoader # For batching and loading data efficiently
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate scheduler to adjust learning rates dynamically

# DO NOT CHANGE HERE
random.seed(13)

# Use this for getting x label
def rand_sequence_var_len(n_seqs: int, lb: int=16, ub: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        seq_len = random.randint(lb, ub)
        yield [random.randint(1, 5) for _ in range(seq_len)]


# Use this for getting y label
def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        # note that seq is a string, not a list
        if dimer == "CG":
            cgs += 1
    return cgs


# Alphabet helpers
alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

# Custom dataset class for handling lists and labels
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lists, labels):
        self.lists = lists # Storing the sequences
        self.labels = labels # Storing the labels (targets)

    def __getitem__(self, index):
        return torch.LongTensor(self.lists[index]), self.labels[index] # Returning the sequence as a LongTensor and its corresponding label

    def __len__(self):
        return len(self.lists) # Returning the number of sequences in the dataset

# Utility class for padding sequences and preparing batches
class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True) # Sorting sequences by length (longest first)
        sequences = [x[0] for x in sorted_batch] # Extracting sequences
        labels = [x[1] for x in sorted_batch] # Extracting labels
        lengths = torch.LongTensor([len(x) for x in sequences]) # Recording original lengths
        padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0) # Padding sequences with zeros
        return padded_seqs, torch.tensor(labels, dtype=torch.float), lengths # Returning padded sequences, labels, and lengths

# Define the neural network model for CpG prediction
class VariableLengthCpGPredictor(nn.Module):
    def __init__(self,
                 input_size=6,
                 embedding_dim=128,
                 hidden_size=256,
                 num_layers=3,
                 dropout=0.4):
        super(VariableLengthCpGPredictor, self).__init__()
         # Embedding layer to represent DNA bases as dense vectors
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        # LSTM layer for handling sequential data with bidirectional processing
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # Fully connected layers for prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # First layer
            nn.LayerNorm(hidden_size),  # Layer normalization
            nn.ReLU(), # Activation function
            nn.Dropout(dropout), # Dropout for regularization
            nn.Linear(hidden_size, hidden_size // 2), # Second layer
            nn.LayerNorm(hidden_size // 2), # Layer normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Output layer
        )

    def forward(self, x, lengths):
        batch_size = x.size(0) # Apply embedding
        embedded = self.embedding(x)

        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True) # Packing sequences for LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded) # Passing through LSTM

        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) # Concatenating final hidden states
        output = self.predictor(final_hidden) # Passing through fully connected layers

        return F.relu(output.squeeze()) # Applying ReLU activation and return

# Function to train the model on variable-length sequences
def train_variable_length_model(model, train_loader, val_loader, num_epochs, learning_rate=0.001):
    # Setting the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device) # Moving the model to the selected device
    criterion = nn.MSELoss() # Defining the loss function (Mean Squared Error)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)# Using AdamW optimizer with weight decay for regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True) # Scheduler to reduce learning rate when validation loss plateaus

    best_val_loss = float('inf')  # Initializing the best validation loss as infinity

    for epoch in range(num_epochs): # Looping over epochs
        # Training phase
        model.train()  # Setting the model to training mode
        train_loss = 0.0 # Initializing training loss

        for batch_idx, (padded_seqs, labels, lengths) in enumerate(train_loader): # Looping over training batches

            # Moving data to device
            padded_seqs = padded_seqs.to(device)
            labels = labels.to(device)


            optimizer.zero_grad() # Clearing gradients from the optimizer
            outputs = model(padded_seqs, lengths)  # Forward pass: computing predictions

            loss = criterion(outputs, labels) # Calculating the loss
            loss.backward() # Backpropagating the gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping gradients to avoid exploding gradients
            optimizer.step() # Updating the model parameters
            train_loss += loss.item() # Accumulating training loss

            if (batch_idx + 1) % 10 == 0: # Printing training progress for every 10th batch
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation phase
        model.eval() # Setting the model to evaluation mode
        val_loss = 0.0 # Initializing validation loss


        with torch.no_grad():# Looping over validation batches

            for padded_seqs, labels, lengths in val_loader:
                # Moving data to device
                padded_seqs = padded_seqs.to(device)
                labels = labels.to(device)
                # Forward pass: computing predictions

                outputs = model(padded_seqs, lengths)
                val_loss += criterion(outputs, labels).item() # Accumulating validation loss

        # Calculating average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Printing epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        print("-" * 40)

        # Adjusting learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Saving the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_variable_length_cpg_predictor.pt')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

# Function to prepare training or testing data
def prepare_data(num_samples=100, min_len=16, max_len=128):
    X_dna_seqs_train = list(rand_sequence_var_len(num_samples, min_len, max_len)) # Generating random integer sequences
    temp = ["".join(intseq_to_dnaseq(seq)) for seq in X_dna_seqs_train] # Converting integer sequences to DNA sequences
    y_dna_seqs = [count_cpgs(seq) for seq in temp] # Counting CpG occurrences in DNA sequences
    return X_dna_seqs_train, y_dna_seqs

# Setting random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Preparing the training data
min_len, max_len = 64, 128
print("Preparing training data...")
train_x, train_y = prepare_data(2048, min_len, max_len)
print("Preparing test data...")
test_x, test_y = prepare_data(512, min_len, max_len)

# Creating dataset and dataloader objects for training and testing
train_dataset = MyDataset(train_x, train_y)
test_dataset = MyDataset(test_x, test_y)

# Using DataLoader with padding for efficient batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                        collate_fn=PadSequence())
test_loader = DataLoader(test_dataset, batch_size=32,
                        collate_fn=PadSequence())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VariableLengthCpGPredictor()
# load the model from best_variable_length_cpg_predictor.pt
model.load_state_dict(torch.load('best_variable_length_cpg_predictor.pt', map_location=device))
model.to(device)
model.eval()

# Function to test the model's prediction for a single input sequence
def test_model_prediction(model: VariableLengthCpGPredictor, sequence: List[int]) -> int:
    """Test the model's prediction for a single sequence."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Converting sequence to tensor and add batch dimension
    seq_tensor = torch.LongTensor([sequence]).to(device)
    seq_length = torch.LongTensor([len(sequence)])

    with torch.no_grad():
        prediction = model(seq_tensor, seq_length)
        # Rounding to nearest integer
        return round(prediction.item())

def compare_predictions(sequence: List[int]) -> dict:
    """Compare model prediction with actual CpG count for a sequence."""
    # Converting int sequence to DNA sequence
    dna_seq = "".join(intseq_to_dnaseq(sequence))
    actual_cpg = count_cpgs(dna_seq)

    # Loading the trained model
    model = VariableLengthCpGPredictor()
    # checkpoint = torch.load('best_variable_length_cpg_predictor.pt',
    #                       map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    model.load_state_dict(torch.load('best_variable_length_cpg_predictor.pt', map_location=device))

    predicted_cpg = test_model_prediction(model, sequence)

    return {
        'sequence': dna_seq,
        'actual_cpg': actual_cpg,
        'predicted_cpg': predicted_cpg,
        'difference': abs(actual_cpg - predicted_cpg),
        'sequence_length': len(sequence)
    }

def run_test_cases():
    """Run multiple test cases and display results."""
    # Test case 1: Short sequence with no CpG
    seq1 = [dna2int[c] for c in "AAATTT"]

    # Test case 2: Short sequence with one CpG
    seq2 = [dna2int[c] for c in "ACGTAT"]

    # Test case 3: Medium sequence with multiple CpGs
    seq3 = [dna2int[c] for c in "ACGTACGTACGT"]

    # Test case 4: Random sequence of medium length
    seq4 = list(rand_sequence_var_len(1, lb=32, ub=32))[0]

    # Test case 5: Random sequence of maximum length
    seq5 = list(rand_sequence_var_len(1, lb=128, ub=128))[0]

    test_sequences = [seq1, seq2, seq3, seq4, seq5]
    results = []

    for i, seq in enumerate(test_sequences, 1):
        result = compare_predictions(seq)
        results.append(result)
        st.header(f"\n**Test Case {i}:**")
        st.write(f"Sequence: {result['sequence']}")
        st.write(f"Length: {result['sequence_length']}")
        st.write(f"Actual CpG count: {result['actual_cpg']}")
        st.write(f"Predicted CpG count: {result['predicted_cpg']}")
        st.write(f"Absolute difference: {result['difference']}")

    # Calculating overall statistics
    differences = [r['difference'] for r in results]
    st.header("\nOverall Statistics:")
    st.write(f"Mean absolute error: {np.mean(differences):.2f}")
    st.write(f"Standard deviation of error: {np.std(differences):.2f}")
    st.write(f"Maximum error: {max(differences)}")
    st.write(f"Minimum error: {min(differences)}")

# Streamlit App
st.title("CpG Prediction in DNA Sequences - part2 - DNA sequences are not the same length")
st.markdown("""
This app predicts the number of CpG counts in a given DNA sequence.
""")
run_test_cases()