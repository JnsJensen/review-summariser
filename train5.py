# Math and data
import numpy as np
import pandas as pd
import polars as pl
import math
# Neural network frameworks
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2TokenizerFast
# Utilities
import re
from enum import Enum
import contractions as ct
import utility as util
import json
import random
import os
from torch.utils.tensorboard import SummaryWriter

# Pytorch device
device = th.device("mps") if th.backends.mps.is_available() else th.device("cuda") if th.cuda.is_available() else th.device("cpu")
if device.type == "cuda":
    print(th.cuda.get_device_name(device))
else:
    print(device)

# device = th.device("cpu")

fig_dir = "img/"

# torch dataset from pandas dataframe
# defines a voacbulary of words and converts the review text to a list of indices
# beware of symbols like ., !, ? etc.
# pad the review text and summary to max_review_len and max_summary_len respectively

"""
ReviewDataset pytorch dataset interface
- expects a polars dataframe with columns reviewText, summary, overall
- expects it in the DatasetType.PRUNED format
- expects a GPT2Tokenizer
"""
class ReviewDataset(Dataset):
    def __init__(self, path: str, tokenizer: GPT2Tokenizer, length = None, dataset_type = util.DatasetType.PRUNED, device = "cpu"):
        self.df = util.load_dataset(path, dataset_type)
        if length is not None:
            # clip the dataset to length
            length = min(length, len(self.df))
            self.df = self.df.sample(length, shuffle=True)
        self.dataset_type = dataset_type

        match path:
            case util.Paths.arts:
                self.max_review_len = util.MaxTokenLength.ARTS_REVIEW
                self.max_summary_len = util.MaxTokenLength.ARTS_SUMMARY
            case util.Paths.video:
                self.max_review_len = util.MaxTokenLength.VIDEO_REVIEW
                self.max_summary_len = util.MaxTokenLength.VIDEO_SUMMARY
            case util.Paths.gift:
                self.max_review_len = util.MaxTokenLength.GIFT_REVIEW
                self.max_summary_len = util.MaxTokenLength.GIFT_SUMMARY
            case _:
                raise ValueError("Invalid path")
        
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = self.df["reviewText"][idx]
        summary = self.df["summary"][idx]
        rating = th.tensor(self.df["overall"][idx])

        # Tokenize the review and summary strings
        review = self.tokenizer.encode(review, add_special_tokens = True, padding = "max_length", truncation = True, max_length=self.max_review_len, return_tensors = "pt").squeeze()
        summary = self.tokenizer.encode(summary, add_special_tokens = True, padding = "max_length", truncation = True, max_length=self.max_summary_len, return_tensors = "pt").squeeze()

        # Move tensors to device
        review = review.to(self.device)
        summary = summary.to(self.device)
        rating = rating.to(self.device)
        
        return review, summary, rating
    
    def detokenize(self, x: th.Tensor):
        # # Remove everything after the first <eos> token
        # # This is important due to the fact that that output token is initialised with zeros
        # is_eos = (x == self.tokenizer.eos_token_id).long()
        # if is_eos.any():
        #     x = x[:is_eos.argmax().item()]

        return self.tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def batch_detokenize(self, x: th.Tensor):
        return [self.detokenize(x[i]) for i in range(len(x))]


"""
Model
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        # self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers, bidirectional=False)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        context_vector, hidden = self.gru(embedded, hidden)
        # context_vector, (hidden, cell_state) = self.lstm(embedded, (hidden, th.zeros_like(hidden, device=device)))
        return context_vector, hidden

    def initHidden(self):
        dimension_1 = self.num_layers * (2 if self.bidirectional else 1)
        return th.zeros(dimension_1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, num_layers=1, bidirectional=True):
        super(AttnDecoderRNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length # The size of the vocabulary - len(tokenizer)

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * (2 + 1 if bidirectional else 0), self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        # self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers, bidirectional=False)
        self.out = nn.Linear(self.hidden_size * (2 if bidirectional else 0), self.output_size)

    def forward(self, input, hidden, context_vector):
        embedded = self.embedding(input).view(1, 1, -1)

        # print(f"embedded.shape: {embedded.shape}")
        # print(f"hidden.shape: {hidden.shape}")
        # print(f"context_vector.shape: {context_vector.shape}")

        attn_weights = nn.functional.softmax(
            self.attn(th.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = th.bmm(attn_weights.unsqueeze(0), context_vector.unsqueeze(0))

        # print(f"attn_weights.shape: {attn_weights.shape}")
        # print(f"attn_applied.shape: {attn_applied.shape}")

        output = th.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        # output, (hidden, cell_state) = self.lstm(output, (hidden, th.zeros_like(hidden, device=device)))

        output = nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        dimension_1 = self.num_layers * (2 if self.bidirectional else 1)
        return th.zeros(dimension_1, 1, self.hidden_size, device=device)

# For saving the states of the model, optimiser and epoch nr.
def save_state(loss, accuracy, encoder_model, decoder_model, encoder_optimiser, decoder_optimiser, iteration, run_dir):
    print(f"Saving model at iteration {iteration} with loss {loss} and accuracy {accuracy}")
    checkpoint = {
        "loss": loss,
        "accuracy": accuracy,
        "encoder_model_state": encoder_model.state_dict(),
        "decoder_model_state": decoder_model.state_dict(),
        "encoder_optimiser_state": encoder_optimiser.state_dict(),
        "decoder_optimiser_state": decoder_optimiser.state_dict(),
        "iteration": iteration
    }
    
    save_dir = f'checkpoints/{run_dir[5:8]}'
    if os.path.exists(save_dir):
        for file in os.listdir(f'checkpoints/{run_dir[5:8]}'):
            os.remove(f"{save_dir}/{file}")
    else:
        os.makedirs(save_dir)
    
    model_checkpoint_path = f"{save_dir}/{run_dir[9:]}_l_{loss:.5f}_a_{accuracy:.5f}_{iteration}.pth"
    th.save(checkpoint, model_checkpoint_path)

# Prepare for training
debugging = False # For debugging prints
model_version = "3.0.0_GRU_bi"
n_epochs = 1000
batch_size = 64
learning_rate = 0.00001
teacher_forcing_ratio = 0.5
hidden_size = 2**8 # 256
dataset_size = 8000
num_layers = 2 # LSTM or GRU layers
bidirectional = True

#-----------------------------------------------------------------------------------------------------------------------------------

run_dir = util.get_run_dir()
# add run info to the directory
run_dir += f"_mv_{model_version}_nl_{num_layers}_bs_{batch_size}_lr_{learning_rate}_tfr_{teacher_forcing_ratio}_hs_{hidden_size}_ds_{dataset_size}"
print(f"Current run_dir: {run_dir}")
# Readying the writer
writer = SummaryWriter(run_dir)

#-----------------------------------------------------------------------------------------------------------------------------------

# criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # TODO: Check without the ignore_index
# criterion = nn.CrossEntropyLoss() # TODO: Check without the ignore_index
criterion = nn.NLLLoss()

# Instantiate tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_bos_token=True, add_prefix_space=True, trim_offsets=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"bos_token": util.BOS_token})

# Max length is the max index of the vocabulary
MAX_LENGTH = len(tokenizer)
print(f"MAX_LENGTH: {MAX_LENGTH}")

encoder = EncoderRNN(MAX_LENGTH, hidden_size, num_layers=num_layers, bidirectional=bidirectional).to(device).train()
decoder = AttnDecoderRNN(hidden_size, MAX_LENGTH, num_layers=num_layers, max_length=MAX_LENGTH, bidirectional=bidirectional).to(device).train()

# encoder_optimizer = th.optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = th.optim.Adam(decoder.parameters(), lr=learning_rate)
encoder_optimizer = th.optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = th.optim.SGD(decoder.parameters(), lr=learning_rate)

#-----------------------------------------------------------------------------------------------------------------------------------

# Create the dataset
dataset = ReviewDataset(util.Paths.arts, tokenizer, length=dataset_size, device=device)

# Calculate the number of elements in each bucket
split_ratios = [0.7, 0.2, 0.1]

# Get the data loaders
train_loader, val_loader, test_loader = util.get_data_loaders(dataset, batch_size, split_ratios)

def val(val_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    with th.no_grad():
        encoder.eval()
        decoder.eval()
        # We calcualte the loss and backpropagate every batch
        # Reset the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        batch_loss = 0
        words_in_batch = 0
        correct_words_in_batch = 0

        val_review_batch, val_summary_batch, val_rating_batch = next(val_iter)
        for val_review, val_summary, val_rating in zip(val_review_batch, val_summary_batch, val_rating_batch):
            # Create the encoder hidden state
            encoder_hidden = encoder.initHidden() # Can be understood as the context vector

            # Get the length of the review
            review_length = val_review.shape[0]
            summary_length = val_summary.shape[0]

            # Create the encoder outputs tensor
            encoder_outputs = th.zeros(MAX_LENGTH, encoder.hidden_size * (2 if encoder.bidirectional else 1), device=device)

            # Encoder forward pass
            for ei in range(review_length):
                encoder_output, encoder_hidden = encoder(val_review[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            # Create the decoder input, the beginning of the sequence, starting with the BOS (Beginning Of String) token
            bos = th.tensor(tokenizer.bos_token_id).to(device)
            decoder_input = th.tensor([bos], device=device, dtype=th.long)

            # Initialize the decoder output
            decoder_output_sequence = th.empty(dataset.max_summary_len, device=device, dtype=th.long).fill_(tokenizer.pad_token_id)
            decoder_output_sequence[0] = decoder_input

            decoder_hidden = encoder_hidden

            # Decoder forward pass
            # Run the decoder
            for target_index, target in enumerate(val_summary[1:]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach() # detach from history as input
                
                # Append the output
                decoder_output_sequence[target_index+1] = decoder_input

                # Count the correct words
                if debugging:
                    print(f"Pred: {decoder_input.item()}, Target: {target.item()}")
                if decoder_input.item() == target.item():
                    correct_words_in_batch += 1

                words_in_batch += 1
                # Calculate the loss
                batch_loss += criterion(decoder_output.squeeze(), target)

                if decoder_input.item() == tokenizer.eos_token_id:
                    #print(f"EOS token found at iteration {target_index+1}")
                    break

        encoder.train()
        decoder.train()

        # Normalize the loss and accuracy
        batch_loss /= words_in_batch
        accuracy = correct_words_in_batch / words_in_batch

        return accuracy, batch_loss.item(), decoder_output_sequence

def train(learning_rate, val_loader, n_epochs, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    min_loss = np.inf
    for epoch in range(n_epochs):

        val_iter = iter(val_loader)

        for batch_idx, (train_review_batch, train_summary_batch, train_rating_batch) in enumerate(train_loader):
            batch_loss = 0
            words_in_batch = 0
            correct_words_in_batch = 0

            # We calcualte the loss and backpropagate every batch
            # Reset the gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            for review, summary, rating in zip(train_review_batch, train_summary_batch, train_rating_batch):
                # Create the encoder hidden state
                encoder_hidden = encoder.initHidden() # Can be understood as the context vector

                # Initialise the encoder output's "feature space"
                context_vector = th.zeros(MAX_LENGTH, encoder.hidden_size * (2 if encoder.bidirectional else 1), device=device)

                # Run the encoder
                for token in review:
                    encoder_output, encoder_hidden = encoder(token, encoder_hidden)
                    context_vector[token] = encoder_output[0, 0]
                
                bos = th.tensor(tokenizer.bos_token_id).to(device)

                # Create the decoder input, the beginning of the sequence, starting with the BOS (Beginning Of String) token
                decoder_input = th.tensor([bos], device=device, dtype=th.long)

                # Initialize the decoder output
                decoder_output_sequence = th.empty(dataset.max_summary_len, device=device, dtype=th.long).fill_(tokenizer.pad_token_id)
                decoder_output_sequence[0] = decoder_input

                # Propagate the decoder hidden state
                decoder_hidden = encoder_hidden

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                if use_teacher_forcing:
                    if debugging:
                        #util.print_mod(f"USING TEACHER FORCING", [util.Modifiers.Colors.GREEN])
                        print("USING TEACHER FORCING")
                    
                    # Teacher forcing: Feed the target as the next input
                    for target_index, target in enumerate(summary[1:]):
                        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, context_vector)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = target # Teacher forcing

                        # Append the output
                        prediction = topi.squeeze().detach() # detach from history as input
                        decoder_output_sequence[target_index+1] = prediction

                        # Count the correct words
                        if prediction.item() == target.item():
                            correct_words_in_batch += 1

                        words_in_batch += 1
                        # Calculate the loss
                        batch_loss += criterion(decoder_output.squeeze(), target)

                        if decoder_input.item() == tokenizer.eos_token_id:
                            #print(f"EOS token found at iteration {target_index+1}")
                            break
                else:
                    if debugging:
                        #util.print_mod(f"NOT USING TEACHER FORCING", [util.Modifiers.Colors.CYAN])
                        print("NOT USING TEACHER FORCING")

                    # Run the decoder
                    for target_index, target in enumerate(summary[1:]):
                        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, context_vector)
                        topv, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze().detach() # detach from history as input
                        
                        # Append the output
                        decoder_output_sequence[target_index+1] = decoder_input

                        # Count the correct words
                        if decoder_input.item() == target.item():
                            correct_words_in_batch += 1

                        words_in_batch += 1
                        # Calculate the loss
                        batch_loss += criterion(decoder_output.squeeze(), target)

                        if decoder_input.item() == tokenizer.eos_token_id:
                            #print(f"EOS token found at iteration {target_index+1}")
                            break
                
                if debugging:
                    # print tokenized output
                    #util.print_mod("Target tokenized:", [util.Modifiers.Styles.BOLD, util.Modifiers.Styles.ITALIC])
                    print(f"Target tokenized:")
                    print(summary.tolist())
                    # util.print_mod("Target Sequence:", [util.Modifiers.Styles.BOLD, util.Modifiers.Styles.ITALIC])
                    print("Target Sequence:")
                    print(dataset.detokenize(summary))

                    # print tokenized output
                    #util.print_mod("Tokenized output:", [util.Modifiers.Styles.BOLD, util.Modifiers.Styles.ITALIC])
                    print(f"Tokenized output:")
                    print(decoder_output_sequence.tolist())
                    #util.print_mod("Detokenized output:", [util.Modifiers.Styles.BOLD, util.Modifiers.Styles.ITALIC])
                    print(f"Detokenized output:")
                    print(f"{dataset.detokenize(decoder_output_sequence)}\n")
            
            # Normalize the loss
            batch_loss /= words_in_batch
            # Backpropagate the loss
            batch_loss.backward()
            # Accuracy
            accuracy = correct_words_in_batch / words_in_batch

            # Update the weights
            encoder_optimizer.step()
            decoder_optimizer.step()

            iteration = epoch * len(train_loader) + batch_idx
            # Print the loss and accuracy
            writer.add_scalar("Loss/train", batch_loss, iteration)
            writer.add_scalar("Accuracy/train", accuracy, iteration)
            # print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {batch_loss}")
            # util.print_mod(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {batch_loss}", [util.Modifiers.Colors.MAGENTA, util.Modifiers.Styles.BOLD])
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {batch_loss}, Accuracy: {accuracy}")

            # Validate the model
            if batch_idx % 5 == 0:
                val_acc, val_loss, val_sequence = val(val_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
                val_sequence_detokenized = dataset.detokenize(val_sequence)
                writer.add_scalar("Loss/val", val_loss, iteration)
                writer.add_scalar("Accuracy/val", val_acc, iteration)
            
            if batch_idx % 20 == 0 and not debugging:
                if val_loss < min_loss and iteration > 100:
                    save_state(val_loss, val_acc, encoder, decoder, encoder_optimizer, decoder_optimizer, iteration, run_dir)

train(learning_rate, val_loader, n_epochs, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)