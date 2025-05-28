
from io import open
import unicodedata
import string
import random
import re

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
import time, copy
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions combined from PyTorch tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
# This is important because we want all words to be formatted the same similar
# to our image normalization
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!'?]+", r" ", s)
    return s

def parse_data(filename):
    # Read the file and split into lines
    lines = open(filename, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Throw out the attribution as it is not a part of the data
    pairs = [[pair[0], pair[1]] for pair in pairs]

    return pairs


# Parse data from the given file
pairs = parse_data("spa.txt")

# We only want the english sentences because we aren't translating
english_sentences = [pair[0] for pair in pairs]

# Shuffle our dataset
random.shuffle(english_sentences)
print("Number of English sentences:", len(english_sentences))


# Since we already shuffled our dataset, grab a random sampling of sentences for our train, val, and test
# Here we are using a small number of Sentences to ease training time. Feel free to use more
train_sentences = english_sentences[:1000]
val_sentences = english_sentences[1000:2000]
test_sentences = english_sentences[2000:3000]

# Using this function we will create a dictionary to use for our one hot encoding vectors
def add_words_to_dict(word_dictionary, word_list, sentences):
    for sentence in sentences:
        for word in sentence.split(" "):
            if word in word_dictionary:
                continue
            else:
                word_list.append(word)
                word_dictionary[word] = len(word_list)-1

english_dictionary = {}
english_list = []
add_words_to_dict(english_dictionary, english_list, train_sentences)
add_words_to_dict(english_dictionary, english_list, val_sentences)
add_words_to_dict(english_dictionary, english_list, test_sentences)


# Now make our training samples:
def create_input_tensor(sentence, word_dictionary):
    words = sentence.split(" ")
    tensor = torch.zeros(len(words), 1, len(word_dictionary)+1)
    for idx in range(len(words)):
        word = words[idx]
        tensor[idx][0][word_dictionary[word]] = 1
    return tensor

def create_target_tensor(sentence, word_dictionary):
    words = sentence.split(" ")
    tensor = torch.zeros(len(words), 1, len(word_dictionary)+1)
    for idx in range(1, len(words)):
        word = words[idx]
        if word not in word_dictionary:
            print("Error: This word is not in our dataset - using a zeros tensor")
            continue
        tensor[idx-1][0][word_dictionary[word]] = 1
    tensor[len(words)-1][0][len(word_dictionary)] = 1 # EOS
    return tensor


train_tensors = [(create_input_tensor(sentence, english_dictionary), create_target_tensor(sentence, english_dictionary)) for sentence in train_sentences]
val_tensors = [(create_input_tensor(sentence, english_dictionary), create_target_tensor(sentence, english_dictionary)) for sentence in val_sentences]
test_tensors = [(create_input_tensor(sentence, english_dictionary), create_target_tensor(sentence, english_dictionary)) for sentence in test_sentences]


def tensor_to_sentence(word_list, tensor):
    sentence = ""
    for i in range(tensor.size(0)):
        topv, topi = tensor[i].topk(1)
        if topi[0][0] == len(word_list):
            sentence += "<EOS>"
            break
        sentence += word_list[topi[0][0]]
        sentence += " "
    return sentence


print("This code helps visualize which words represent an input_tensor and its corresponding target_tensor!")
examples_to_show = 6
count = 1
for input, target in train_tensors:
    print(tensor_to_sentence(english_list, input))
    print(tensor_to_sentence(english_list, target))
    count +=1
    if count > examples_to_show:
        break


# Let's look at a few sentence encodings, to see what those look like:
for i in range(6):
    print(train_sentences[i], "[encode as]", train_tensors[i][0])


dataloaders = {'train': train_tensors,
               'val': val_tensors,
               'test': test_tensors}

dataset_sizes = {'train': len(train_tensors),
                 'val': len(val_tensors),
                 'test': len(test_tensors)}

print(f'dataset_sizes = {dataset_sizes}')


class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TwoLayerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Set batch_first to False to align with the per-step processing in train_lstm
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    # forward to accept hidden state
    def forward(self, x, hidden):
        # x should have shape (input_size) when coming from the train loop step
        # hidden should be a tuple of (h_n, c_n), each with shape (num_layers, batch_size=1, hidden_size)

        x = x.unsqueeze(0) # shape becomes (1, input_size) - sequence length dimension
        x = x.unsqueeze(0) # shape becomes (1, 1, input_size) - batch dimension (batch_size=1)

        # Forward propagate LSTM, The LSTM layer returns output, (h_n, c_n)
        # out shape: (sequence_length, batch_size, hidden_size * num_directions) -> (1, 1, hidden_size)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size) -> (num_layers, 1, hidden_size)
        # c_n shape: (num_layers * num_directions, batch_size, hidden_size) -> (num_layers, 1, hidden_size)
        out, hidden = self.lstm(x, hidden)

        # Decode the hidden state of the last time step (which is the only step in this case)
        # out shape is now (1, 1, hidden_size)
        # Squeeze the sequence_length dimension before the linear layer
        # The batch_size dimension (dim=0 after squeezing sequence_length) should be kept
        # We also need to squeeze the batch dimension (dim=0) from the output (1, 1, hidden_size) to get (1, hidden_size) before the FC layer expects (batch_size, hidden_size) -> (1, hidden_size)
        out = self.fc(out.squeeze(0).squeeze(0)) # shape becomes (hidden_size) -> FC converts to (output_size)
        return out, hidden

    def initHidden(self):
        # We need two hidden layers because of our two layered lstm!
        # The hidden state should have shape (num_layers, batch_size, hidden_size)
        # In our case, batch_size is 1 when processing sequence step-by-step
        return (torch.zeros(self.num_layers, 1, self.hidden_size).to(device),
                torch.zeros(self.num_layers, 1, self.hidden_size).to(device))


def train_lstm(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # keep the best weights stored separately
    best_loss = np.inf
    best_epoch = 0

    # Each epoch has a training, validation, and test phase
    phases = ['train', 'val', 'test']

    # Keep track of how loss evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase+'_loss'] = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data (each item is an input/target sequence pair)
            for input_sequence, target_sequence in dataloaders[phase]:
                # Now Iterate through each sequence here:

                hidden = model.initHidden() # Start with a fresh hidden state for each sequence
                input_sequence = input_sequence.squeeze(1).to(device) # shape becomes (sequence_length, input_size)
                target_sequence = target_sequence.squeeze(1).to(device) # shape becomes (sequence_length, input_size)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    loss = 0
                    # Make a prediction for each element in the sequence,
                    # keeping track of the hidden state along the way
                    for i in range(input_sequence.size(0)):
                        # Pass one time step of the sequence and the current hidden state
                        # The model's forward method now expects input of shape (input_size) and hidden state
                        output, hidden = model(input_sequence[i], hidden)

                        # target_sequence[i] has shape (input_size), need to get the index of the target word
                        # This assumes target_sequence is one-hot encoded
                        # Use torch.argmax to get the index of the target word
                        target_index = torch.argmax(target_sequence[i])

                        # target_index shape is now () (scalar)
                        l = criterion(output.unsqueeze(0), target_index.unsqueeze(0)) # Add batch dimension to output and target
                        loss += l

                    # backward + update weights only if in training phase at the end of a sequence
                    if phase == 'train':
                        # Normalize loss by the sequence length
                        loss = loss / input_sequence.size(0)
                        loss.backward()
                        optimizer.step()

                # The loss variable already contains the sum of losses for the sequence steps
                running_loss += loss.item()

            if phase == 'train' and scheduler is not None: # Check if scheduler exists before stepping
                scheduler.step()

            # Normalize epoch_loss by the number of sequences in the dataset
            epoch_loss = running_loss / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)

            print(f'{phase:5} Loss: {epoch_loss:.4f}')

            # deep copy the model if it's the best loss
            # Note: We are using the train loss here to determine our best model
            if phase == 'train' and epoch_loss < best_loss:
              best_epoch = epoch
              best_loss = epoch_loss
              best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best train Loss: {best_loss:4f} at epoch {best_epoch}')


    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, training_curves


num_epochs = 25
input_size = len(english_dictionary) + 1
hidden_size = 100
num_layers = 2
output_size = len(english_dictionary) + 1

model = TwoLayerLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer really doesn't use a scheduler

scheduler = None # Set to None if not using a scheduler


# JUST PRINTING MODEL & PARAMETERS
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


lstm, training_curves = train_lstm(model, dataloaders, dataset_sizes,
                                     criterion, optimizer, scheduler, num_epochs=num_epochs)

def predict(model, word_dictionary, word_list, input_sentence, max_length = 20):
    model.eval() # Set model to evaluation mode
    output_sentence = input_sentence + " "
    input_tensor = create_input_tensor(input_sentence, word_dictionary).squeeze(1).to(device) # shape becomes (sequence_length, vocab_size)
    hidden = model.initHidden()

    # Process the input sentence word by word to get the final hidden state
    # and the output prediction for the last word
    last_output = None
    for i in range(input_tensor.size(0)):
        with torch.no_grad(): # No need to calculate gradients during prediction
            last_output, hidden = model(input_tensor[i], hidden) # input_tensor[i] is (vocab_size), which is the expected input shape for forward

    # Now use the output from the last input word to predict the next word
    # last_output has shape (output_size)
    predicted_word_tensor = last_output.unsqueeze(0) # Add batch dimension for topk: (1, output_size)

    for i in range(len(input_sentence.split(" ")), max_length):
        with torch.no_grad(): # No need to calculate gradients during prediction
            topv, topi = predicted_word_tensor.topk(1) # topk returns (values, indices)
            predicted_index = topi.item() # Get the index of the most likely word

            if predicted_index == len(word_dictionary): # Check for EOS token
                # print("Hit the EOS")
                break

            word = word_list[predicted_index]
            output_sentence += word
            output_sentence += " "

            next_input_tensor = create_input_tensor(word, word_dictionary).squeeze(1).squeeze(0).to(device) # shape (1, vocab_size) -> squeeze(1) -> (vocab_size) -> squeeze(0) -> (vocab_size)

            output, hidden = model(next_input_tensor, hidden)
            predicted_word_tensor = output.unsqueeze(0) # Add batch dimension for the next iteration's topk: (1, output_size)


    return output_sentence.strip() # Strip trailing space


print(predict(lstm, english_dictionary, english_list, "what is"))
print(predict(lstm, english_dictionary, english_list, "my name"))
print(predict(lstm, english_dictionary, english_list, "how are"))
print(predict(lstm, english_dictionary, english_list, "hi"))
print(predict(lstm, english_dictionary, english_list, "choose"))



