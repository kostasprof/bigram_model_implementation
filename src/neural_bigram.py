#Neural Network Approach for Bigram Prediction
import string
from collections import defaultdict
import torch

'''
Assign integers to every letter in the pairs so we can later use them in 
tensors
'''
def pairs_to_integers(pairs):
    alphabet = string.ascii_lowercase
    numbers = [x for x in range(26)]
    alphabet_codes = defaultdict()
    alphabet_codes = {letter:number for letter,number in zip(alphabet,numbers)}
    alphabet_codes["<start>"] = 26
    alphabet_codes["<end>"] = 27
    position = []
    destination = []

    for pair in pairs:
        if pair[0] == "<start>":
            position.append(alphabet_codes["<start>"])
            destination.append(alphabet_codes[pair[1]])
        elif pair[1] == "<end>":
            position.append(alphabet_codes[pair[0]])
            destination.append(alphabet_codes["<end>"])
        else:
            position.append(alphabet_codes[pair[0]])
            destination.append(alphabet_codes[pair[1]])

    return position,destination

def model(position,destination):
    #turn the lists into tensors and then make all numbers into one-hot vectors (e.g. 00000010000)
    position_tensor = torch.tensor(position)
    destination_tensor = torch.tensor(destination)
    position_tensor = torch.nn.functional.one_hot(position_tensor, num_classes=28).float()
    destination_tensor = torch.nn.functional.one_hot(destination_tensor, num_classes=28).float()
    
    
    return position_tensor