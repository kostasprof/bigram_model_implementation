#Neural Network Approach for Bigram Prediction
import string
from collections import defaultdict
import torch
import torch.nn.functional as fu

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


'''
training function
'''

def train(position,destination,weights):
    
    #turn the lists into tensors and then make all numbers into one-hot vectors (e.g. 00000010000)
    position_tensor = torch.tensor(position)
    destination_tensor = torch.tensor(destination)
    
    position_tensor = torch.nn.functional.one_hot(position_tensor, num_classes=28).float()
    
    for epoch in range(1000):
        
        logits = position_tensor @ weights
        loss = fu.cross_entropy(logits, destination_tensor) + 0.01 * (weights ** 2).mean()
        
        if weights.grad is not None:
            weights.grad.zero_()
        
        #get gradients
        loss.backward()
        
        #update weights
        with torch.no_grad():
            weights -= 50.0 * weights.grad
        if epoch % 10 ==0:
            print(f"For epoch {epoch} loss is: {loss.item():.4f}")
        
    
    return weights

#function to generate a name
@torch.no_grad()
def generate(weights):
    name = []
    curr = 26  # <start>

    while True:
        logits = weights[curr].clone()

        #nothing can ever transition INTO <start>
        logits[26] = float("-inf")
        #an empty name (<start> -> <end>) is not allowed either
        if curr == 26:
            logits[27] = float("-inf")

        probs = fu.softmax(logits, dim=0)
        curr = torch.multinomial(probs, num_samples=1).item()

        if curr == 27:  #<end>
            break
        name.append(curr)

    return name

@torch.no_grad()
def evaluate(position, destination, weights):
    """Average negative log likelihood per transition (same metric as the counting model)."""
    logits = weights[torch.tensor(position)]   # row lookup == one_hot @ weights
    return fu.cross_entropy(logits, torch.tensor(destination)).item()

def decode_names(name):
    alphabet = string.ascii_lowercase
    name = [alphabet[letter] for letter in name if letter != 26 and letter != 27]
    return "".join(name)
        
        