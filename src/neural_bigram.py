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

def model(position,destination,weights):
    
    #turn the lists into tensors and then make all numbers into one-hot vectors (e.g. 00000010000)
    position_tensor = torch.tensor(position)
    destination_tensor = torch.tensor(destination)
    
    position_tensor = torch.nn.functional.one_hot(position_tensor, num_classes=28).float()
    
    for epoch in range(1000):
        
        logits = position_tensor @ weights
        loss = fu.cross_entropy(logits,destination_tensor)
        
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
def generate(weights):
    name = [26]
    curr_letter = torch.tensor(26)
    while True:
        
        curr_letter = fu.one_hot(curr_letter,num_classes=28).float()
        logits = curr_letter @ weights
        probs = fu.softmax(logits,dim=0)
        next_letter = torch.multinomial(probs, num_samples=1).item()
        name.append(next_letter)
        
        #if we find the "<end>" token we break
        if next_letter == 27:
            break
        curr_letter = torch.tensor(next_letter)
        
    return name

def decode_names(name):
    alphabet = string.ascii_lowercase
    name = [alphabet[letter] for letter in name if letter != 26 and letter != 27]
    return "".join(name)
        
        