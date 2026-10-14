# Bigram Model Implementation
from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import string
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/names.txt")


'''
first get all names from the txt file without any \n characters
then remove duplicate names and split the text file into a 80/20 ratio
so we can have names to compute the probabilities and names for testing 
the model
'''
def handle_input():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        names = [name.strip() for name in file]
    names = sorted(set(names))
    train,test = train_test_split(names,test_size=0.2,train_size=0.8,random_state=42,shuffle=True)
    return train, test

# then we get all bigrams of letters
def get_pairs(names):
    letter_bigrams = []
    for name in names:
        # get the first letter ,with the start word token, and the bigrams's count in the dictionary
        for i in range(len(name)):
            if i == 0:
                letter_bigrams.append(("<start>", name[0]))
            elif i == len(name)-1:
                letter_bigrams.append((name[i], "<end>"))
                break
            letter_bigrams.append((name[i], name[i+1]))
            
    return letter_bigrams

# create a dictionary to store letter bigrams and their counts
def count_bigrams(pairs):
    bigrams_dict = defaultdict(int)
    for pair in pairs:
        bigrams_dict[pair] += 1
    return dict(sorted(bigrams_dict.items(), key = lambda kv: kv[1], reverse = True))
    
''' 
calculate the propabilities for every pair by finding the total occurences of a starting letter 
and then find the probability by diving the count of the bigram with the total occurences of the first letter
also find bigrams that have not been found and give them a very small probability (Laplace Smoothing)
'''
def calculate_probabilities(bigrams):
    probabilities = defaultdict()
    letter_counts = defaultdict(int)
    alphabet = list(string.ascii_lowercase) 
    alphabet.append("<end>")
    alphabet.append("<start>")
    
    for b,count in bigrams.items():
        letter_counts[b[0]] += count
    
    # find bigrams that where not in the dataset
    for curr_letter in alphabet:
        for next_letter in alphabet:
            pair = (curr_letter, next_letter)
            is_unseen = pair not in bigrams
            valid_source = curr_letter != "<end>"
            valid_destination = next_letter != "<start>"
            not_empty_name = pair != ("<start>", "<end>")
            if is_unseen and valid_source and valid_destination and not_empty_name:
                bigrams[pair] = 0
    
    #assign probabilities to bigrams with Laplace Smoothing for bigrams that we didnt encounter 
    for b,count in bigrams.items():
        # time for Laplace smoothing so when the model finds a bigram that does not exist
        if b[0] != "<start>":
            probabilities[b] = (count+1)/(letter_counts[b[0]]+27)
        else:
            #26 because no name can be <start><end>
            probabilities[b] = (count+1)/(letter_counts[b[0]]+26)
    return probabilities

# we run this to check that if we add all the probabilities of one starting letter add
# up to 1
def check_probabilities(probs):
    total = 0
    keys = set() 
    for b in probs.keys():
        keys.add(b[0])
    
    for key in keys:
        for b,p in probs.items():
            if b[0] == key:
                total += p
                
        print(f"{key}:{math.isclose(total,1)}")
        print(total)
        total = 0 

# pick the characters based on their probabilities and generate a name
def generate_names(probs):
    current_char = "<start>"
    characters = []
    weights = []
    name = []
    
    while current_char != "<end>":
        for bigram,prob in probs.items():
            if bigram[0] == current_char:
                characters.append(bigram[1])
                weights.append(prob)
        
        current_char = np.random.choice(characters, p=weights)
        if current_char != "<end>":
            name.append(current_char)
        characters.clear()
        weights.clear()
    name_string = "".join(name)
    return name_string

'''
pick a name form the test dataset and calculate the 
probability of the name based on the training data'''
def evaluation(test,probs):
    name = np.random.choice(test)
    current_char = "<start>"
    total_probability = 1
    for i in range(0,len(name)):
        total_probability *= probs.get((current_char,name[i]),0)
        if i != (len(name)-1):
            current_char = name[i]
        else:
            total_probability *= probs.get((name[i],"<end>"),0)
            
    return name,total_probability

def negative_log_likelyhood(test,probs):
    pass
    
def main():
    
    print("Bigram Language Model")
    train,test = handle_input()
    pairs = get_pairs(train)
    probabilities = calculate_probabilities(count_bigrams(pairs))
    #check_probabilities(probabilities)
    for i in range(10):
        name,probability = evaluation(test,probabilities)
        print(f"Name:{name}, probability:{probability}")

if __name__ == "__main__":
    main()