# Bigram Model Implementation
from pathlib import Path
from collections import defaultdict
import math

DATA_PATH = Path("data/names.txt")

# first get all names from the txt file without any \n characters
def handle_input():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        names = [name.strip() for name in file]
        return names

# then we get all pairs of letters
def get_pairs(names):
    letter_pairs = []
    for name in names:
        # get the first letter ,with the start word token, and the pair's count in the dictionary
        for i in range(len(name)):
            if i == 0:
                letter_pairs.append(("<start>", name[0]))
            elif i == len(name)-1:
                letter_pairs.append((name[i], "<end>"))
                break
            letter_pairs.append((name[i], name[i+1]))
            
    return letter_pairs

# create a dictionary to store letter pairs and their counts
def count_bigrams(pairs):
    bigrams_dict = defaultdict(int)
    for pair in pairs:
        bigrams_dict[pair] += 1
    return dict(sorted(bigrams_dict.items(), key = lambda kv: kv[1], reverse = True))
    
''' 
calculate the propabilities for every pair by finding the total occurences of a starting letter 
and then find the probability by diving the count of the bigram with the total occurences of the first letter
'''
def calculate_probabilities(bigrams):
    probabilities = defaultdict()
    letter_counts = defaultdict(int)

    for b,count in bigrams.items():
        letter_counts[b[0]] += count
    
    for b,count in bigrams.items():
        probabilities[b] = count/letter_counts[b[0]]
        
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
        total = 0 

def main():
    print("Bigram Language Model")
    names = handle_input()
    pairs = get_pairs(names)
    #print(pairs[:5])
    #print(count_bigrams(pairs))
    
    probabilities = calculate_probabilities(count_bigrams(pairs))
    #print(probabilities)
    check_probabilities(probabilities)
if __name__ == "__main__":
    main()