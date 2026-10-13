# Bigram Model Implementation
from pathlib import Path
from collections import defaultdict
DATA_PATH = Path("data/names.txt")

# first get all names from the txt file without any \n characters
def handle_input():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        names = [name.strip() for name in file]
        return names

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

def count_bigrams(pairs):
    bigrams_dict = defaultdict(int)
    for pair in pairs:
        bigrams_dict[pair] += 1
    return bigrams_dict
    

def main():
    print("Bigram Language Model")
    names = ['kostas','ilias','nikos','kosmas']
    pairs = get_pairs(names)
    print(pairs)
    print(count_bigrams(pairs))

if __name__ == "__main__":
    main()