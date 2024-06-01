import numpy as np


# Maps letter into number between 0 - 25
def abt(letter):
    return ord(letter) - 97


words = dict()
zeProb = 0
with open("300k.txt", "r") as file:
    for line in file.readlines():
        word, freq = tuple(line.split("\t"))
        words[word] = int(freq)
        if len(word) > 1 and word[0] == "z" and word[1] == "e":
            zeProb += int(freq)
print(zeProb)
matrix = np.zeros((27, 27, 18), np.float64)

for word in words.keys():
    if len(word) <= 18:
        if len(word) > 1 and word[0] == "z" and word[1] == "e":
            zeProb -= words[word]
        for n, next_letter in enumerate(word[1:]):
            curr_letter = word[n]
            matrix[abt(curr_letter)][abt(next_letter)][n] += words[word]
        matrix[abt(word[len(word) - 1])][26][n + 1] += words[word]

print(matrix[0][0][0])
print(zeProb)
print(matrix[25][4][0])
