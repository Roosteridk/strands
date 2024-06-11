import numpy as np
import math


# Maps letter into number between 0 - 25
def abt(letter):
    return ord(letter) - 97


def train_most_probable_matrix(length=18):
    words = dict()
    with open("300k.txt", "r") as file:
        for line in file.readlines():
            word, freq = tuple(line.split("\t"))
            words[word] = int(freq)
    matrix = np.zeros((length, 27, 27), np.float64)

    for word in words.keys():
        if 3 < len(word) <= 18:
            for n, next_letter in enumerate(word[1:]):
                curr_letter = word[n]
                matrix[n][abt(curr_letter)][abt(next_letter)] += words[word]
            matrix[n + 1][abt(word[len(word) - 1])][26] += words[word]
    for n in range(17):
        for curr in range(26):
            log_freqs = np.log(matrix[n][curr])
            print(log_freqs)
            exp_vector = np.exp(log_freqs)
            print(exp_vector)
            prob_vector = np.divide((exp_vector), np.sum(exp_vector))
            print(prob_vector)
            print(np.sum(prob_vector))
            quit()
    return matrix


matrix = train_most_probable_matrix()

# Turn matrix values to probabilities
# for n in range(17):
#     for curr in range(26):
#         log_freqs = np.log(matrix[n][curr])
#         print(log_freqs)
#         exp_vector = np.exp(log_freqs)
#         print(exp_vector)
#         prob_vector = np.divide((exp_vector), np.sum(exp_vector))
#         print(prob_vector)
#         print(np.sum(prob_vector))
#         quit()


# Might be better to continually count as we iterate
# Raid->Raide->Raider, rather than continually
# check Raid, Raide, Raider
def find_string_prob(word):
    return
