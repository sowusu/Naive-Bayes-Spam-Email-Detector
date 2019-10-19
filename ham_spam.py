# decision_tree.py
# ---------
#
#
# INSTRUCTIONS:
# ------------
# 1. 
#
# 2. 
#
# 3. 
#
# 4. 


'''
Get all stop words
Get P(ham) and P(spam) - count of all 
total words in ham
total of each word in ham

P(spam|file) = P(file|spam)P(spam)
p(file|spam) = PI(P(word|spam))
P(ham|file)

'''

import os
import numpy as np
import math
import matplotlib.pyplot as plt 


class DataSet:
	def __init__(self):
		self.num_of_files = 0
		self.num_of_words = 0
		self.word_freqs = {}

	def prob_word_given_class(word, size_of_vocab):
		word_ocurs = self.word_freqs[word] if word in self.word_freqs else 0

		return (word_ocurs + 1)/(size_of_vocab + self.num_of_words)





def readDataset(relative_folder_path, vocab):
	newDataSet = DataSet()
	for filename in os.listdir(os.getcwd() + relative_folder_path):
		newDataSet.num_of_files += 1
		f = open(os.getcwd() + relative_folder_path+ "/"+ filename)
		for line in f.readlines():
			a_line = line.rstrip("\n\r")
			line_list = a_line.split()
			for word in line_list:
				vocab.add(word)
				newDataSet.num_of_words += 1
				if word not in newDataSet.word_freqs:
					newDataSet.word_freqs[word] = 0
				newDataSet.word_freqs[word] += 1
	return newDataSet


def prob_class_given_test(test, isHam, hamData, spamData, size_of_vocab):
	d_set = spamData
	if (isHam):
		d_set = hamData
	p = 1
	probs = [d_set.prob_class_given_test(word, size_of_vocab) for word in test]
	return p



if __name__ == '__main__':
    
    # Get all stop words
    f = open("stop_words")
    f_lines = f.readlines()
    stop_words = [line.rstrip("\n\r") for line in f_lines]

    vocab = set()
    #read Ham Training Data
    #hamData = readDataset('/train/ham', vocab)
    #spamData = readDataset('/train/spam', vocab)

    hamData = readDataset('/hamtest', vocab)
    spamData = readDataset('/spamtest', vocab)

    prob_ham = hamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)
    prob_spam = spamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)

    print(hamData.num_of_files)
    print(hamData.num_of_words)
    print(hamData.word_freqs)

    print(spamData.num_of_files)
    print(spamData.num_of_words)
    print(spamData.word_freqs)

    print(prob_ham)
    print(prob_spam)








    

    