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
import pdb


class DataSet:
	def __init__(self):
		self.num_of_files = 0
		self.num_of_words = 0
		self.word_freqs = {}

	def prob_word_given_class(self, word, size_of_vocab):
		word_ocurs = self.word_freqs[word] if word in self.word_freqs else 0
		return (word_ocurs + 1)/(size_of_vocab + self.num_of_words)


def readDataset(relative_folder_path, vocab):
	newDataSet = DataSet()
	for filename in os.listdir(os.getcwd() + relative_folder_path):
		newDataSet.num_of_files += 1
		file_words = extract_words_from_file(os.getcwd() + relative_folder_path+ "/"+ filename)
		for word in file_words:
			vocab.add(word)
			newDataSet.num_of_words += 1
			if word not in newDataSet.word_freqs:
				newDataSet.word_freqs[word] = 0
			newDataSet.word_freqs[word] += 1
	return newDataSet


def prob_class_given_test(test, isHam, hamData, spamData, size_of_vocab, ham_prob, spam_prob):
	d_set = spamData
	p_set = spam_prob
	if (isHam):
		d_set = hamData
		p_set = ham_prob
	probs = [d_set.prob_word_given_class(word, size_of_vocab) for word in test]
	return log_sum_of_list(probs)+math.log(p_set)

def log_sum_of_list(l):#using log sum to prevent underflow
	result = 0
	for num in l:
		result += math.log(num)
	return result

def runTests():
	vocab = set()
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

	print(hamData.prob_word_given_class("Chinese", len(vocab)))
	print(hamData.prob_word_given_class("Japan", len(vocab)))
	print(hamData.prob_word_given_class("Tokyo", len(vocab)))

	print(spamData.prob_word_given_class("Chinese", len(vocab)))
	print(spamData.prob_word_given_class("Japan", len(vocab)))
	print(spamData.prob_word_given_class("Tokyo", len(vocab)))

	print(prob_class_given_test(["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"], True, hamData, spamData, len(vocab), prob_ham, prob_spam))
	print(prob_class_given_test(["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"], False, hamData, spamData, len(vocab), prob_ham, prob_spam))

def extract_words_from_file(filename):
	f = open(filename, errors='replace')
	words = []
	for line in f.readlines():
		a_line = line.rstrip("\n\r")
		line_list = a_line.split()
		for word in line_list:
			words.append(word)
	return words

def NB_classify(test_words, isHam, hamData, spamData, size_of_vocab, ham_prob, spam_prob):#returns True if ham false otherwise
	h = prob_class_given_test(test_words, True, hamData, spamData, size_of_vocab, ham_prob, spam_prob)
	s = prob_class_given_test(test_words, False, hamData, spamData, size_of_vocab, ham_prob, spam_prob)
	return h >= s

def NB_readTestSet(isHam, relative_folder_path,hamData, spamData, size_of_vocab, ham_prob, spam_prob):
	correct = 0
	wrong = 0
	for filename in os.listdir(os.getcwd() + relative_folder_path):
		file_words = extract_words_from_file(os.getcwd() + relative_folder_path+ "/"+ filename)
		classifiedHam = NB_classify(file_words, isHam, hamData, spamData, size_of_vocab, ham_prob, spam_prob)
		if (isHam and classifiedHam):
			correct += 1
			#print("Ham email correctly classified!")
		elif (isHam and not classifiedHam):
			wrong += 1
			#print("Ham email wrongly classified!")
		elif (not isHam and classifiedHam):
			wrong += 1
			#print("Spam email wrongly classified!")
		else:
			correct += 1
			#print("Spam email correctly classified!")
	if (isHam):
		print("NAIVE BAYES Ham classification accuracy --> " + str((correct/(correct + wrong))*100) + "%!")
	else:
		print("NAIVE BAYES Spam classification accuracy --> " + str((correct/(correct + wrong))*100) + "%!")



if __name__ == '__main__':
    
	# Get all stop words
	#stop_words = extract_words_from_file("stop_words")

	vocab = set()

	#read Ham Training Data
	hamData = readDataset('/train/ham', vocab)
	spamData = readDataset('/train/spam', vocab)

	prob_ham = hamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)
	prob_spam = spamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)
	#pdb.set_trace()
	NB_readTestSet(True, "/test/ham",hamData, spamData, len(vocab), prob_ham, prob_spam)
	NB_readTestSet(False, "/test/spam",hamData, spamData, len(vocab), prob_ham, prob_spam)

	#testing
	#runTests()












    

    