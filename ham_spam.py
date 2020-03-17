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
NB
Get all stop words
Get P(ham) and P(spam) - count of all 
total words in ham
total of each word in ham

P(spam|file) = P(file|spam)P(spam)
p(file|spam) = PI(P(word|spam))
P(ham|file)

LOGREG
each word that ever occurs in each section of files is attribute X and value of that attribute is frequency
each file is a data point with y being ham or spam


'''

import os
import numpy as np
import math
import matplotlib.pyplot as plt 
import pdb
import time

###################################################################
###################################################################
####
####			NAIVE BAYES FUNCTIONS AND CLASSES
####
####
###################################################################
###################################################################

class DataSet:
	def __init__(self):
		self.num_of_files = 0
		self.num_of_words = 0
		self.word_freqs = {}
		#for log regression
		self.docs = []


	def prob_word_given_class(self, word, size_of_vocab):
		word_ocurs = self.word_freqs[word] if word in self.word_freqs else 0
		return (word_ocurs + 1)/(size_of_vocab + self.num_of_words)


def readDataset(relative_folder_path, vocab, remove_SW, stop_words):
	newDataSet = DataSet()
	for filename in os.listdir(os.getcwd() + relative_folder_path):
		newDataSet.num_of_files += 1
		file_words = extract_words_from_file(os.getcwd() + relative_folder_path+ "/"+ filename)
		file_dict = {}
		for word in file_words:
			if (remove_SW and (word in stop_words)):
				continue
			if (word.isalpha()):
				vocab.add(word)
				if word not in file_dict:
					file_dict[word] = 0
				file_dict[word] += 1
				newDataSet.num_of_words += 1
				if word not in newDataSet.word_freqs:
					newDataSet.word_freqs[word] = 0
				newDataSet.word_freqs[word] += 1
		newDataSet.docs.append(file_dict)
	return newDataSet


def prob_class_given_test(test, isHam, hamData, spamData, size_of_vocab, ham_prob, spam_prob):
	d_set = spamData
	p_set = spam_prob
	if (isHam):
		d_set = hamData
		p_set = ham_prob
	probs = [d_set.prob_word_given_class(word, size_of_vocab) for word in test]
	#print(probs)
	return log_sum_of_list(probs)+math.log(p_set)

def log_sum_of_list(l):#using log sum to prevent underflow
	result = 0
	for num in l:
		#print(math.log(num))
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

	X,Y = LG_generate_data(vocab, hamData, spamData,w_2_idx)
	print(X)
	print(Y)

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

def NB_readTestSet(isHam, relative_folder_path,hamData, spamData, size_of_vocab, ham_prob, spam_prob, remove_SW, stop_words):
	correct = 0
	wrong = 0
	for filename in os.listdir(os.getcwd() + relative_folder_path):
		file_words = extract_words_from_file(os.getcwd() + relative_folder_path+ "/"+ filename)
		if (remove_SW):
			file_words = [wd for wd in file_words if (wd not in stop_words)]
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
	return correct,wrong

###################################################################
###################################################################
####
####			LOGISTICAL REGRESSION FUNCTIONS AND CLASSES
####
####
###################################################################
###################################################################

def LG_generate_data(vocab, hamData, spamData,w_2_idx):
	x_array = []
	y_array = []
	num_of_attr = len(vocab)
	doc_to_array(x_array, y_array, num_of_attr, hamData, w_2_idx, True)
	doc_to_array(x_array, y_array, num_of_attr, spamData, w_2_idx, False)
	return x_array,y_array

def doc_to_array(x_array, y_array, num_of_attributes, d_set, w_2_idx, isHam):
	for file_dict in d_set.docs:
		file_arr = [0]*num_of_attributes
		for key,value in file_dict.items():
			file_arr[w_2_idx[key]] = value
		x_array.append(file_arr)
		y_array.append(1 if isHam else 0)

def learn_weights_batch(W, X, Y, learning_eta, reg_lambda, iterations):#X is the 2D array of data pnts by attributes
	new_w  = W[:]
	for j in range(iterations):
		old_w = new_w[:]
		for i in range(len(new_w)):
			sum_term = 0
			for l in range(len(Y)):
				if (X[l][i] == 0):
					sum_term += 0
				else:
					sum_term += X[l][i]*(Y[l] - LG_function(old_w, X[l]))
			reg_term = old_w[i]*reg_lambda
			#print(sum_term)
			#print(reg_term)
			new_w[i] = old_w[i] + learning_eta*(sum_term - reg_term)
	return new_w

def learn_weights_stoch(W, X, Y, learning_eta, reg_lambda, iterations):#X is the 2D array of data pnts by attributes
	new_w  = W[:]
	for j in range(iterations):
		old_w = new_w[:]
		for l in range(len(Y)):
			for i in range(len(new_w)):
				err_term = X[l][i]*(Y[l] - LG_function(old_w, X[l]))
				reg_term = (old_w[i]**2)*reg_lambda
				new_w[i] = old_w[i] + learning_eta*(err_term - reg_term)
	return new_w

def LG_function(W, X, w_o=0.00):#X is a single data point
	expo = expo_term(W, w_o, X)
	return (expo/(1 + expo))

def expo_term(W, w_o, X):#X is a single data point
	return math.exp(np.dot(W, X) + w_o)

def LG_classify(learned_W, test_X, w_o=0.00):
	sum_term = w_o + np.dot(learned_W, test_X)
	return sum_term > 0

def LG_readTestSet(isHam, relative_folder_path, w_2_idx, num_of_attr, learned_W, remove_SW, stop_words):
	correct = 0
	wrong = 0
	for filename in os.listdir(os.getcwd() + relative_folder_path):
		file_words = extract_words_from_file(os.getcwd() + relative_folder_path+ "/"+ filename)
		if (remove_SW):
			file_words = [wd for wd in file_words if (wd not in stop_words)]
		test_X = [0]*num_of_attr
		for word in file_words:
			if word in w_2_idx:
				test_X[w_2_idx[word]] += 1
		classifiedHam = LG_classify(learned_W, test_X)
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
		print("LOGISTICAL REGRESSION Ham classification accuracy --> " + str((correct/(correct + wrong))*100) + "%!")
	else:
		print("LOGISTICAL REGRESSION Spam classification accuracy --> " + str((correct/(correct + wrong))*100) + "%!")
	return correct,wrong

if __name__ == '__main__':
    
	# Get all stop words
	stop_words = extract_words_from_file("stop_words")

	vocab = set()

	###############		NB WITH stop words  ########################
	#read Ham Training Data
	hamData = readDataset('/train/ham', vocab, False, stop_words)
	spamData = readDataset('/train/spam', vocab, False, stop_words)

	prob_ham = hamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)
	prob_spam = spamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)
	#pdb.set_trace()
	ham_correct,ham_wrong = NB_readTestSet(True, "/test/ham",hamData, spamData, len(vocab), prob_ham, prob_spam, False, stop_words)
	spam_correct,spam_wrong = NB_readTestSet(False, "/test/spam",hamData, spamData, len(vocab), prob_ham, prob_spam, False, stop_words)
	print("NAIVE BAYES WITH STOP WORDS COMBINED classification accuracy --> " + str(((ham_correct + spam_correct)/(spam_correct + spam_wrong + ham_correct + ham_wrong))*100) + "%!")



	
	#testing functions
	#runTests()

	w_2_idx = {word:i for i,word in enumerate(vocab)}
	X,Y = LG_generate_data(vocab, hamData, spamData,w_2_idx)
	W = [0.01]*len(vocab)
	#Learning weights settings
	learning_eta = 0.001
	reg_lambda = [1000.0 ,100.0 ,10.0, 1, 0.1]
	iterations = 20

	for lamb in reg_lambda:
		print("############ Lamba " + str(lamb) + "######################")
		print("########################################################")
		#print(len(W))
		start = time.time()
		learned_W = learn_weights_batch(W, X, Y, learning_eta, lamb, iterations)
		end = time.time()
		print(str(iterations) + " took " + str(end-start) + " seconds")

		ham_correct,ham_wrong = LG_readTestSet(True, "/test/ham", w_2_idx, len(vocab), learned_W, False, stop_words)
		spam_correct,spam_wrong = LG_readTestSet(False, "/test/spam", w_2_idx, len(vocab), learned_W, False, stop_words)
		print("LOGISTICAL REGRESSION WITH STOP WORDS COMBINED classification accuracy --> " + str(((ham_correct + spam_correct)/(spam_correct + spam_wrong + ham_correct + ham_wrong))*100) + "%!")
		print("\n\n")
		

	###############		NB WITHOUT stop words  ########################
	vocab = set()
	#read Ham Training Data
	hamData = readDataset('/train/ham', vocab, True, stop_words)
	spamData = readDataset('/train/spam', vocab, True, stop_words)

	prob_ham = hamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)
	prob_spam = spamData.num_of_files/(hamData.num_of_files + spamData.num_of_files)
	#pdb.set_trace()
	ham_correct,ham_wrong = NB_readTestSet(True, "/test/ham",hamData, spamData, len(vocab), prob_ham, prob_spam, True, stop_words)
	spam_correct,spam_wrong = NB_readTestSet(False, "/test/spam",hamData, spamData, len(vocab), prob_ham, prob_spam, True, stop_words)
	print("NAIVE BAYES WITHOUT STOP WORDS COMBINED classification accuracy --> " + str(((ham_correct + spam_correct)/(spam_correct + spam_wrong + ham_correct + ham_wrong))*100) + "%!")


	###############		LR WITHOUT stop words  ########################

	w_2_idx = {word:i for i,word in enumerate(vocab)}
	X,Y = LG_generate_data(vocab, hamData, spamData,w_2_idx)
	W = [0.01]*len(vocab)
	#Learning weights settings
	learning_eta = 0.001
	reg_lambda = [1000.0 ,100.0 ,10.0, 1, 0.1]
	iterations = 20

	for lamb in reg_lambda:
		print("############ Lamba " + str(lamb) + "######################")
		print("########################################################")
		#print(len(W))
		start = time.time()
		learned_W = learn_weights_batch(W, X, Y, learning_eta, lamb, iterations)
		end = time.time()
		print(str(iterations) + " took " + str(end-start) + " seconds")

		ham_correct,ham_wrong = LG_readTestSet(True, "/test/ham", w_2_idx, len(vocab), learned_W, True, stop_words)
		spam_correct,spam_wrong = LG_readTestSet(False, "/test/spam", w_2_idx, len(vocab), learned_W, True, stop_words)
		print("LOGISTICAL REGRESSION WITHOUT STOP WORDS COMBINED classification accuracy --> " + str(((ham_correct + spam_correct)/(spam_correct + spam_wrong + ham_correct + ham_wrong))*100) + "%!")
		print("\n\n")
		









    

    