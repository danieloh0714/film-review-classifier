import math
import re


class Bayes_Classifier:

    def __init__(self):
        self.p_count_dict = dict()  # Dictionary of (word, number of times word is in a positive review).
        self.n_count_dict = dict()  # Dictionary of (word, number of times word is in a negative review).
        self.p_freqs = 0  # For every unique word, sum of count(word, pos).
        self.n_freqs = 0  # For every unique word, sum of count(word, neg).
        self.V = 0  # Total number of unique words.
        self.p_pos = 0  # P(pos).
        self.p_neg = 0  # P(neg).

    def train(self, lines):
        for line in lines:
            fields = line.split('|')
            sentiment = fields[0]
            if sentiment == '5':
                self.p_pos += 1
            words = self.tokenize(fields[2])
            for word in words:
                if word not in self.p_count_dict and word not in self.n_count_dict:
                    self.V += 1
                if sentiment == '5':
                    self.p_freqs += 1
                    if word not in self.p_count_dict:
                        self.p_count_dict[word] = 0
                    self.p_count_dict[word] += 1
                else:  # if sentiment == '1'
                    self.n_freqs += 1
                    if word not in self.n_count_dict:
                        self.n_count_dict[word] = 0
                    self.n_count_dict[word] += 1

        self.p_pos = self.p_pos / len(lines)
        self.p_neg = 1 - self.p_pos

    def classify(self, lines):
        sentiments = []

        for line in lines:
            fields = line.split('|')
            words = self.tokenize(fields[2])
            p_pos = math.log(self.p_pos)
            p_neg = math.log(self.p_neg)
            for word in words:
                if word not in self.p_count_dict:
                    p_pos += math.log(1 / (self.p_freqs + self.V))
                else:
                    p_pos += math.log((self.p_count_dict[word] + 1) / (self.p_freqs + self.V))
                if word not in self.n_count_dict:
                    p_neg += math.log(1 / (self.n_freqs + self.V))
                else:
                    p_neg += math.log((self.n_count_dict[word] + 1) / (self.n_freqs + self.V))
            if p_pos > p_neg:
                sentiments.append('5')
            else:
                sentiments.append('1')

        return sentiments

    def tokenize(self, line):
        improved_line = line.replace('\n', '')
        improved_line = improved_line.replace('!', '')
        improved_line = improved_line.replace('?', '')
        improved_line = improved_line.replace('.', '')
        improved_line = improved_line.replace(',', '')
        improved_line = improved_line.replace('(', '')
        improved_line = improved_line.replace(')', '')
        improved_line = improved_line.replace('(', '')
        improved_line = improved_line.replace('-', '')
        improved_line = improved_line.replace('\'', '')
        improved_line = improved_line.replace('"', '')
        improved_line = improved_line.replace('-', '')
        improved_line = improved_line.replace('&', '')
        improved_line = improved_line.replace('and', '')
        improved_line = improved_line.replace('the', '')
        improved_line = improved_line.replace('it', '')
        improved_line = improved_line.lower()
        words = improved_line.split(' ')
        return words
