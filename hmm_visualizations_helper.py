# for helping in visualize the HMM

from collections import Counter
import numpy as np
from string import punctuation
import matplotlib.pyplot as plt

# takes in some text and outputs a list with the top ten most common words
def top_ten_words(text):
    count = Counter()
    lst = text.split()
    count.update(word.lower().rstrip(punctuation) for word in lst)
    
    # return a list with top ten most common words from each 
    return [x for x in count.most_common(10)]

# takes in a list of text (corresponding to each state) and outputs
# the top ten from each, along with bar graphs
def plot_top_ten(sentences):
    most_freq = []
    for i in range(len(sentences)):
        x = []
        y = []
        top_ten = top_ten_words(sentences[i])
        # print top 10 words
        print("The top 10 words for state " + str(i) + " are:")
        for j in range(len(top_ten)):
            print(top_ten[j][0] + ": " + str(top_ten[j][1]))
            x.append(top_ten[j][0])
            y.append(top_ten[j][1])
        print()

        # plot top 10 words
        plt.figure(i)
        x_pos = [i for i, _ in enumerate(x)]
        plt.bar(x_pos, y)
        plt.xlabel("Word")
        plt.ylabel("Frequency")
        title = "Top 10 Words in State " + str(i) + " By Frequency"
        plt.title(title)
        plt.xticks(x_pos, x)
        plt.show()