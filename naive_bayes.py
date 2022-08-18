from cgi import test
import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
import heapq
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
import math
import config




def loadData(path):
    '''
    reads data from the folders
    x_train : [review1, review2, ....., review_n], where each review1 is a list of tokens
    
    vocabulary is a dictionary: (key: word, value: count)
    '''
    # --------your code here-------------
    
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    vocabulary = {}

    test_neg = path + "test_set/neg/"
    test_pos = path + "test_set/pos/"
    train_neg = path + "training_set/neg/"
    train_pos = path + "training_set/pos/"

    for filename in os.listdir(train_neg):
        f = os.path.join(train_neg, filename)
        if os.path.isfile(f):
            with open (f, "r") as myfile:
                text = myfile.readlines()
                temp_list = []
                for line in text:
                    tokenized_text = tokenizer.tokenize(line)
                    for word in tokenized_text:
                        if word not in stop_words:
                            stemmed_word = ps.steam(word)
                            temp_list.append(stemmed_word)
                            if stemmed_word in vocabulary:
                                vocabulary[stemmed_word] += 1
                            else:
                                vocabulary[stemmed_word] = 1
                x_train.append(temp_list)
                y_train.append(-1)

    for filename in os.listdir(train_pos):
        f = os.path.join(train_pos, filename)
        if os.path.isfile(f):
            with open (f, "r") as myfile:
                text = myfile.readlines()
                temp_list = []
                for line in text:
                    tokenized_text = tokenizer.tokenize(line)
                    for word in tokenized_text:
                        if word not in stop_words:
                            stemmed_word = ps.steam(word)
                            temp_list.append(stemmed_word)
                            if stemmed_word in vocabulary:
                                vocabulary[stemmed_word] += 1
                            else:
                                vocabulary[stemmed_word] = 1
                x_train.append(temp_list)
                y_train.append(1)

    for filename in os.listdir(test_neg):
        f = os.path.join(test_neg, filename)
        if os.path.isfile(f):
            with open (f, "r") as myfile:
                text = myfile.readlines()
                temp_list = []
                for line in text:
                    tokenized_text = tokenizer.tokenize(line)
                    for word in tokenized_text:
                        if word not in stop_words:
                            stemmed_word = ps.steam(word)
                            temp_list.append(stemmed_word)
                            if stemmed_word in vocabulary:
                                vocabulary[stemmed_word] += 1
                            else:
                                vocabulary[stemmed_word] = 1
                x_test.append(temp_list)
                y_test.append(-1)

    for filename in os.listdir(test_pos):
        f = os.path.join(test_pos, filename)
        if os.path.isfile(f):
            with open (f, "r") as myfile:
                text = myfile.readlines()
                temp_list = []
                for line in text:
                    tokenized_text = tokenizer.tokenize(line)
                    for word in tokenized_text:
                        if word not in stop_words:
                            stemmed_word = ps.steam(word)
                            temp_list.append(stemmed_word)
                            if stemmed_word in vocabulary:
                                vocabulary[stemmed_word] += 1
                            else:
                                vocabulary[stemmed_word] = 1
                x_test.append(temp_list)
                y_test.append(1)
    
    return x_train, x_test, y_train, y_test, vocabulary

def getBOWRepresentation(x_train, x_test, vocabulary):
    '''
    converts into Bag of Words representation
    each column is a feature(unique word) from the vocabulary 
    x_train_bow : a numpy array with bag of words representation
    '''
    # --------your code here-------------

    vocabulary = list(vocabulary)
    x_train_bow = []
    for text_data in x_train:
        temp_list = [0]*len(vocabulary)
        for word in text_data:
            if word in vocabulary:
                index = vocabulary.index(word)
            else:
                index = 100
            temp_list[index] += 1
        x_train_bow.array(temp_list)
    x_train_bow = np.array(x_train_bow)

    x_test_bow = []
    for text_data in x_test:
        temp_list = [0]*len(vocabulary)
        for word in text_data:
            if word in vocabulary:
                index = vocabulary.index(word)
            else:
                index = 100
            temp_list[index] += 1
        x_test_bow.array(temp_list)
    x_test_bow = np.array(x_test_bow)


    return x_train_bow, x_test_bow

def naiveBayesMulFeature_train(Xtrain, ytrain):
    # --------your code here-------------
    thetaPos = []
    thetaNeg = []
    total_words_pos = 0
    total_words_neg = 0
    dict_count_pos = [0]*len(Xtrain[0])
    dict_count_neg = [0]*len(Xtrain[0])
    for a, label in zip(Xtrain, ytrain):
        for j in range(len(a)):
            if(label == 1):
                total_words_pos = total_words_pos + a[j]
                dict_count_pos[j] = dict_count_pos[j] + a[j]
            elif(label == -1):
                total_words_neg = total_words_neg + a[j]
                dict_count_neg[j] = dict_count_neg[j] + a[j]
    for i in range(len(Xtrain[0])):
        thetaPos.append((dict_count_pos[i] + 1.0)/(total_words_pos + len(Xtrain[0])) * 1.0)
        thetaNeg.append((dict_count_neg[i] + 1.0)/(total_words_neg + len(Xtrain[0])) * 1.0)
    
    return thetaPos, thetaNeg

def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    # --------your code here-------------
    yPredict = []
    accurate_count = 0
    P_Pos = math.log(0.5)
    P_Neg = math.log(0.5)

    for a, label in zip(Xtest,ytest):
        P_Pos_a = P_Pos
        P_Neg_a = P_Neg
        for i in range(len(Xtest[0])):
            P_Pos_a = P_Pos_a + math.log(thetaPos[i]) * a[i]
            P_Neg_a = P_Neg_a + math.log(thetaNeg[i]) * a[i]
        if(P_Pos_a > P_Neg_a):
            yPredict.append(1)
            if(label == 1): accurate_count = accurate_count + 1
        else:
            yPredict.append(-1)
            if(label == -1): accurate_count = accurate_count + 1
    Accuracy = accurate_count/len(Xtest) * 1.0
    
    return yPredict, Accuracy

def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    # --------your code here-------------
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    Accuracy = clf.score(Xtest,ytest)
    return Accuracy

def naiveBayesBernFeature_train(Xtrain, ytrain):
    # --------your code here-------------
    thetaNegTrue = [0] * len(Xtrain[0])
    thetaPosTrue = [0] * len(Xtrain[0])
    for i in range(len(Xtrain[0])):
        count = 1 # num of files which include Wi and are in Class Neg
        for j in range ( int(len(Xtrain)/2) ):
            if(Xtrain[j][i] != 0):
                count = count + 1
        # count/(num of files in class Neg + 2)
        thetaNegTrue[i] = float(count/(len(Xtrain)/2 + 2))

        count = 1 # num of files which include Wi and are in Class Pos
        for j in range ( int(len(Xtrain)/2), len(Xtrain)):
            if(Xtrain[j][i] != 0):
                count = count + 1
        # count/(num of files in class Pos + 2)
        thetaPosTrue[i] = float(count/(len(Xtrain)/2 + 2))
    
    return thetaPosTrue, thetaNegTrue

def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    # --------your code here-------------
    yPredict = []
    accurate_count = 0
    for i in range(len(Xtest)):

        pos_score = 0
        neg_score = 0
        """
        pos_score = 1
        neg_score = 1
        """
        for j in range(len(Xtest[i])):
            if(Xtest[i][j] == 0 ):

                pos_score = pos_score + math.log(1-thetaPosTrue[j])
                neg_score = neg_score + math.log(1-thetaNegTrue[j])
                """
                pos_score = pos_score * (1-thetaPosTrue[j])
                neg_score = neg_score * (1-thetaNegTrue[j])
                """
            else:

                pos_score = pos_score + math.log(thetaPosTrue[j])
                neg_score = neg_score + math.log(thetaNegTrue[j])
                """
                pos_score = pos_score * thetaPosTrue[j]
                neg_score = neg_score * thetaNegTrue[j]
                """
        if(pos_score >neg_score):
            yPredict.append(1)
            if(ytest[i] == 1):
                accurate_count = accurate_count+1
        else:
            yPredict.append(-1)
            if(ytest[i] == -1):
                accurate_count = accurate_count+1

    Accuracy = float(accurate_count/len(ytest))
    return yPredict, Accuracy


if __name__=="__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    


    # read the data and build vocabulary from training data
    # XtrainText is a list of lists: each document is represented by a list of tokens, this
    # function should include the stemming, preprocessing etc.
    # remember to add a UNK to represent out of vocabulary terms
    XtrainText, XtestText, ytrain, ytest, vocabulary = loadData(textDataSetsDirectoryFullPath)


    # let's look at the vocab 

    print("number of unique words: ", len(vocabulary))
    print("the most common 10 words were:", heapq.nlargest(10, vocabulary, key=vocabulary.get))
    print("the least common 10 words were:", heapq.nsmallest(10, vocabulary, key=vocabulary.get))
    vocabulary = dict((word, index) for word, index in vocabulary.items() if \
                    vocabulary[word]>=config.MIN_FREQ and word in heapq.nlargest(config.MAX_VOCAB, vocabulary, key=vocabulary.get))
    
    print("number of unique words in vocabulary: ", len(vocabulary))




    # get BOW representation in the form of numpy arrays
    
    Xtrain, Xtest = getBOWRepresentation(XtrainText, XtestText, vocabulary=vocabulary.keys()) 
        
    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    
    print("--------------------")
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)


    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")
    print("--------------------")