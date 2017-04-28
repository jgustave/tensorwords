import re
import time
import random
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop

START_CHAR = "["
END_CHAR = "]"


class NotePrep:
    def prepData(self, path, seqLen, stepSize):
        """
        Create test/train/validation data and determine the alphabet.

        :param path: 
        :param seqLen: 
        :param stepSize: 
        :return: 
        """
        notes = open(path).read().lower()
        lines = notes.splitlines()
        random.shuffle(lines)
        alpha = sorted(list(set(notes)))

        # Prepend with start chars.. append with end char
        prep = [START_CHAR * seqLen + x + END_CHAR for x in lines]

        train = prep[0:int(len(prep) * .6)]
        test = prep[len(train):len(train) + int(len(prep) * .2)]
        valid = prep[len(train) + len(test):len(prep)]

        testDat = self.prepSentences(test, seqLen, stepSize)
        validDat = self.prepSentences(valid, seqLen, stepSize)
        trainDat = self.prepSentences(train, seqLen, stepSize)

        return (alpha, trainDat, validDat, testDat)

    def prepSentences(self, lines, seqLen, stepSize):
        """
        Create the sentences and Response character
        :param lines: 
        :param seqLen: 
        :param stepSize: 
        :return: 
        """
        sentences = []
        nextChars = []
        for line in lines:
            for i in range(0, len(line) - seqLen, stepSize):
                sentences.append(line[i:i + seqLen])
                nextChars.append(line[i + seqLen])

        return (sentences, nextChars)

    def vectorize(self, sentences, responses, charToIndex, seqLen, isOneHotInput):
        """
        This sets up a 1-hot encoding for inputs and outputs.
        Will also experiment with a normalized input (ID/numChars)

        :param sentences: 
        :param responses: 
        :param charToIndex: 
        :param seqLen: 
        :param isOneHotInput: 
        :return: 
        """
        numChars = len(charToIndex)  # Number of characters in the alphabet

        if isOneHotInput:
            input = np.zeros((len(sentences), seqLen, numChars), dtype=np.bool)
        else:
            input = np.zeros((len(sentences), seqLen), dtype=np.double)

        output = np.zeros((len(sentences), numChars), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            output[i, charToIndex[responses[i]]] = 1
            for j, char in enumerate(sentence):
                if isOneHotInput:
                    input[i, j, charToIndex[char]] = 1
                else:
                    input[i, j] = charToIndex[char] / numChars

        return (input, output)

    def createModel(self, seqLen, numChars):
        model = Sequential()
        model.add(
            LSTM(128, input_shape=(seqLen, numChars)))  # 128 memory units, shape is( sentence_length X num_chars )
        model.add(Dropout(0.5))
        model.add(
            Dense(numChars))  # Dense output of num_chars with a softmax activation (I think this can be combined)
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)  # Optimizer to use
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer)  # Categorical since we are 1-hot categorical.

        pass


def main():
    print("Hello World")
    prep = NotePrep()
    seqLen = 10
    stepSize = 3
    isOneHotInput = True
    data = prep.prepData("/Users/jerdavis/PycharmProjects/hello/out.txt", seqLen, stepSize)

    alpha = data[0]
    trainDat = data[1]  # Sentence and next char
    validDat = data[2]  # Sentence and next char
    testDat = data[3]  # Sentence and next char

    indexToChar = dict((i, c) for i, c in enumerate(alpha))
    charToIndex = dict((c, i) for i, c in enumerate(alpha))

    # Vectorize
    trainDat = prep.vectorize(trainDat[0], trainDat[1], charToIndex, seqLen, isOneHotInput)
    validDat = prep.vectorize(validDat[0], validDat[1], charToIndex, seqLen, isOneHotInput)
    testDat = prep.vectorize(testDat[0], testDat[1], charToIndex, seqLen, isOneHotInput)

    # Create Model
    model = prep.createModel()

    # train?


if __name__ == '__main__':
    main()
