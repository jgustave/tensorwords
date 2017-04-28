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
        notes = notes[0:10000]

        lines = notes.splitlines()
        random.shuffle(lines)

        alpha = sorted(list(set(notes)))
        alpha.append(START_CHAR)
        alpha.append(END_CHAR)

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
        model.add(LSTM(128, input_shape=(seqLen, numChars)))  # 128 memory units, shape is( sentence_length X num_chars )
        model.add(Dropout(0.5))
        model.add(Dense(numChars))  # Dense output of num_chars with a softmax activation (I think this can be combined)
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)  # Optimizer to use
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer)  # Categorical since we are 1-hot categorical.

        return model

    def gofit(self, model, trainDat,validDat,testDat):

        filepath = "/tmp/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True,
                                     save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001)
        callbacks_list = [checkpoint, early_stopping]

        for iteration in range(1, 60):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            hist = model.fit(trainDat[0], trainDat[1],
                             batch_size=128,
                             # Batch size before backprop. Tradeoff smaller might give better model, Larger might be faster.
                             epochs=3,
                             callbacks=callbacks_list,
                             shuffle=True,
                             validation_data=(validDat[0], validDat[1]) )

            print("Hello")
            print(hist.history)

            # Get loss on test data
            foo = model.evaluate(testDat[0], testDat[1], batch_size=128, verbose=1, sample_weight=None)
            print("Test Loss {}".format(foo))
            print(foo)

            # start_index = random.randint(0, len(text) - maxlen - 1)
            #
            # for diversity in [0.2, 0.5, 1.0, 1.2]:
            #     print()
            #     print('----- diversity:', diversity)
            #
            #     generated = ''
            #     sentence = text[start_index: start_index + maxlen]
            #     generated += sentence
            #     print('----- Generating with seed: "' + sentence + '"')
            #     sys.stdout.write(generated)
            #
            #     for i in range(400):  # Make a 400 character prediction.
            #         x = np.zeros((1, maxlen, len(chars)))  # why the extra dimension of 1 sentence?
            #         for t, char in enumerate(sentence):
            #             x[0, t, char_indices[char]] = 1.
            #
            #         preds = model.predict(x, verbose=0)[0]
            #         next_index = sample(preds, diversity)
            #         next_char = indices_char[next_index]
            #
            #         generated += next_char
            #         sentence = sentence[1:] + next_char
            #
            #         sys.stdout.write(next_char)
            #         sys.stdout.flush()
            #     print()
            #
            # model.save_weights("/tmp/jd-weights-improvement-{epoch:02d}-{loss:.2f}.hdf5")

        pass

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds) #Get in to probability space?
    preds = exp_preds / np.sum(exp_preds) # Normalize to sum to 1.
    # Given the probabilities (preds) which should sum to 1.
    # Roll the dice once (First arg), and return an array with the NUMBER of times the each element was selected at random
    # given the probabilities... Do this (1) time (Third arg)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas) # Returns index of maximum argument.

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

    numChars = len(alpha)

    indexToChar = dict((i, c) for i, c in enumerate(alpha))
    charToIndex = dict((c, i) for i, c in enumerate(alpha))

    # Vectorize
    trainDat = prep.vectorize(trainDat[0], trainDat[1], charToIndex, seqLen, isOneHotInput)
    validDat = prep.vectorize(validDat[0], validDat[1], charToIndex, seqLen, isOneHotInput)
    testDat = prep.vectorize(testDat[0], testDat[1], charToIndex, seqLen, isOneHotInput)

    # Create Model
    model = prep.createModel(seqLen,numChars)
    print(model.summary())
    # train?
    prep.gofit(model,trainDat,validDat,testDat)


if __name__ == '__main__':
    main()
