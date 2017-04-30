import re
import time
import random
import numpy as np
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop
import argparse
import gzip



START_CHAR = "["
STOP_CHAR = "]"

#python3 TextGenLearn2.py /home/ubuntu/devhome/tensorwords2/wine_corpus.txt /home/ubuntu/devhome/tensorwords2/out2/ 200000
class TextGenLearn:

    def prepData(self, path, seqLen, stepSize, maxLines):
        """
        Create test/train/validation data and determine the alphabet.

        :param path: 
        :param seqLen: 
        :param stepSize: 
        :return: 
        """
        notes = gzip.open(path,'rt').read().lower()

        lines = notes.splitlines()
        random.shuffle(lines)

        if maxLines < len(lines):
            lines = lines[0:maxLines]

        temp = list(set(notes))
        temp.append(START_CHAR)
        temp.append(STOP_CHAR)
        alpha = sorted( temp )
        print(alpha)
        # Prepend with start chars.. append with end char
        prep = [START_CHAR * seqLen + x + STOP_CHAR for x in lines]

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
        input  = self.vectorizeInput(sentences, charToIndex, seqLen, isOneHotInput)
        output = self.vectorizeOutput(responses, charToIndex)
        return( input,output)

    def vectorizeInput(self, sentences, charToIndex, seqLen, isOneHotInput):
        numChars = len(charToIndex)  # Number of characters in the alphabet

        if isOneHotInput:
            input = np.zeros((len(sentences), seqLen, numChars), dtype=np.bool)
        else:
            input = np.zeros((len(sentences), seqLen), dtype=np.double)

        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                if isOneHotInput:
                    input[i, j, charToIndex[char]] = 1
                else:
                    input[i, j] = charToIndex[char] / numChars

        return input

    def vectorizeOutput(self, responses, charToIndex):
        numChars = len(charToIndex)  # Number of characters in the alphabet

        output = np.zeros((len(responses), numChars), dtype=np.bool)

        for i, sentence in enumerate(responses):
            output[i, charToIndex[responses[i]]] = 1

        return output

    def createModel(self, seqLen, numChars, lstmSize, numLayers, dropout, learnRate ):
        model = Sequential()

        for i in range(numLayers-1):
            model.add(LSTM(lstmSize, input_shape=(seqLen, numChars), return_sequences=True ))
            model.add(Dropout(dropout))

        if numLayers == 1:
            model.add(LSTM(lstmSize, input_shape=(seqLen, numChars)))
        else:
            model.add(LSTM(lstmSize))

        model.add(Dropout(dropout))
        model.add(Dense(numChars))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=learnRate)  # Optimizer to use
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer)  # Categorical since we are 1-hot categorical.

        return model

    def gofit(self, model, trainDat,validDat,testDat, outputPath, nextEpoch, earlyPatience):

        filepath = outputPath + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=False,
                                     save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', patience=earlyPatience, min_delta=0.0001, mode='auto', verbose=1)
        callbacks_list = [checkpoint, early_stopping]

        #for iteration in range(1, 60):
        print()
        print('-' * 50)
        #print('Iteration', iteration)
        hist = model.fit(trainDat[0], trainDat[1],
                         batch_size=128,
                         # Batch size before backprop. Tradeoff smaller might give better model, Larger might be faster.
                         epochs=60,
                         callbacks=callbacks_list,
                         shuffle=True,
                         validation_data=(validDat[0], validDat[1]),
                         initial_epoch=nextEpoch)

        print("Hello")
        print(hist.history)

        # Get loss on test data
        foo = model.evaluate(testDat[0], testDat[1], batch_size=128, verbose=1, sample_weight=None)
        print("Test Loss {}".format(foo))
        print(foo)

        pass

    def generate(self,model,seedText,charToIndex,indexToChar,seqLen,isOneHotInput):
        #start_index = random.randint(0, len(text) - maxlen - 1)

        #TODO prepend seedText with START_CHAR as needed.
        if len(seedText) < seqLen:
            seedText = START_CHAR*(seqLen-len(seedText))+seedText



        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = seedText
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):  # Make a 400 character prediction.

                x= self.vectorizeInput([sentence],charToIndex,seqLen,isOneHotInput)

                preds       = model.predict(x, verbose=0)[0] #First and only prediction
                nextIndex   = sample(preds, diversity)
                nextChar    = indexToChar[nextIndex]

                if nextChar==STOP_CHAR:
                    break

                generated += nextChar
                sentence = sentence[1:] + nextChar

                sys.stdout.write(nextChar)
                sys.stdout.flush()
            print()

        pass

    def generateFoo(self,model,seedText,charToIndex,indexToChar,seqLen,isOneHotInput, diversity):


        if len(seedText) < seqLen:
            seedText = START_CHAR*(seqLen-len(seedText))+seedText

        maxStrLen = 400

        generated = ''
        sentence = seedText
        generated += sentence

        for i in range(maxStrLen):  # Make a 400 character prediction.

            x= self.vectorizeInput([sentence],charToIndex,seqLen,isOneHotInput)

            preds       = model.predict(x, verbose=0)[0] #First and only prediction
            nextIndex   = sample(preds, diversity)
            nextChar    = indexToChar[nextIndex]

            if nextChar==STOP_CHAR:
                break

            generated += nextChar
            sentence = sentence[1:] + nextChar

        return generated

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
    #"/Users/jerdavis/PycharmProjects/hello/out.txt"
    print("Hello World")
    parser = argparse.ArgumentParser(description='Do Stuff')
    parser.add_argument('--input', default = "./wine_corpus.txt.gz")
    parser.add_argument('--output',default="./out")
    parser.add_argument('--maxlines',type=int,default=2000 )
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--load', default=None)
    parser.add_argument('--seqlen', type=int, default=10)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--onehot', type=bool, default=True)
    parser.add_argument('--lstmsize', type=int, default=128)
    parser.add_argument('--numlayers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learnrate', type=float, default=0.01)
    parser.add_argument('--early', type=int, default=3)

    #args.lstmSize, args.numLayers, args.dropout, args.learnRate
    args = parser.parse_args()
    print(args)

    inputPath       = args.input
    outputPath      = args.output
    maxLines        = args.maxlines
    loadModelName   = args.load
    nextEpoch       = args.epoch

    print("InputPath:" + inputPath)
    print("OutputPath:" + outputPath)
    print("MaxLines:{}".format( maxLines) )

    prep = TextGenLearn()
    seqLen = args.seqlen
    stepSize = args.step
    isOneHotInput = args.onehot
    data = prep.prepData(inputPath, seqLen, stepSize, maxLines)

    alpha = data[0]
    trainDat = data[1]  # Sentence and next char
    validDat = data[2]  # Sentence and next char
    testDat = data[3]  # Sentence and next char

    print("Alphabet Size:{}".format(len(alpha)))
    print("Training Data Size:{}".format(len(trainDat[0])))
    print("Validation Data Size:{}".format(len(validDat[0])))
    print("Test Data Size:{}".format(len(testDat[0])))

    numChars = len(alpha)

    indexToChar = dict((i, c) for i, c in enumerate(alpha))
    charToIndex = dict((c, i) for i, c in enumerate(alpha))

    # Vectorize
    trainDat = prep.vectorize(trainDat[0], trainDat[1], charToIndex, seqLen, isOneHotInput)
    validDat = prep.vectorize(validDat[0], validDat[1], charToIndex, seqLen, isOneHotInput)
    testDat = prep.vectorize(testDat[0], testDat[1], charToIndex, seqLen, isOneHotInput)

    # Create Model
    if loadModelName is None:
        model = prep.createModel(seqLen,numChars,args.lstmsize,args.numlayers,args.dropout,args.learnrate)
    else:
        model = load_model(loadModelName)

    print(model.summary())
    # train?
    prep.gofit(model,trainDat,validDat,testDat, outputPath, nextEpoch, arg.early)

    prep.generate(model, "a bold", charToIndex, indexToChar, seqLen, isOneHotInput)
    #prep.generate(model,"a wine wit",charToIndex,indexToChar,seqLen,isOneHotInput)

if __name__ == '__main__':
    main()
