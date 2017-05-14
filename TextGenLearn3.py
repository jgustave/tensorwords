import re
import time
import random
import numpy as np
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, regularizers
from keras.layers import LSTM
from keras.optimizers import RMSprop
import argparse
import gzip

from multi_gpu import make_parallel

START_CHAR = "["
STOP_CHAR = "]"

#python3 TextGenLearn2.py /home/ubuntu/devhome/tensorwords2/wine_corpus.txt /home/ubuntu/devhome/tensorwords2/out2/ 200000
class TextGenLearn:


    def prePrep(self, path, seqLen, maxLines):
        """
        Get the alphabet and the number of sentences (assuming stepSize of 1) 
        :param path: 
        :param seqLen: 
        :param stepSize: 
        :param maxLines: 
        :return: 
        """
        alpha = set()
        numSentences = 0
        lineCount = 0

        with gzip.open(path,'rt') as f:
            for line in f:
                line=line.lower().rstrip()
                # len+1 because of start/stop chars
                numSentences+=len(line)+1
                alpha |= set(line)
                lineCount+=1
                if lineCount >= maxLines:
                    break

        alpha.update(START_CHAR)
        alpha.update(STOP_CHAR)
        alpha = sorted(list(alpha))

        return (alpha, numSentences)

    def vectorizeInputFoo(self, path, numSentences, charToIndex, seqLen):
        numChars = len(charToIndex)  # Number of characters in the alphabet

        with gzip.open(path, 'rt') as f:

            input = np.zeros((numSentences, seqLen, numChars), dtype=np.bool)

            sid = 0
            for line in f:

                #TODO change line in to sentences
                #TODO: handle newline and length -1
                sentences = []
                line = START_CHAR * seqLen + line.rstrip().lower() + STOP_CHAR
                for i in range(0, len(line) - seqLen):
                     sentences.append(line[i:i + seqLen])


                for i, s in enumerate(sentences):
                    for j, char in enumerate(s):
                        input[sid, j, charToIndex[char]] = 1
                    sid=sid+1

                if sid >= numSentences:
                    break

        return input

    def vectorizeInputHelper(self, sentences, charToIndex, seqLen):
        numChars = len(charToIndex)  # Number of characters in the alphabet

        input = np.zeros((len(sentences), seqLen, numChars), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                input[i, j, charToIndex[char]] = 1

        return input


    def vectorizeOutputFoo(self, path, numSentences, charToIndex, seqLen):
        numChars = len(charToIndex)  # Number of characters in the alphabet
        sid = 0

        with gzip.open(path, 'rt') as f:
            output = np.zeros((numSentences, numChars), dtype=np.bool)

            for line in f:
                responses = []
                line = START_CHAR * seqLen + line.rstrip().lower() + STOP_CHAR
                for i in range(0, len(line) - seqLen ):
                    responses.append(line[i + seqLen])

                for i, sentence in enumerate(responses):
                    output[sid, charToIndex[responses[i]]] = 1
                    sid+=1

                if sid >= numSentences:
                    break

        return output

    def createModel(self, seqLen, numChars, lstmSize, numLayers, dropout, learnRate ):
        model = Sequential()

        for i in range(numLayers-1):
            model.add(LSTM(lstmSize,
                           input_shape=(seqLen, numChars),
                           return_sequences=True
                           # kernel_regularizer=regularizers.l2(0.01),
                           # activity_regularizer = regularizers.l2(0.01),
                           # recurrent_regularizer=regularizers.l2(0.01)
                           ))
            model.add(Dropout(dropout))

        if numLayers == 1:
            model.add(LSTM(lstmSize,
                           input_shape=(seqLen, numChars)
                           # kernel_regularizer=regularizers.l2(0.01),
                           # activity_regularizer=regularizers.l2(0.01),
                           # recurrent_regularizer=regularizers.l2(0.01)
            ))
        else:
            model.add(LSTM(lstmSize
                           # kernel_regularizer=regularizers.l2(0.01),
                           # activity_regularizer = regularizers.l2(0.01),
                           # recurrent_regularizer=regularizers.l2(0.01)
            ))

        model.add(Dropout(dropout))
        model.add(Dense(numChars))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=learnRate)  # Optimizer to use
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer)  # Categorical since we are 1-hot categorical.

        return model

    def gofit(self, model, trainDat,validDat,outputPath, nextEpoch, earlyPatience, batchSize):

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
                         batch_size=batchSize,
                         # Batch size before backprop. Tradeoff smaller might give better model, Larger might be faster.
                         epochs=60,
                         callbacks=callbacks_list,
                         shuffle=True,
                         validation_data=(validDat[0], validDat[1]),
                         initial_epoch=nextEpoch)

        print("Hello")
        print(hist.history)

    def generateFoo(self,model,seedText,charToIndex,indexToChar,seqLen, diversity):


        if len(seedText) < seqLen:
            seedText = START_CHAR*(seqLen-len(seedText))+seedText

        maxStrLen = 400

        generated = ''
        sentence = seedText
        generated += sentence

        for i in range(maxStrLen):  # Make a 400 character prediction.

            x= self.vectorizeInputHelper([sentence],charToIndex,seqLen)

            preds       = model.predict(x, verbose=0)[0]
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
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=128)

    #args.lstmSize, args.numLayers, args.dropout, args.learnRate
    args = parser.parse_args()
    print(args)

    prep = TextGenLearn()
    prepData = prep.prePrep(args.input, args.seqlen, args.maxlines)

    numSentences = prepData[1]
    alpha = prepData[0]
    numChars = len(alpha)

    print("Total Sentences {}".format(numSentences))
    cutoff = int(numSentences*0.8)
    print("train/valid cutoff {}".format(cutoff))
    print("Alphabet Size:{}".format(len(alpha)))
    print(alpha)


    indexToChar = dict((i, c) for i, c in enumerate(alpha))
    charToIndex = dict((c, i) for i, c in enumerate(alpha))

    input = prep.vectorizeInputFoo(args.input,numSentences,charToIndex,args.seqlen)
    prepData = prep.vectorizeOutputFoo(args.input, numSentences, charToIndex, args.seqlen)


    inputTrain = input[0:cutoff]
    inputValid = input[cutoff:numSentences]
    responseTrain = prepData[0:cutoff]
    responseValid = prepData[cutoff:numSentences]

    print("Training Data Size:{}".format(len(inputTrain)))
    print("Validation Data Size:{}".format(len(inputValid)))


    # Create Model
    if args.load is None:
        model = prep.createModel(args.seqlen,numChars,args.lstmsize,args.numlayers,args.dropout,args.learnrate)
    else:
        model = load_model(args.load)

    batchSize = args.batchsize
    if args.parallel > 1:
        model = make_parallel(model,args.parallel)
        batchSize *= args.parallel

    print(model.summary())
    # train?
    prep.gofit(model,(inputTrain,responseTrain),(inputValid,responseValid), args.output, args.epoch, args.patience, batchSize)

if __name__ == '__main__':
    main()
