import sys
from keras.models import load_model

from TextGenLearn import TextGenLearn
import argparse

START_CHAR = "["
STOP_CHAR = "]"
#CUDA_VISIBLE_DEVICES=""
#STATIC_ALPHA = ['\n', ' ', '!', '"', '$', '%', '&', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', '>', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
STATIC_ALPHA = [' ', '!', '#', ',', '.', '?', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#export CUDA_VISIBLE_DEVICES=""
#python3 TextGenPlay.py ./out/weights-improvement-03-2.20.hdf5 0.8


def main() :
    #print("Hello World")
    parser = argparse.ArgumentParser(description='Do Stuff')
    parser.add_argument('--input', default="./out/2/w-512-05-30-0.96.hdf5")
    parser.add_argument('--diversity', type=float, default=0.8)
    parser.add_argument('--maxlen', type=int, default=1024)
    parser.add_argument('--seqlen', type=int, default=30)
    parser.add_argument('--text', default="hints of")
    args = parser.parse_args()
    #print(args)

    inputPath = args.input
    diversity = args.diversity
    model = load_model(inputPath)

    prep = TextGenLearn()

    indexToChar = dict((i, c) for i, c in enumerate(STATIC_ALPHA))
    charToIndex = dict((c, i) for i, c in enumerate(STATIC_ALPHA))
    isOneHotInput = True

#TODO sanatize etc.
    seedText = args.text.lower()

    result = prep.generateFoo(model, seedText, charToIndex, indexToChar, args.seqlen, isOneHotInput, diversity, args.maxlen)
    s = ""
    for i in range(len(result)):
        if result[i] != '[' : s+=result[i]

    print(s)


if __name__ == '__main__':
    main()