import sys
from keras.models import load_model

from TextGenLearn import TextGenLearn
import argparse

CUDA_VISIBLE_DEVICES=""
#STATIC_ALPHA = ['\n', ' ', '!', '"', '$', '%', '&', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', '>', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
STATIC_ALPHA = [' ', '!', '#', ',', '.', '?', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#export CUDA_VISIBLE_DEVICES=""
#python3 TextGenPlay.py ./out/weights-improvement-03-2.20.hdf5 0.8

#192 ok
def main() :
    print("Hello World")
    parser = argparse.ArgumentParser(description='Do Stuff')
    parser.add_argument('--input', default="./out/weights-improvement-03-2.20.hdf5")
    parser.add_argument('--diversity', type=float, default=0.8)
    parser.add_argument('--maxlen', type=int, default=1024)
    parser.add_argument('--seqlen', type=int, default=10)
    args = parser.parse_args()
    print(args)

    inputPath = args.input
    diversity = args.diversity
    model = load_model(inputPath)

    prep = TextGenLearn()

    indexToChar = dict((i, c) for i, c in enumerate(STATIC_ALPHA))
    charToIndex = dict((c, i) for i, c in enumerate(STATIC_ALPHA))
    isOneHotInput = True


    while True:
        seedText = input('Seed Text: ').lower()

        text = prep.generateFoo(model, seedText, charToIndex, indexToChar, args.seqlen, isOneHotInput, diversity, args.maxlen)

        print(text)
        print("-"*50)

        pass
    pass


if __name__ == '__main__':
    main()