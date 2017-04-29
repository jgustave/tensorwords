import sys
from keras.models import load_model

from TextGenLearn import TextGenLearn

CUDA_VISIBLE_DEVICES=""
STATIC_ALPHA = ['\n', ' ', '!', '"', '$', '%', '&', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', '>', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#export CUDA_VISIBLE_DEVICES=""
#python3 TextGenPlay.py ./out/weights-improvement-03-2.20.hdf5 0.8


def main() :
    inputPath = sys.argv[1]
    diversity = float(sys.argv[2])
    print("Loading:" + inputPath)
    print("Diversity:{}".format(diversity))
    model = load_model(inputPath)

    prep = TextGenLearn()

    indexToChar = dict((i, c) for i, c in enumerate(STATIC_ALPHA))
    charToIndex = dict((c, i) for i, c in enumerate(STATIC_ALPHA))
    isOneHotInput = True
    seqLen = 10


    while True:
        seedText = input('Seed Text: ').lower()

        text = prep.generateFoo(model, seedText, charToIndex, indexToChar, seqLen, isOneHotInput, diversity)

        print(text)
        print("-"*50)

        pass
    pass


if __name__ == '__main__':
    main()