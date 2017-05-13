import numpy as np
import os
import psutil

def main():
    numSentences = 1000000
    seqLen = 10
    numChars = 57

    process = psutil.Process(os.getpid())
    start = process.memory_info().rss
    print(start)

    input = np.zeros((numSentences, seqLen, numChars), dtype=np.bool)

    for s in range(0,numSentences):
        for ss in range(0,seqLen):
            for r in range(0, numChars):
                input[s,ss,r] = 1

    process = psutil.Process(os.getpid())
    end = process.memory_info().rss

    print(end)
    print(end-start)



if __name__ == '__main__':
    main()
