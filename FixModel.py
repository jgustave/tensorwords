from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

STATIC_ALPHA = ['\n', ' ', '!', '"', '$', '%', '&', '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', '>', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def main() :
    seqLen=10
    numChars=len(STATIC_ALPHA)
    model = Sequential()
    model.add(LSTM(128, input_shape=(seqLen, numChars)))  # 128 memory units, shape is( sentence_length X num_chars )
    model.add(Dropout(0.5))
    model.add(Dense(numChars))  # Dense output of num_chars with a softmax activation (I think this can be combined)
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)  # Optimizer to use
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer)  # Categorical since we are 1-hot categorical.
    model.load_weights("/home/ubuntu/devhome/tensorwords2/out/weights-improvement-00-1.65.hdf5")
    model.save("/home/ubuntu/devhome/tensorwords2/out/fix.hdf5", )

if __name__ == '__main__':
    main()
