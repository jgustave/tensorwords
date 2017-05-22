from keras.callbacks import Callback


class ResetStateCallback(Callback):
    def __init__(self, max_len):
        self.counter = 0
        self.max_len = max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
        self.counter += 1
