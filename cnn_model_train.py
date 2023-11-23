import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D,Flatten,MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix
import itertools
from keras.callbacks import Callback
from IPython.display import clear_output

def read_csv_data(csv_filename, header=None, num_funcs=6, shuffle_data=True):
    """helper function to read data from csv file
    return X and Y
    X.shape = (num_samples, length_seq, num_funcs)
    Y.shape = (num_samples, num_classes)
    """
    # read data
    data = pd.read_csv(csv_filename, header=header)
    data = data.to_numpy()
    if (shuffle_data):
        data = shuffle(data)

    # extract X and Y
    X = data[:, :-1]#.reshape((-1, int((data.shape[1]-1)/num_funcs), num_funcs))
    X = data[:, :-1].reshape(X.shape[0], 1, 24, 6)
    Y = data[:, -1]

    # convert Y to one-hot vectors
    Y = to_categorical(Y, num_classes=Y.max()+1)

    return X, Y

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        #self.plot_history()

    def plot_history(self):
        #f, ax1 = plt.subplots()
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        clear_output(wait=True)
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        plt.show()

def train(X_train,Y_train,X_test,Y_test,plot_learning):
    input_shape = (1, 24, 6)
    model = Sequential()

    num_classes=7
    model.add(Conv2D(32, kernel_size=(1, 2),
                    activation='elu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (1, 2), activation='elu', name = "conv2"))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='selu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train,
            batch_size=64,
            epochs=100,
            verbose=2,
            validation_data=(X_test, Y_test), callbacks=[plot_learning])
def main():
    folder_path = 'datasets/64bitGF/delay/'
    csv_filename = 'test.csv'
    X, Y = read_csv_data(folder_path + csv_filename, header=None, num_funcs=6)
    print("X.shape = ", X.shape)
    print("Y.shape = ", Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
    plot_learning = PlotLearning()
    train(X_train,Y_train,X_test,Y_test,plot_learning.plot_history())


if __name__ == "__main__":
    main()