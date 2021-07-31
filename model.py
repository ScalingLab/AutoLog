import numpy as np
import pandas as pd
import matplotlib as mpl
import tensorflow as tf
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Dropout
from keras import regularizers, initializers
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path

from sklearn.metrics import roc_curve
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax
from time import time


#BG/L
np.random.seed(4999)
tf.random.set_seed(4999)

class MultilayerAutoEncoder():

    def __init__(self, input_dim):

        input_layer = Input(shape=(input_dim,))

        layer = Dense(128, activation='relu',   #128
                        activity_regularizer=regularizers.l1(10e-5),
                        kernel_initializer=initializers.RandomNormal())(input_layer)

        #v2
        layer = Dropout(rate=0.6)(layer);
        #layer = Dropout(rate=0.5)(layer);

        #layer = Dropout(rate=0.5)(input_layer);


        layer = Dense(64, activation='tanh',    #64
                      activity_regularizer=regularizers.l1(10e-5),
                      kernel_initializer=initializers.RandomNormal())(layer)# RandomNormal())(layer)

        #v2
        layer = Dropout(rate=0.6)(layer);
        #layer = Dropout(rate=0.5)(layer);

        layer = Dense(128, activation='relu',   #128
                      activity_regularizer=regularizers.l1(10e-5),
                      kernel_initializer=initializers.RandomNormal())(layer)

        output_layer = Dense(input_dim, activation='relu',
                      activity_regularizer=regularizers.l1(10e-5),
                      kernel_initializer=initializers.RandomNormal())(layer)



        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)

        plot_model(self.autoencoder, to_file='model_plot.png', show_shapes=True,
                   show_layer_names=True)  # , rankdir='LR')

    def summary(self, ):
        self.autoencoder.summary()

    def train(self, x, y):


        #epochs = 50
        #v2
        epochs = 50
        batch_size = 2048
        #v1
        #epochs = 100
        #batch_size = 4
        validation_split = 0.1

        print('Start training.')

        #lr = 0.0002
        #print('lr:', lr)  # Adagrad    #sgd    #adadelta  #rmsprop
        self.autoencoder.compile(optimizer='rmsprop',
                                 loss='mean_squared_error')
        start = time()
        history = self.autoencoder.fit(x, y,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       validation_split=validation_split,
                                       verbose=2)
        print(time() - start)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


        # --- determine treshold ---
        x_val = x[x.shape[0]-(int)(x.shape[0]*validation_split):x.shape[0]-1, :]

        print(' + validation_sise    : ',  x_val.shape)
        print ( x_val[6] )
        print(x_val[16])

        val_predictions = self.autoencoder.predict(x_val)
        val_mse = np.mean(np.power(x_val - val_predictions, 2), axis=1)

        #threshold = np.median ( val_mse )   #np.mean ( val_mse ) + 3*np.std ( val_mse )
        threshold = np.percentile(val_mse , 90) #np.mean ( val_mse ) + np.std ( val_mse )
        print('Current threshold: ')
        print(threshold)
        # ---

        plt.plot(val_mse)
        plt.show()

        df_history = pd.DataFrame(history.history)
        return df_history, threshold

    def evaluate(self, x_test, y_test, threshold):
        predictions = self.autoencoder.predict(x_test)

        mse=np.mean (np.power(x_test - predictions, 2), axis=1)

        #mse = np.sum(np.power(x_test - predictions, 2), axis = 1)#
        #mse = abs (  np.min(x_test - predictions, axis=1) )
        y_test = y_test.reset_index(drop=True)
        df_error = pd.DataFrame({'reconstruction_error' : mse, 'true_class' : y_test})
        print(mse)
        #y_test = y_test.reset_index(drop=True)
        print('y_test***: ')
        #pd.option_context("display.max_rows", None, "display.max_columns", None)
        print(df_error.to_string())
        print(df_error.describe(include='all'))

        plot_reconstruction_error(df_error, threshold)
        compute(df_error, threshold)

        plot_thresold(df_error.true_class, df_error.reconstruction_error)




from pathlib import Path
base_dir = str(Path().resolve().parent)
def plot_reconstruction_error(errors, threshold):
    groups = errors.groupby('true_class')
    a_list = list(range(1, 373))
    fig, ax = plt.subplots(figsize=(5, 7))
    right = 0
    a_list = list(range(1, 141))
    for name, group in groups:
        #print('Index: ')
        #print(group.index)
        if max(group.index) > right: right = max(group.index)

        ax.plot(group.index, group.reconstruction_error,
                ms = 5, linestyle = '', markeredgecolor = 'black', markersize=5,
                label = 'Normal' if int(name) == 0 else 'Anomaly', marker ='o' if int(name) == 0 else 'v', color = 'lightgreen' if int(name) == 0 else 'red')
        a_list = list(range(1, 233))
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors = 'red', zorder = 100, label = 'Threshold',linewidth=2,linestyles='dashed')
    ax.semilogy()
    ax.legend()
    plt.minorticks_off()
    #ax.set_xticks([754])
    #ax.set_xticklabels(["754"])
    ax.set_xticklabels([])
    BIGGER_SIZE = 22
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.xlim(left = 0, right = right)
    #plt.ylim(bottom= 0.00003, top=1000)
    plt.yticks(fontsize=22)
    ax.legend(prop={'size': 15}, loc='lower right' )
    #pyplot.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.title('Reconstruction error for different classes')
    #plt.grid(True)
    plt.ylabel('Reconstruction error')
    plt.xlabel('Data point index')
    plt.savefig(base_dir + '/reconstruction_error.png', bbox_inches = 'tight', dpi = 500)
    #plt.figure(1)
    plt.show()


def compute(df_error, threshold):
    y_pred = [1 if e > threshold else 0 for e in df_error.reconstruction_error.values]
    conf_matrix = confusion_matrix(df_error.true_class, y_pred)

    tn, fp, fn, tp = conf_matrix.ravel()

    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = 2 * ( (precision*recall) / (precision+recall) )
    false_alarm = 1. * fp / (tn + fp)

    print('R  = ', recall );
    print('P  = ', precision);
    print('F1 = ', f1);
    print('false alarm = ', false_alarm);

    sns.heatmap(conf_matrix, xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], annot=True,fmt='d');
    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(base_dir + '/confusion_matrix.png', bbox_inches='tight', dpi=500)
    #plt.figure(2)
    plt.show()
    #fig2.show()



def plot_thresold(true_class, re):

    # new Threshold
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(true_class, re)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='AutoLog')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend(prop={'size': 17})
    # show the plot
    pyplot.show()
    # show the plot
    pyplot.show()
