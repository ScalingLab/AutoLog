# coding=utf-8

from __future__ import unicode_literals
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, Normalizer, RobustScaler, MinMaxScaler, Binarizer


import pandas as pd
import numpy as np

#BG/L
names = ['ID','NULL','R00','R01','R02','R03','R04','R05','R06','R07','R10','R11','R12','R13','R14','R15','R16','R17','R20',
       'R21','R22','R23','R24','R25','R26','R27','R30','R31','R32','R33','R34','R35','R36','R37','R40','R41','R42','R43',
         'R44','R45','R46','R47','R50','R51','R52','R53','R54','R55','R56','R57','R60','R61','R62','R63','R64','R65',
         'R66','R67','R70','R71','R72','R73','R74','R75','R76','R77','UNKNOWNLOCATION','UNSPECIFIED','Label']

features = ['NULL','R00','R01','R02','R03','R04','R05','R06','R07','R10','R11','R12','R13','R14','R15','R16','R17','R20',
       'R21','R22','R23','R24','R25','R26','R27','R30','R31','R32','R33','R34','R35','R36','R37','R40','R41','R42','R43',
         'R44','R45','R46','R47','R50','R51','R52','R53','R54','R55','R56','R57','R60','R61','R62','R63','R64','R65',
         'R66','R67','R70','R71','R72','R73','R74','R75','R76','R77','UNKNOWNLOCATION','UNSPECIFIED']
def get_data_from_file(file):
    df = pd.read_csv(file, names=names, header=None, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    df['Label'] = np.where(df['Label'] == 'OK', 0, 1)
    return df

def processAP ( df_train, test_size ):

    print('AP all data size    : ', df_train.shape)

    # -- good records -- #
    goodrecords = df_train[df_train['Label'] == 0]
    print('All goods    : ', goodrecords.shape)
    badrecords = df_train[df_train['Label'] == 1]
    print('All bads    : ', badrecords.shape)

    #BG/L
    random_seed = 345
    #AutoLog
    #random_seed = 101
    x_train, x_test = train_test_split(goodrecords, test_size=test_size, random_state=random_seed)

    print(' + good train    : ', x_train.shape)
    print(' + good test    : ', x_test.shape)

    # --- merge x_test + bad records --- #
    frames = [x_test, badrecords]
    testfinal = pd.concat(frames)

    print(' + testfinal    : ', testfinal.shape)

    # TRAIN -> x_train
    # TEST  -> testfinal

    y_train = pd.to_numeric(x_train['Label'], errors='coerce')
    x_train = x_train.drop(['Label'], axis=1)

    y_test = pd.to_numeric(testfinal['Label'], errors='coerce')
    print('y_test: ')
    print(y_test)
    x_test = testfinal.drop(['Label'], axis=1)


    x_train = x_train[features]
    x_test = x_test[features]


    x_train = x_train.values
    x_test = x_test.values

    # MaxAbsScaler, MinMaxScaler, StandardScaler, Binarizer, Normalizer, RobustScaler
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)  # fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)  # <- perchè non fit

    # --- no scale code --- #
    #x_train = x_train.astype('float')
    #x_test = x_test.astype('float')
    #x_train_scaled = x_train
    #x_test_scaled = x_test

    print (x_train_scaled[23 ]);
    print (x_test_scaled[123]);

    # return x_train, x_test, y_train, y_test
    return x_train_scaled, x_test_scaled, y_train, y_test


