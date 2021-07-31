from dataAP import processAP, get_data_from_file
from modelAP import MultilayerAutoEncoder
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
# importing libaries ----
import numpy as np
import datetime
base_dir = str(Path().resolve().parent)



base_dir = str(Path().resolve().parent)


file = base_dir + '/dataset/BGLvector.csv'
dataset = get_data_from_file(file)

x_train, x_test, y_train, y_test =  processAP(dataset, 0.0206)

print("def = " + str(x_train.shape) + " - " + str(y_train.shape) )
print("def = " + str(x_test.shape) + " - " +  str(y_test.shape) )

input_dim = x_train.shape[1]
autoencoder = MultilayerAutoEncoder(input_dim = input_dim)

autoencoder.summary()

history, threshold = autoencoder.train(x_train, x_train)

autoencoder.evaluate(x_test, y_test, threshold)


#Isolation Forest
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=len(x_train), random_state=rng)


#clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12), \
                        #max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)

clf.fit(x_train)
pred = clf.predict(x_test)

pred[pred == 1] = 0
pred[pred == -1] = 1

print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

#One class

clf2 = OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05,
                                         max_iter=-1).fit(x_train)

pred2 = clf2.predict(x_test)
pred2[pred2 == 1] = 0
pred2[pred2 == -1] = 1
print(classification_report(y_test, pred2))

