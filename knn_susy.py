import time
import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd

data = pd.read_csv('SUSY.csv')
data.columns = ['class', 'l_1_pt', 'l_1_eta', 'l_1_phi', 'l_2_pT', 'l_2_eta', 'l_2_phi', 'mis_ene_magnitude',
                'mis_ene_phi', 'met_rel', 'a_mel', 'm_r', 'm_tr_2', 'r', 'mt2', 's_r', 'm_del_r', 'dPhi_r_b',
                'cos_t_r1']

X = np.array(data.drop(['class'], 1))
y = np.array(data['class'])

# train set %80, test set %20
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

# k = 5 default value
clf = neighbors.KNeighborsClassifier()

t_baslangic = time.time()
# train part
clf.fit(X_train, y_train)

t_bitis = time.time()

print("train süresi", t_bitis - t_baslangic)

te_baslangic = time.time()
# test ve accuracy
accuracy = clf.score(X_test, y_test)

te_bitis = time.time()
print("test süresi", te_bitis - te_baslangic)
print("accuracy:", accuracy)

exp_vektor = np.array([[5, -1, 6, 2, 6, 1, 8, -2, -3, 4, 7, 2, 9, -7, 1, 5, -6, 0]])
exp_vektor = exp_vektor.reshape(1, -1)

exp_baslangic = time.time()

exp_predict = clf.predict(exp_vektor)

exp_bitis = time.time()
print("örnek için tahmin edilen sınıf", exp_predict)
print("yeni bir örnek için tahmin süresi", exp_bitis - exp_baslangic)
