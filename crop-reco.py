import pandas as pd
crop = pd.read_csv("croppredict.csv")
crop.head(10)

#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.distplot(crop['N'])
#plt.show()

crop_dict = {
    "rice": 1,
    "maize": 2,
    "jute": 3,
    "cotton": 4,
    "coconut": 5,
    "papaya": 6,
    "orange": 7,
    "apple": 8,
    "muskmelon": 9,
    "watermelon": 10,
    "grapes": 11,
    "mango": 12,
    "banana": 13,
    "pomegranate": 14,
    "lentil": 15,
    "blackgram": 16,
    "mungbean": 17,
    "mothbeans": 18,
    "pigeonpeas": 19,
    "kidneybeans": 20,
    "chickpea": 21,
    "coffee": 22
}
crop["crop_num"] = crop['label'].map(crop_dict)

crop.drop(['label'], axis=1, inplace=True)

## train test split
X = crop.drop(['crop_num'],axis=1)
Y = crop['crop_num']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
ms.fit(x_train)
x_train = ms.transform(x_train)
x_test = ms.transform(x_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

## training the models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


models = {
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Support Vector Classifier': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Extra Tree Classifier': ExtraTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Bagging Classifier': BaggingClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'AdaBoost Classifier': AdaBoostClassifier()
}

for name,md in models.items():
    md.fit(x_train,y_train)
    ypred = md.predict(x_test)
    print(f"{name} with accuracy {accuracy_score(y_test,ypred)}")


rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
ypred = rfc.predict(x_test)
accuracy_score(y_test,ypred)


## prediction system
import numpy as np
def recommandation(N,P,K,temperature,humidity,ph,rainfall):
    features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    prediction = rfc.predict(features).reshape(1,-1)
    return prediction[0]

## prediction system
import numpy as np
def recommandation(N,P,K,temperature,humidity,ph,rainfall):
    features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    prediction = rfc.predict(features).reshape(1,-1)
    return prediction[0]

N = 20
P = 30
K = 40
temperature = 40.0
humidity = 20
ph = 30
rainfall = 50

predict = recommandation(N,P,K,temperature,humidity,ph,rainfall)

crops_dict = {
    1: "rice",
    2: "maize",
    3: "jute",
    4: "cotton",
    5: "coconut",
    6: "papaya",
    7: "orange",
    8: "apple",
    9: "muskmelon",
    10: "watermelon",
    11: "grapes",
    12: "mango",
    13: "banana",
    14: "pomegranate",
    15: "lentil",
    16: "blackgram",
    17: "mungbean",
    18: "mothbeans",
    19: "pigeonpeas",
    20: "kidneybeans",
    21: "chickpea",
    22: "coffee"
}
if predict[0] in crops_dict:
    crop = crops_dict[predict[0]]
    print("{} is a best crop to be cultivated".format(crop))
else:
    print("Sorry we could not predict the crop to cultivate")

import pickle

pickle.dump(rfc,open('model.pkl','wb'))
