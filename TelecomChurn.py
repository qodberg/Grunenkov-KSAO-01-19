# импорт библиотек
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# чтение данных из файла
df = pd.read_csv('telecom_churn.csv')
# выведем первые 5 строк
df.head(5)

df['Churn'] = df['Churn'].astype('int64')

df['Churn'].value_counts()

df['Churn'].mean()

# заменим категориальные признаки числовыми
df['International plan'] = pd.factorize(df['International plan'])[0]
df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
# приведем целевую переменную к числовому виду
df['Churn'] = df['Churn'].astype('int')
# отдельно скопируем стобик с названием штата
states = df['State']
# отдельно выделим целевую переменную
y = df['Churn']
# удалим из датасета стобцы с названием штата и целевую переменную
df.drop(['State', 'Churn'], axis=1, inplace=True)

# импорт нужных функций
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# разделение на ренировочный и тестовый набор
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3,
random_state=17)

# создание двух класссификаторов
tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)

# тренировка моделей
tree.fit(X_train, y_train)
knn.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

tree_pred = tree.predict(X_holdout)
accuracy_score(y_holdout, tree_pred) # 0.94

knn_pred = knn.predict(X_holdout)
accuracy_score(y_holdout, knn_pred) # 0.88

from sklearn.model_selection import GridSearchCV, cross_val_score

tree_params = {'max_depth': range(1,11),
'max_features': range(4,19)}

tree_grid = GridSearchCV(tree, tree_params,
cv=5, n_jobs=-1,
verbose=True)

tree_grid.fit(X_train, y_train)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors': range(1, 10)}

knn_grid = GridSearchCV(knn_pipe, knn_params,
cv=5, n_jobs=-1,
verbose=True)

knn_grid.fit(X_train, y_train)

knn_grid.best_params_, knn_grid.best_score_