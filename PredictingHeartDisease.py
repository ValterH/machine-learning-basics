import numpy as np
from sklearn import tree, neighbors
from sklearn import preprocessing, model_selection
import pandas as pd

df = pd.read_csv('processed.all.data.txt')
df.replace ('?', -99999, inplace = True)
df.drop(['ST_depression','ST_slope'], 1, inplace = True)
x = np.array(df.drop(['diagnosis'],1))
y = np.array(df['diagnosis'])

x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

#clf = neighbors.KNeighborsClassifier()
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
    
print(accuracy)

example_measures = np.array([28,1,2,130,132,0,2,185,0,0,3])

example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print(prediction)
