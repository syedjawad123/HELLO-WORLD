import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation,neighbours
df=pd.read_csv("iris.csv")
df.replace("setosa",1,inplace=True)
df.replace("versicolor",2,inplace=True)
df.replace("viginica",3,inplace=True)
df.replace("?",-999,inplace=True)
x=np.array(df.drop(["species"],1))
y=np.array(df["species"])
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)
plt.plot(x_train,y_train)
clf=neighbours.KNeighboursclassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print("Accuracy:",accuracy)
print("------------------------")
example_measures=np.array([[4.7,3.2,2,0.2],[5.1,2.4,4.3,1.3]])
prediction=clf.predict(example_measures)
print("Prediction=",prediction )