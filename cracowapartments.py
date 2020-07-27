import pandas as pd
a=pd.read_csv(r"G:\machine learning\LIN REG\cracow_apartments.csv")
print(a)

x=a.iloc[:,:-1]
y=a.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state= 0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import r2_score
c=r2_score(y_test,y_pred)
print(c)

import pickle
pickle.dump(lr,open('model.pkl','wb'))