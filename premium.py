import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle

df=pd.read_csv("Medicalpremium.csv")
x=df.drop(columns=["PremiumPrice"])
y=df["PremiumPrice"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

lin_reg= LinearRegression()
dec_tree=DecisionTreeRegressor(criterion="squared_error",max_depth=7,min_samples_split=5,min_samples_leaf=20)
rand_for =RandomForestRegressor(criterion='squared_error',n_estimators=100,max_depth=10,min_samples_split=15)
xgb=XGBRegressor(n_estimators=100,max_depth=4,reg_lambda=.2,eta=0.3,eval_metric='rmse',gamma=0.5,objectives='reg:squarederror',
                random_state=0,reg_alpha=0)
adrf=RandomForestRegressor(criterion='squared_error',n_estimators=100,max_depth=10,min_samples_split=15)
ada_reg=AdaBoostRegressor(base_estimator=adrf,n_estimators=50)

lin_reg = lin_reg.fit(x_train,y_train)
dec_tree=dec_tree.fit(x_train,y_train)
rand_for=rand_for.fit(x_train,y_train)
xgb=xgb.fit(x_train,y_train)
ada_reg=ada_reg.fit(x_train,y_train)

pickle.dump(lin_reg,open('lin_model.pkl','wb'))
pickle.dump(dec_tree,open('dec_tree_model.pkl','wb'))
pickle.dump(rand_for,open('rand_for_model.pkl','wb'))
pickle.dump(xgb,open('xgb_model.pkl','wb'))
pickle.dump(ada_reg,open('ada_reg_model.pkl','wb'))