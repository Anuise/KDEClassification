from sklearn.model_selection import train_test_split
from memory_profiler import profile
from model import model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df=pd.read_csv("adult.csv")
df[df=='?']=np.nan
null_columns =['workclass','occupation','native.country']
for i in null_columns:
    df.fillna(df[i].mode()[0], inplace=True)

# print(df["native.country"].unique())
native=df["native.country"].value_counts().to_dict()
df["native.country"]=df["native.country"].map(native)

le=LabelEncoder()
df_cols=("workclass","education","marital.status","occupation","relationship","race","sex")
for i in df_cols:
    df[i]=le.fit_transform(df[i])
# print(df.head())

X=df.iloc[:,:-1].values
print(X)
y=df.iloc[:,-1].values
print(y)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=0)

# @profile
# def main():
#     model(X_train, y_train, X_test, y_test)

# if __name__ == '__main__':
#     main()