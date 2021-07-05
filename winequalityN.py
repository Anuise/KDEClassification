from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from memory_profiler import profile
from model import model
import pandas as pd

df=pd.read_csv("winequalityN.csv")
df = df.fillna(value=0)
df['Quality']=0
df.loc[df['quality']>6, 'Quality']=1
df.drop('quality', axis=1, inplace=True)
X= df.drop('Quality', axis=1).drop('type', axis=1)
y= df['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# ss=StandardScaler()
# X_train= ss.fit_transform(X_train.values)
# X_test= ss.transform(X_test.values)
# X_train= pd.DataFrame(X_train, columns= X.columns).values
# X_test=pd.DataFrame(X_test, columns=X.columns).values

@profile
def main():
    model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()