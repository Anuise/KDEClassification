from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from memory_profiler import profile
from model import model

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

@profile
def main():
    model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()