from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time
from KDEClassifier import KDEClassifier
import numpy as np

def model(X_train, y_train, X_test, y_test):
    bandwidths = 10 ** np.linspace(0, 5, 5)
    grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})

    start = time.time()
    grid.fit(X_train,y_train)
    end = time.time()
    training_time = end - start
    
    start = time.time()
    y_pred = (grid.predict(X_test))
    end = time.time()
    predict_time = end - start

    print(f"training time: {training_time}")
    print(f"predict time: {predict_time}")
    print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
    return 0
