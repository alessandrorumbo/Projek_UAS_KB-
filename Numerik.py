import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = np.array([
    [1.0, 2.0, 0],
    [1.5, 1.8, 0],
    [5.0, 8.0, 1],
    [8.0, 8.0, 1],
    [1.0, 0.6, 0],
    [9.0, 11.0, 1],
    [8.0, 2.0, 1],
    [10.0, 2.0, 1],
    [0.5, 1.0, 0]
])

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model:", accuracy)

data_baru = np.array([[2.0, 3.0], [6.0, 9.0]])
class_baru = knn.predict(data_baru)
print("Prediksi kelas untuk data baru:", class_baru)
