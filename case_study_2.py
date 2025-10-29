# Handwritten Digit Recognition using SVM and KNN

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
digits = datasets.load_digits()
X, y = digits.data / 16.0, digits.target  # Normalize pixel values (0-16 -> 0-1)

# Display some samples
plt.figure(figsize=(6, 3))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title("Label: " + str(digits.target[i]))
    plt.axis("off")
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM Model
svm = SVC(kernel="rbf", C=10, gamma=0.01)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Accuracy
svm_acc = accuracy_score(y_test, y_pred_svm)
knn_acc = accuracy_score(y_test, y_pred_knn)

print("SVM Accuracy:", round(svm_acc * 100, 2), "%")
print("KNN Accuracy:", round(knn_acc * 100, 2), "%")

# Confusion Matrices
print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nKNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
