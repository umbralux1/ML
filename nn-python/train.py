import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 10

ALPHA = 0.0002
NUM_EPOCHS = 400
BATCH_SIZE = 50
SEED = 42

# Фиксируем seed, чтобы результаты повторялись от запуска к запуску
random.seed(SEED)
np.random.seed(SEED)


def relu(t):
    return np.maximum(t, 0)


def relu_deriv(t):
    return (t >= 0).astype(float)


def softmax_batch(t):
    # Вычитаем максимум, чтобы np.exp не переполнялся на больших значениях
    out = np.exp(t - np.max(t, axis=1, keepdims=True))
    return out / np.sum(out, axis=1, keepdims=True)


def sparse_cross_entropy_batch(z, y):
    return -np.log(z[np.arange(len(y)), y])


def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    y_full[np.arange(len(y)), y] = 1
    return y_full


# Данные: 80% на обучение, 20% на честную проверку
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=SEED, stratify=iris.target
)
dataset = [(x_train[i][None, ...], y_train[i]) for i in range(len(y_train))]

# Инициализация весов в диапазоне ±sqrt(1/n)
W1 = (np.random.rand(INPUT_DIM, H_DIM) - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
b1 = (np.random.rand(1, H_DIM) - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
W2 = (np.random.rand(H_DIM, OUT_DIM) - 0.5) * 2 * np.sqrt(1 / H_DIM)
b2 = (np.random.rand(1, OUT_DIM) - 0.5) * 2 * np.sqrt(1 / H_DIM)

loss_arr = []

for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):
        batch_x, batch_y = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Forward
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y))

        # Backward
        y_full = to_full_batch(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Update (SGD)
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)


def predict(x):
    h1 = relu(x @ W1 + b1)
    return softmax_batch(h1 @ W2 + b2)


def calc_accuracy(x, y):
    y_pred = np.argmax(predict(x), axis=1)
    return np.mean(y_pred == y)


print(f"Train accuracy: {calc_accuracy(x_train, y_train):.4f}")
print(f"Test accuracy:  {calc_accuracy(x_test, y_test):.4f}")

# Сохраняем веса, чтобы inference.py мог их загрузить
np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Weights saved to weights.npz")

plt.plot(loss_arr)
plt.xlabel("Шаг обучения")
plt.ylabel("Ошибка (сумма по батчу)")
plt.title("Кривая обучения")
plt.savefig("loss.png")
plt.show()
