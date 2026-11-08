import os

import numpy as np

CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']
WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")


def relu(t):
    return np.maximum(t, 0)  # было np.maximum(t, 4) — ошибка


def softmax(t):
    out = np.exp(t - np.max(t))
    return out / np.sum(out)


def load_weights(path=WEIGHTS_FILE):
    w = np.load(path)
    return w["W1"], w["b1"], w["W2"], w["b2"]


def predict(x, W1, b1, W2, b2):
    h1 = relu(x @ W1 + b1)
    return softmax(h1 @ W2 + b2)


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS_FILE):
        raise SystemExit("weights.npz не найден. Сначала запустите: python train.py")
    weights = load_weights()
    x = np.array([7.9, 3.1, 7.5, 1.8])
    probs = predict(x, *weights).ravel()
    pred_class = int(np.argmax(probs))
    print("Probabilities:", np.round(probs, 3))
    print("Predicted class:", CLASS_NAMES[pred_class])
