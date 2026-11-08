# nn-python
Simple two-layer fully-connected neural network, implemented in pure Python/NumPy

## Run training
Splits Iris into train/test (80/20), prints both accuracies,
saves weights to `weights.npz` and the loss curve to `loss.png`.

```
python train.py
```

## Run inference
Loads weights from `weights.npz` (run training first).

```
python inference.py
```
