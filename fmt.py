import gzip
import pickle
import numpy as np


def save_data_as_npy():
    with gzip.open("data/mnist.pkl.gz", "rb") as f:
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = pickle.load(f, encoding="latin1")

    # ensure the arrays have correct shapes and types
    # input data: float64 arrays of shape (N, 784)
    # labels: uint8 arrays of shape (N,)
    tr_x = np.asarray(tr_x, dtype=np.float64)
    tr_y = np.asarray(tr_y, dtype=np.uint8)
    va_x = np.asarray(va_x, dtype=np.float64)
    va_y = np.asarray(va_y, dtype=np.uint8)
    te_x = np.asarray(te_x, dtype=np.float64)
    te_y = np.asarray(te_y, dtype=np.uint8)

    # verify shapes are correct
    # training: (50000,784), validation/test: (10000,784)
    assert tr_x.ndim == 2 and tr_x.shape[1] == 784
    assert va_x.ndim == 2 and va_x.shape[1] == 784
    assert te_x.ndim == 2 and te_x.shape[1] == 784
    assert tr_y.ndim == 1
    assert va_y.ndim == 1
    assert te_y.ndim == 1

    np.save("train_x.npy", tr_x)
    np.save("train_y.npy", tr_y)
    np.save("valid_x.npy", va_x)
    np.save("valid_y.npy", va_y)
    np.save("test_x.npy", te_x)
    np.save("test_y.npy", te_y)


if __name__ == "__main__":
    save_data_as_npy()
