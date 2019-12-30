# Fast TSNE

This is a fork of [Multicore t-SNE](https://github.com/DmitryUlyanov/Multicore-TSNE) cython wrapper. This code has similar speed with Multicore t-SNE. In addition, support partial fitting function to continuously add points into tsne map.

<center><img src="mnist-tsne.png" width="512"></center>


# How to use

Cython wrappers are available.

## Python

### Pre-Requirements 
- Python3
- Numpy (>=1.18.0)
- Cython (>=0.28.2)
- cysignals
- cmake
- OpenMP 2(slow if not install)

### Install
```
pip install numpy cython cysignals
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE --DPYTHON_EXECUTABLE=$(which python) ..
make
make install

```

### UnInstall

`pip uninstall PyFastTsne`

Tested with 3.6 (conda) and Ubuntu 16.04.

### Run

You can use it as a near drop-in replacement for [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

```
from PyFastTsne import PyTsne

x_dim = 728
y_dim = 2
tsne = PyTsne(x_dim, y_dim)
Y = tsne.fit_transform(X)

## continuously add extra points 
tsne = tsne.partial_fit(extra_X, ret_Y, n_iter=300)

```

Please refer to [sklearn TSNE manual](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) for parameters explanation.

This implementation `n_components=2`, which is the most common case (use [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne) or sklearn otherwise). Also note that some parameters are there just for the sake of compatibility with sklearn and are otherwise ignored. See `MulticoreTSNE` class docstring for more info.


### Test

You can test it on MNIST dataset with the following command:

```
cd build/PyFastTsne
python test.py

```

# License

Inherited from [original repo's license](https://github.com/lvdmaaten/bhtsne).

# Future work

- Allow other types than double
- Improve step 2 performance (possible)

# Citation

- [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
- [L. Van der Maaten's paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

