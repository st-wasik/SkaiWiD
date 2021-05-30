import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float

def pad_matrix(mx, shape):
    if mx.shape[0] < shape[0]:
        rows_to_add = shape[0] - mx.shape[0]
        pad_mx = np.zeros(rows_to_add * mx.shape[1]).reshape(rows_to_add, mx.shape[1])
        mx = np.concatenate([mx, pad_mx], axis=0)
    
    if mx.shape[1] < shape[1]:
        cols_to_add = shape[1] - mx.shape[1]
        pad_mx = np.zeros(mx.shape[0] * cols_to_add).reshape(mx.shape[0], cols_to_add)
        mx = np.concatenate([mx, pad_mx], axis=1)
    
    return mx

def custom_svd(X):
    C = X.T @ X
    eigval, eigvec = np.linalg.eigh(C)
    ord = np.argsort(eigval)[::-1]

    s = eigval[ord]
    v = eigvec[:, ord]

    s_plus = np.diag(s)
    s_plus = pad_matrix(s_plus, X.shape).T
    s_plus = np.divide(1, s_plus, where=~np.isclose(s_plus, 0.0))

    u = X @ v @ s_plus
    return u, s, v.T

def svd_3dim_array(image_array, svd_fun, k):
    layers = [svd_2dim_array(image_array[:,:,x], svd_fun, k) for x in range(image_array.shape[2])]
    return np.stack(layers, axis=2)

def svd_2dim_array(image_array, svd_fun, k):
    u, s, vh = svd_fun(image_array)

    diag = np.diag(s)
    size = u.shape[1], vh.shape[0]
    diag = pad_matrix(diag, size)
    print(image_array.shape," -> ", u.shape, "@", diag.shape, "@", vh.shape)

    if k is None: 
        result = u @ diag @ vh
    else:
        result = u[:, :k] @ diag[:k, :k] @ vh[:k, :]

    return np.clip(result, 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",   help="plik z oryginalnym obrazkiem", required=True)
    parser.add_argument("-out", help="nazwa pliku wyjściowego")
    parser.add_argument("-svd", help="implementacja SVD do użycia", choices=['custom', 'library'], required=True)
    parser.add_argument("-k",   help="liczba wartości osobliwych użyta do kompresji", type=int)
    args = parser.parse_args()

    f   = args.f
    out = args.out
    svd = args.svd
    k   = args.k

    if f is None:
        print("ERROR: Image path not specified!")
        exit(-1)

    image = img_as_float(plt.imread(args.f))

    if svd == 'custom':
        svd_fun = custom_svd
    elif svd == 'library':
        svd_fun = np.linalg.svd
    else:
        print("ERROR: Wrong svd value '" + svd + "'")
        exit(-1)

    processed_image = svd_3dim_array(image, svd_fun, k)

    if out is None:
        plt.imshow(processed_image)
        plt.show()
    else:
        plt.imsave(fname=out, arr=processed_image)
    
main()