#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    # TODO: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig
    eigvec = np.linalg.eig(A)[1].T
    # TODO: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
    visualize_vectors(eigvec)


def EVD_decomposition(A):
    # TODO: Zad. 4.2. Uzupełnij funkcję tak by obliczała rozkład EVD zgodnie z zadaniem.
    eigval, eigvec = np.linalg.eig(A)
    L   = np.diag(eigval)
    K   = eigvec
    K_1 = np.linalg.inv(K)

    print("K\n",   K) 
    print("K_1\n", K_1) 
    print("L\n",   L) 

    A_ = K @ L @ K_1
    A_ = A_.astype(float)

    print("A\n",  A)
    print("A_\n", A_)

    print("\n? A == A_: ", np.all(np.isclose(A, A_)))
    print("******\n\n")


def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.

    eigval, eigvec = np.linalg.eig(A)
    eigvec = eigvec.T

    atr     = [(eigvec[0], "red")]
    atr_inv = [(eigvec[0] * (-1), "orange")]

    if not np.allclose(eigvec[0], eigvec[1]):
        atr = atr + [(eigvec[1], "green")]
        atr_inv = atr_inv + [(eigvec[1] * (-1), "blue")]

    for i, x in enumerate(atr):
        v,c = x
        v = v / np.sqrt(np.sum(v**2))
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.009, color=c, scale_units='xy', angles='xy', scale=1, zorder=4)
        plt.text(v[0], v[1], "{0}".format(np.around(eigval[i]), 2))

    for i, x in enumerate(atr_inv):
        v,c = x
        v = v / np.sqrt(np.sum(v**2))
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.009, color=c, scale_units='xy', angles='xy', scale=1, zorder=4)

    for i, vect_base in enumerate(vectors):
        V = vect_base
        for i in range(1000):
            V = A.dot(V)
            V = V / np.sqrt(np.sum(V**2))
        
        plot_color = "black"
        for vect, color in atr + atr_inv:
            dist = np.sqrt((V[0] - vect[0])**2 + (V[1] - vect[1])**2)
            if 0.3 > dist:
                plot_color = color
                break

        vect_base = vect_base / np.sqrt(np.sum(vect_base**2))
        plt.quiver(0.0, 0.0, vect_base[0], vect_base[1], width=0.004, color=plot_color, scale_units='xy', angles='xy', scale=1, zorder=4)
        
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.xlim([-1.3, 1.3])
    plt.ylim([-1.3, 1.3])
    plt.margins(0.05)
    plt.show()

def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=20)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)


    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)
