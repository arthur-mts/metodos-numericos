import numpy as np


def plu(A):
    # Get the number of rows
    n = A.shape[0]

    # Allocate space for P, L, and U
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    P = np.eye(n, dtype=np.double)

    # Loop over rows
    for i in range(n):

        # Permute rows if needed
        for k in range(i, n):
            if ~np.isclose(U[i, i], 0.0):
                break
            U[[k, k + 1]] = U[[k + 1, k]]
            P[[k, k + 1]] = P[[k + 1, k]]

        # Eliminate entries below i with row
        # operations on U and #reverse the row
        # operations to manipulate L
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return P, L, U


def main():
    a = np.array([[1, 1, 1], [4, 4, 2], [2, 1, -1]], dtype=float)

    p, l, u = plu(a)

    res = p@l@u
    print(p)


if __name__ == '__main__':
    main()
