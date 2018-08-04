import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """

    out = np.matmul(vector1, vector2)
    return out

def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (1, x)
    """
    p1 = np.matmul(vector1,vector2)
    p2 = np.matmul(M,vector1)
    out =p1*p2


    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    Mt = matrix.transpose()
    MtM = np.matmul(Mt,matrix)
    eig_val, eig_vec = np.linalg.eig( MtM)
    s = np.sqrt(eig_val)
    sing_vec = np.eye(eig_val.shape[0])*s
    v = eig_vec 
    u = np.matmul(matrix,np.matmul( np.linalg.inv(sing_vec),v))
    return u, s, v

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """

    u, s, v = svd(matrix)
    singular_values = s[:n]
    return singular_values

def eigen_decomp(matrix):
    """ Implement Eigen Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, )

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    v,w = np.linalg.eig(matrix)
    return w, v

def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return
        
    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    w,v = eigen_decomp(matrix)
    v_args = (-v).argsort()
    eigen_values = v[v_args][:num_values]
    eigen_vectors = w[v_args][:num_values]

    return eigen_values, eigen_vectors
