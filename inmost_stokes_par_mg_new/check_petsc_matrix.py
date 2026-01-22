import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
from scipy.linalg import null_space
from numpy.linalg import svd, eig
import matplotlib.pyplot as plt 

ia = np.loadtxt("ia.txt", dtype=np.dtype(int))
ja = np.loadtxt("ja.txt", dtype=np.dtype(int))
data = np.loadtxt("data.txt", dtype=np.dtype(float))

A = sparse.csr_matrix((data, ja, ia))

A_dense = A.todense()

# print(null_space(A_dense))

# svd_val = linalg.svds(A, k = 5, which = "SM")

svd_val = svd(A_dense)

print(svd_val)

print(svd_val[2][-1])

eig_val = eig(A_dense)

print(eig_val)

# print(linalg.inv(A))
  
# extract real part using numpy array 
x = eig_val[0].real 
# extract imaginary part using numpy array 
y = eig_val[0].imag 
  
# plot the complex numbers 
plt.plot(x, y, 'g*') 
plt.ylabel('Imaginary') 
plt.xlabel('Real') 
plt.show() 
