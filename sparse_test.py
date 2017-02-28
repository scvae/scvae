import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix

def get_sparse_tensor(sparse_set):
	n = sparse_set.shape[0]
	indptr = sparse_set.indptr
	row = []
	for i in range(n):
		row += (indptr[i+1]-indptr[i])*[i]

	indices = np.array([[r,c] for r,c in zip(row, sparse_set.indices)], dtype=np.int64)

	values = sparse_set.data
	shape = np.array(sparse_set.shape, dtype=np.int64)

	return tf.SparseTensorValue(indices, values, shape)

num_features = 5

a = np.eye(num_features)
a[2,3] = 1
a[4,1] = 1

a_sparse = csr_matrix(a)
# a_sparse = a_sparse[0:3] 

# print(a)
# print(a_sparse.indices)
# print(a_sparse.indptr)
# print(a_sparse.data)
# print(a_sparse.toarray())

# n = a_sparse.shape[0]

# indptr = a_sparse.indptr
# row = []
# for i in range(n):
# 	row += (indptr[i+1]-indptr[i])*[i]

# col = a_sparse.indices

# indices = [[r,c] for r,c in zip(row, col)]
# print(indices)

# values = np.array(a_sparse.data, dtype=np.int32)
# shape = np.array(a_sparse.shape, dtype=np.int64)
# indices = np.array(indices, dtype=np.int64)



# initialize placeholders, symbolics, with shape (batchsize, features)
x = tf.sparse_placeholder(dtype=tf.int32)
y = tf.sparse_reduce_sum(x)

# Parameters in Variable objects, which can be accessed inside the session
W = tf.Variable(tf.ones([num_features, 10], dtype=tf.int32))
b = tf.Variable(tf.zeros([10, 5]))

z = tf.sparse_tensor_dense_matmul(x, W)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  subset = np.array([0, 1, 4])
  print(sess.run(y, feed_dict={x: get_sparse_tensor(a_sparse[subset])}))  # Will succeed.




