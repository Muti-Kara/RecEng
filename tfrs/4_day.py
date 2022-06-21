"""
tf.data.Dataset:
This files includes some usage of this library
"""
##	
import tensorflow as tf
##	
dataset = tf.data.Dataset.from_tensor_slices( [1, 2, 4, 5, 10] )
print( list(dataset.as_numpy_iterator()) )
"""
	[1, 2, 4, 5, 10]
"""
##	
dataset = dataset.map(lambda x: x**2)
print( list(dataset.as_numpy_iterator()) )
dataset = dataset.filter(lambda x: x > 5)
print( list(dataset.as_numpy_iterator()) )
"""
	[1, 4, 16, 25, 100]
	[16, 25, 100]
"""
##	
dataset = tf.data.Dataset.range(20)
print( list(dataset.as_numpy_iterator()) )
"""
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
"""
##	
dataset = dataset.batch(batch_size=4)
print( list(dataset.as_numpy_iterator()) )
dataset = dataset.batch(batch_size=4)
print( list(dataset.as_numpy_iterator()) )
"""
	[array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11]), array([12, 13, 14, 15]), array([16, 17, 18, 19])]
	[array([[ 0,  1,  2,  3],
		   [ 4,  5,  6,  7],
		   [ 8,  9, 10, 11],
		   [12, 13, 14, 15]]), array([[16, 17, 18, 19]])]
"""
##	
dataset = tf.data.Dataset.range(20)
print( list(dataset.as_numpy_iterator()) )
dataset = dataset.batch(7, drop_remainder=True)
print( list(dataset.as_numpy_iterator()) )
"""
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	[array([0, 1, 2, 3, 4, 5, 6]), array([ 7,  8,  9, 10, 11, 12, 13])]
"""
##	
dataset = tf.data.Dataset.range(10)
dataset = dataset.cache("cache")
print( list(dataset.as_numpy_iterator()) )
"""
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""
##	
dataset = tf.data.Dataset.range(20)
dataset = dataset.cache("cache")
print( list(dataset.as_numpy_iterator()) )
"""
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""
##	
dataset = tf.data.Dataset.range(5)
dataset = dataset.enumerate(start=7)
print( list(dataset.as_numpy_iterator()) )
"""
	[(7, 0), (8, 1), (9, 2), (10, 3), (11, 4)]
"""
##	
dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
print(list(dataset.as_numpy_iterator()))
"""
	[(1, 3, 5), (2, 4, 6)]
"""
##	
dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
print(list(dataset.as_numpy_iterator()))
"""
	[{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
"""
##	
features = tf.constant([[1.6, 55], [1.8, 86], [1.7, 60]])
labels   = tf.constant(["K", "E", "K"])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(list(dataset.as_numpy_iterator()))
"""
	[(array([ 1.6, 55. ], dtype=float32), b'K'), (array([ 1.8, 86. ], dtype=float32), b'E'), (array([ 1.7, 60. ], dtype=float32), b'K')]
"""
##	
features_dataset = tf.data.Dataset.from_tensor_slices(features)
labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
print(list(dataset.as_numpy_iterator()))
"""
	[(array([ 1.6, 55. ], dtype=float32), b'K'), (array([ 1.8, 86. ], dtype=float32), b'E'), (array([ 1.7, 60. ], dtype=float32), b'K')]
"""
