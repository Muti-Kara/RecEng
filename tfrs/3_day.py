"""
Embedding Layer:

Turns positive integers (indexes) into dense vectors of fixed size.
e.g. '[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]'
This layer can only be used on positive integer inputs of a fixed range. The

	'tf.keras.layers.TextVectorization', 
	'tf.keras.layers.StringLookup',
	'tf.keras.layers.IntegerLookup' 

preprocessing layers can help prepare inputs for an 'Embedding' layer.
This layer accepts 'tf.Tensor' and 'tf.RaggedTensor' inputs. It cannot be
called with 'tf.SparseTensor' input.

Note: for turkish I can initialize my vectors from GloVe (inzva should have one)
"""
import tensorflow as tf
##	
sentences = [
	"bence iyi",
	"kötü ve beğenmedim",
	"iyi ve bence",
	"kötü beğenmedim"
]
labels = [
	1, 
	0, 
	1, 
	0,
]
##	
text_dataset = tf.data.Dataset.from_tensor_slices((sentences, labels))
for input, label in text_dataset:
	print(input)
	print(label)
	print("========")
"""
	tf.Tensor(b'Bu iyi bir \xc5\x9fey.', shape=(), dtype=string)
	tf.Tensor(b'good', shape=(), dtype=string)
	========
	tf.Tensor(b'Pek be\xc4\x9fenmedim.', shape=(), dtype=string)
	tf.Tensor(b'bad', shape=(), dtype=string)
	========
	tf.Tensor(b'G\xc3\xbczel ve iyi bence.', shape=(), dtype=string)
	tf.Tensor(b'good', shape=(), dtype=string)
	========
	tf.Tensor(b'K\xc3\xb6t\xc3\xbc oldu\xc4\x9fundan be\xc4\x9fenmedim.', shape=(), dtype=string)
	tf.Tensor(b'bad', shape=(), dtype=string)
"""
##	
from keras.layers import TextVectorization
vocab_size = 20
max_len = 5
vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=max_len)
##	
vectorizer.adapt(sentences)
##	
vectorizer.call("kötü beğenmedim")
vectorizer.call("bence de iyi bir filmdi.")
"""
	<tf.Tensor: shape=(5,), dtype=int64, numpy=array([3,  5,  0,  0,  0])>
	<tf.Tensor: shape=(5,), dtype=int64, numpy=array([6,  1,  4,  1,  1])>
"""
##	
from keras.models import Sequential
from keras import Input
import keras
model: keras.Sequential = Sequential()
model.add(Input(shape=(1,), dtype=tf.string)) # there should at least one string input
model.add(vectorizer)
##	
model.predict(sentences)
"""
	1/1 [==============================] - 0s 63ms/step
	Remote 
	array([[12,  2, 13,  6,  0],
		   [ 8,  4,  0,  0,  0],
		   [11,  7,  2, 14,  0],
		   [10,  9,  4,  0,  0]])
"""
##	
from keras.layers import Embedding
embedding = Embedding(input_dim=vocab_size, output_dim=2, input_length=max_len)
model.add(embedding)
##	
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
##	
print( model.predict(["iyi mi iyi bence", "kötü bence"]) )
"""
Returns every word into a vector with size of 2
Afterwards trains parameters for creating those vectors,
with considering their relations with each other.

	1/1 [==============================] - 0s 35ms/step
	[
		[
			[ 0.03097531 -0.04502175]
			[-0.03307929  0.04493332]
			[ 0.03097531 -0.04502175]
			[-0.04290104 -0.04562867]
			[ 0.01941857 -0.00536176]
		]

		[
			[ 0.04429903  0.01423142]
			[-0.04290104 -0.04562867]
			[ 0.01941857 -0.00536176]
			[ 0.01941857 -0.00536176]
			[ 0.01941857 -0.00536176]
		]
	]
"""
##	
from keras.layers import Flatten, Dense
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
##	
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
##	
model.fit(sentences, labels, epochs=1500, verbose=2)
##	
print (model.predict(["iyi"]))
print (model.predict(["kötü"]))
print (model.predict(["bence"]))
print (model.predict(["beğenmedim"]))
print (model.predict(["ve"]))
print (model.predict(["rastgele bilinmeyen kelimeler"]))
"""
	1/1 [==============================] - 0s 19ms/step
	[[0.76893115]]
	1/1 [==============================] - 0s 17ms/step
	[[0.18155155]]
	1/1 [==============================] - 0s 17ms/step
	[[0.8181293]]
	1/1 [==============================] - 0s 17ms/step
	[[0.14950147]]
	1/1 [==============================] - 0s 17ms/step
	[[0.5019548]]
	1/1 [==============================] - 0s 17ms/step
	[[0.48070532]]
"""
##	
model2: keras.Sequential = Sequential([
	Input(shape=(1,), dtype=tf.string),
	vectorizer,
	embedding
])
##	
model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
##	
print( model2.predict(["iyi"]) )
print( model2.predict(["bence"]) )
print( model2.predict(["beğenmedim"]) )
print( model2.predict(["kötü"]) )
print( model2.predict(["ve"]) )
print( model2.predict(["rastgele"]) )
"""
With the embedding layer similar words that mean similar things have more close positions
	1/1 [==============================] - 0s 19ms/step
	[[[-0.5381482  -0.42297935] ...
	1/1 [==============================] - 0s 17ms/step
	[[[-0.5956968  -0.6100751 ] ...
	1/1 [==============================] - 0s 16ms/step
	[[[0.58169144 0.8015361 ] ...
	1/1 [==============================] - 0s 17ms/step
	[[[0.56382346 0.6285776 ] ...
	1/1 [==============================] - 0s 17ms/step
	[[[-0.09368262  0.08554994] ...
	1/1 [==============================] - 0s 16ms/step
	[[[0.03923881 0.02706058] ...
"""
