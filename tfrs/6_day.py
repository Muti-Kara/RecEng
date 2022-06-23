import numpy as np
##	
tr_glove = open("/home/yuio/rec_eng/tfrs/dataset/turkish_glove.txt", "r")
word_dict = {}
for index, line in enumerate(tr_glove):
	tokens = line.split()
	word_dict[tokens[0]] = np.array(tokens[1:], dtype=np.float64)
##	
from TurkishStemmer import TurkishStemmer
stem = TurkishStemmer()
##	
def find_close(input: str, word_num: int):
	array = np.zeros(300)
	try:
		array = word_dict[input]
	except:
		try:
			print("\"" + input + "\" is stemmed to: " + stem.stem(input))
			array = word_dict[stem.stem(input)]
		except:
			return
	temp = {k: v for k, v in sorted(word_dict.items(), key=lambda x: np.linalg.norm(x[1] - array))}
	return list(temp.keys())[1:(word_num+1)]
##	
print( find_close("selam", 15) )
"""
	['selamlar', 'selâm', 'aleyküm', 'merhaba', 'herkese',
	'akşamlar', 'slm', 'sevgili', 'aleykum', 'efendim',
	'olsun', 'diyorum', 'kardeşim', 'gönderiyorum', 'selamı']
"""
##	
"""
To put pre trained values to embedding layer:
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)
where embedding_matrix is a np matrix with shape (num_tokens=voc_size+2, embedding_dim)
"""
##	
import tensorflow as tf
from keras.utils import pad_sequences
import keras
##	
sentences = [
	"matematikte problem sorularını çözmekte zorlanıyorum",
	"kuantum fiziği kafa karıştırıcı olabiliyor",
	"sayılar teorisiyle aram iyidir",
	"kütleçekim kanununu newton bulmuş"
]
labels = [0, 1, 0, 1]
##	
max_words = 100
max_seq_len = 10
##	
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<unk>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
sequences = pad_sequences(sequences, padding="post", maxlen=max_seq_len)
##	
num_tokens = len( tokenizer.word_index ) + 1
rev_index = {v: k for k, v in tokenizer.word_index.items()}
embedding_dim = 300
##	
embedding_matrix = np.zeros(shape=(num_tokens, embedding_dim))
hits = 0
miss = 0
##	
for i in range(1, num_tokens):
	word = rev_index[i]
	word_vector = word_dict.get(word) # prevents exceptions
	if word_vector is not None:
		embedding_matrix[i] = word_vector
		hits += 1
	else:
		miss += 1
print(f"num of hits: {hits}")
print(f"num of miss: {miss}")
##	
model = keras.Sequential([
	keras.layers.Input(shape=(max_seq_len,), dtype="int32"),
	keras.layers.Embedding(
						input_dim=num_tokens, 
						output_dim=embedding_dim, 
						embeddings_initializer=keras.initializers.Constant(embedding_matrix),
						trainable=False,
					),
	keras.layers.Conv1D(128, 5, activation="relu"),
	keras.layers.MaxPooling1D(5),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dropout(0.5),
	keras.layers.Flatten(),
	keras.layers.Dense(1, activation="sigmoid")
])
##	
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
##	
model.fit(np.array( sequences ), np.array( labels ), epochs=3, verbose=2)
##	
def find_close_in_embedding(input):
	if word_dict.get(input) is None:
		return "<unk>"
	a = list( tokenizer.word_index.keys() )
	a.sort(key=lambda x: np.linalg.norm(embedding_matrix[tokenizer.word_index[x]] - word_dict.get(input)))
	if 7 > np.linalg.norm(embedding_matrix[tokenizer.word_index[a[0]]] - word_dict.get(input)):
		return a[0]
	else:
		return "<unk>"
##	
def preprocess(sentence):
	new_sentence: str = ""
	for word in sentence.split():
		new_sentence += find_close_in_embedding(word) + " "
	print(new_sentence)
	new_sentence = tokenizer.texts_to_sequences([new_sentence])
	return pad_sequences(new_sentence, padding="post", maxlen=max_seq_len)
##	
print( model.predict(preprocess("izafiyet fiziğin en önemli isimlerinden einstein")) )
print( model.predict(preprocess("sayısal matematik soruları çok zor")) )
"""
	teorisiyle kuantum olabiliyor olabiliyor <unk> kuantum 
	1/1 [==============================] - 0s 42ms/step1/1 [==============================] - ETA: 0s
	[[0.7876264]]
	sayılar matematikte sorularını iyidir olabiliyor 
	1/1 [==============================] - 0s 10ms/step1/1 [==============================] - ETA: 0s
	[[0.25270164]]
"""
