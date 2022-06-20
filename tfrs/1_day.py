import numpy as np
import keras
##	
model = keras.Sequential( [ keras.layers.Dense(units=1, input_shape=[1]) ] )
model.compile(optimizer="sgd", loss="mean_squared_error")
##	
x_inp = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
x_out = np.array([-3, -1, 1, 3, 5, 7], dtype=float)
##	
model.fit(x_inp, x_out, epochs=300)
##	
print( model.predict([10]) )
##			
##			
from keras.preprocessing.text import Tokenizer
##	
sentences = [
	"I love my dog.",
	"You love your dog and my cat.",
	"I, love my cat."
]
##	
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
##	
print( tokenizer.word_index )
##	
sequences = tokenizer.texts_to_sequences(sentences)
print( sequences )
##	
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
##	
print( tokenizer.word_index )
##	
test_sent = [
	"I love horses more than you"
]
##	
test_seq = tokenizer.texts_to_sequences(test_sent)
print( test_seq )
##	
from keras.utils import pad_sequences
##	
pad_seq = pad_sequences(sequences)
##	
print(pad_seq)
##			
##			
import json
##	
with open("dataset/Sarcasm_Headlines_Dataset.json", "r") as file:
	dataset = json.load(file)
##	
headlines = []
labels = []
for item in dataset["all"]:
	headlines.append(item["headline"])
	labels.append(item["is_sarcastic"])
##	
train_size = int(len(headlines) * 0.8)

train_headlines = headlines[:train_size]
train_labels = labels[:train_size]

test_headlines = headlines[train_size:]
test_labels = labels[train_size:]
##	
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
##	
vocab_size = 10000
embedding_dim = 64
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
##	
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_headlines)
##	
train_seq = tokenizer.texts_to_sequences(train_headlines)
train_pad = pad_sequences(train_seq, padding="post", maxlen=max_length, truncating=trunc_type)

test_seq = tokenizer.texts_to_sequences(test_headlines)
test_pad = pad_sequences(test_seq, padding="post", maxlen=max_length, truncating=trunc_type)
##	
import numpy as np
##	
train_labels = np.array(train_labels)
train_pad = np.array(train_pad)
test_labels = np.array(test_labels)
test_pad = np.array(test_pad)
##	
import keras
##	
model = keras.Sequential([
	keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
	keras.layers.Bidirectional(keras.layers.LSTM(64)),
	keras.layers.Dense(64, activation="relu"),
	keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
##	
model.fit(train_pad, train_labels, epochs=4, validation_data=(test_pad, test_labels), verbose=2)
##	
sentences = [
	"granny starting to fear spiders in the garden might be real",
	"the weather today is bright and sunny",
	"That's just what I needed today!",
	"I work 40 hours a week for me to be this poor.",
	"Really, Sherlock? No! You are clever.",
	"people are hungry in not developed countries"
]
##	
sentences = tokenizer.texts_to_sequences(sentences)
sentences = pad_sequences(sentences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
##	
print( model.predict(sentences) )
