import tensorflow_datasets as tfds
import tensorflow as tf
##	
ratings: tf.data.Dataset = tfds.load("movielens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
	"movie_title": x["movie_title"],
	"user_id": x["user_id"],
	"user_rating": x["user_rating"]
})
##	
tf.random.set_seed(42)
shuffled: tf.data.Dataset = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train: tf.data.Dataset = shuffled.take(80_000)
test: tf.data.Dataset = shuffled.skip(80_000).take(20_000)
##	
import numpy as np
##	
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
movie_title = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
unique_title = np.unique(np.concatenate(list(movie_title)))
unique_id = np.unique(np.concatenate(list(user_ids)))
##	
import tensorflow_recommenders as tfrs
from typing import Dict, Text
import keras
##	
class RankingModel(keras.Model):
	def __init__(self):
		super().__init__()
		self.embedding_dim = 32
		
		self.user_emb = keras.Sequential([
			keras.layers.preprocessing.string_lookup.StringLookup(vocabulary=unique_id, mask_token=None),
			keras.layers.Embedding(input_dim=len(unique_id)+1, output_dim=self.embedding_dim)
		])
		self.movie_emb = keras.Sequential([
			keras.layers.preprocessing.string_lookup.StringLookup(vocabulary=unique_title, mask_token=None),
			keras.layers.Embedding(input_dim=len(unique_title)+1, output_dim=self.embedding_dim)
		])
		self.ratings = keras.Sequential([
			keras.layers.Dense(256, activation="relu"),
			keras.layers.Dense(1)
		])
	
	def call(self, inputs):
		user_id, movie_title = inputs
		user_emb = self.user_emb(user_id)
		movie_emb = self.movie_emb(movie_title)
		return self.ratings(tf.concat([user_emb, movie_emb], axis=1))
##	
class MovieLensModel(tfrs.Model):
	def __init__(self):
		super().__init__()
		self.ranking_model = RankingModel()
		self.task = tfrs.tasks.Ranking(
			loss=keras.losses.MeanSquaredError(),
			metrics=[keras.metrics.RootMeanSquaredError()]
		)
			
	def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
		return self.ranking_model((features["user_id"], features["movie_title"]))
	
	def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
		labels = features.pop("user_rating")
		rating_predictions = self.ranking_model((
			features["user_id"], 
			features["movie_title"]
		))
		return self.task(labels=labels,	predictions=rating_predictions)
##	
model: keras.Model = MovieLensModel()
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
##	
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
##	
model.fit(cached_train, validation_data=cached_test, epochs=200)
"""
When I trained model above approximately 150 epoch with learning rate 0.3
I could get results of rmse lower than 0.5 however due to overfit validation
rmse were about 1.5!

The best balance I could find was 0.82 on training and 0.9 on test data
"""
##	
test_ratings = {}
test_movie_titles = ["M*A*S*H (1970)", "Two if by Sea (1996)", "Clear and Present Danger (1994)", "Grease (1978)"]
for movie_title in test_movie_titles:
	test_ratings[movie_title] = model({
		"user_id": np.array(["42"]),
		"movie_title": np.array([movie_title])
	})

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
	print(f"{title}: {score}")

