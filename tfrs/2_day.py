import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import keras

from typing import Dict, Text
##	
ratings = tfds.load("movielens/100k-ratings", split="train")
movies  = tfds.load("movielens/100k-movies", split="train")
##	
ratings = ratings.map(lambda x: {
	"movie_title": x["movie_title"],
	"user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])
##	
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)
##	
movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
##	
unique_titles = np.unique(np.concatenate(list(movie_titles)))
unique_ids = np.unique(np.concatenate(list(user_ids)))
##	
embedding_dim = 32
##	
user_model = keras.Sequential([
	keras.layers.preprocessing.string_lookup.StringLookup(vocabulary=unique_ids, mask_token=None),
	keras.layers.Embedding(len(unique_ids) + 1, embedding_dim)
])
##	
movie_model = keras.Sequential([
	keras.layers.preprocessing.string_lookup.StringLookup(vocabulary=unique_titles, mask_token=None),
	keras.layers.Embedding(len(unique_titles) + 1, embedding_dim)
])
##	
metric = tfrs.metrics.FactorizedTopK(
	candidates=movies.batch(128).map(movie_model), 
	k=100
)
task = tfrs.tasks.Retrieval(metrics=metric)
##	
class MovieLensModel(tfrs.Model):
	def __init__(self, user_model, movie_model):
		super().__init__()
		self.movie_model: keras.Model = movie_model
		self.user_model: keras.Model = user_model
		self.task: keras.layers.Layer = task
	
	def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
		user_emb = self.user_model(features["user_id"])
		positive_movie_emb = self.movie_model(features["movie_title"])
		
		return self.task(user_emb, positive_movie_emb)
##	
model: keras.Model = MovieLensModel(user_model, movie_model)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
##	
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
##	
model.fit(cached_train, epochs=2)
##	
model.evaluate(cached_test, return_dict=True)
##	
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
##	
index.index_from_dataset(
	tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)
##	
user = 57
_, titles = index(tf.constant([f"{user}"]))
print(f"\nrecommended movies for user with id: {user}")
print("\n".join(map(str, list(np.array(titles[0])))))
##			
##			
"""
		Some outputs:
		recommended movies for user with id: 55
		b'Executive Decision (1996)'
		b'Lost World: Jurassic Park, The (1997)'
		b'Men in Black (1997)'
		b'Broken Arrow (1996)'
		b'Con Air (1997)'
		b'Eraser (1996)'
		b'Rock, The (1996)'
		b'Mission: Impossible (1996)'
		b'Independence Day (ID4) (1996)'
		b'Star Trek: First Contact (1996)'
	==================================================
		recommended movies for user with id: 56
		b"Pete's Dragon (1977)"
		b"Weekend at Bernie's (1989)"
		b'Santa Clause, The (1994)'
		b'First Knight (1995)'
		b'Under Siege 2: Dark Territory (1995)'
		b'Judge Dredd (1995)'
		b'Robin Hood: Prince of Thieves (1991)'
		b'Ace Ventura: Pet Detective (1994)'
		b'Transformers: The Movie, The (1986)'
		b'Star Trek V: The Final Frontier (1989)'
	==================================================
		recommended movies for user with id: 57
		b'Fled (1996)'
		b'Michael (1996)'
		b'Down Periscope (1996)'
		b'Beverly Hills Ninja (1997)'
		b'Lost World: Jurassic Park, The (1997)'
		b'Broken Arrow (1996)'
		b'First Kid (1996)'
		b'Thinner (1996)'
		b'Space Jam (1996)'
		b'Nutty Professor, The (1996)'
"""
