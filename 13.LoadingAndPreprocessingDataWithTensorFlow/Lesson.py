# %%
import tensorflow as tf
# %%
tf.config.experimental.list_physical_devices('GPU')
# %%
X = tf.range(10)
X
# %%
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
# %%
# The same dataset with range
dataset = tf.data.Dataset.range(10)
dataset
# %%
# Iterate over the dataset
for x in dataset:
    print(x)
# %% [markdown]
# Chaining Transformations
# %%
dataset = dataset.repeat(3).batch(7)

for item in dataset:
    print(item)
# %%
# Map to transform the dataset
dataset = dataset.map(
    lambda x: x * 2
)
for item in dataset:
    print(item)
# %%
# Unbatch the dataset
dataset = dataset.unbatch()
dataset
# %%
dataset = dataset.filter(lambda x: x < 6)
dataset
# %%
# Take elements from the dataset
dataset = dataset.take(3)
for item in dataset:
    print(item)
# %% [markdown]
# ### Shuffling the Data
# %%
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
for item in dataset:
    print(item)
# %%
[0.] * 8 + [tf.constant([], dtype=tf.float32)]
# %%
