#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

import io
import numpy as np

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
# https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network


# Quick look at data

# In[3]:


import sqlite3
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

conn = sqlite3.connect("/home/pawel/git/orientation_test.db", isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES)
query = "SELECT orientation FROM data WHERE id = 33"
cursor = conn.cursor()
cursor.execute(query)
print(cursor.fetchall())

query = "SELECT orientation, voxels, pca, plane, cylinder, sphere FROM data WHERE id = 31"
cursor = conn.cursor()
cursor.execute(query)
sample = cursor.fetchall()
orientation, voxels, pca, plane, cylinder, sphere = sample[0]
nvoxels = np.asarray(voxels).reshape((9, 9, 9, 1))
print([[np.cos(angle)*0.5+0.5, np.sin(angle)*0.5+0.5] for angle in orientation])
print("1.) Voxels")
print(nvoxels.min(), nvoxels.max(), nvoxels.shape)
def plot_voxels(nvoxels, resolution=9, point_size=10):
    xs = list()
    ys = list()
    zs = list()
    sizes = list()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xs.append(x)
                ys.append(y)
                zs.append(z)
                sizes.append(point_size*(nvoxels.max() - nvoxels[x, z, y, 0] ))
    print("plot")
    ax.scatter(xs, ys, zs, s=sizes, c=sizes)
plot_voxels(nvoxels, resolution=9, point_size=70)
plt.show()

print("2.) PCA")
print(pca)
print("3.) plane")
print(plane.shape, plane.min(), plane.max())
plane = plane.reshape(128, 128)
plane -= plane.min()
plane /= plane.max()
plt.imshow(plane,cmap='nipy_spectral')
plt.colorbar()
plt.show()
print("4.) cylinder")
print(cylinder.shape, cylinder.min(), cylinder.max())
cylinder = cylinder.reshape(128, 128)
cylinder -= cylinder.min()
cylinder /= cylinder.max()
plt.imshow(cylinder,cmap='nipy_spectral')
plt.colorbar()
plt.show()
print("4.) sphere")
print(sphere.shape, sphere.min(), sphere.max())
sphere = sphere.reshape(128, 128)
sphere -= sphere.min()
sphere /= sphere.max()
plt.imshow(sphere,cmap='nipy_spectral')
plt.colorbar()
plt.show()


# In[4]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, Conv2D, MaxPooling3D, MaxPooling2D
from keras.utils import to_categorical
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping


# Generator for data from sqlite:

# In[5]:


import keras
import io
import math
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, num_features, end_index=None, start_index=1, batch_size=25,
                 shuffle=True, dim=(9, 9, 9), column="voxels"):
        'Initialization'
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        self.path = path
        self.dim = dim
        self.db = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        self.db_cursor = self.db.cursor()
        self.N = num_features
        self.start_index = start_index
        if end_index is None:
            end_index =  num_features + 1
        self.end_index = end_index
        self.column = column
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.sample_index = np.arange(self.start_index, self.end_index)
        self.on_epoch_end()
        self.db.close()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.N / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        samples_batch = np.arange((index) * self.batch_size, (index+1) * self.batch_size)

        # Generate data
        x, y = self.__data_generation(samples_batch)

        return  x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.sample_index)

    def __data_generation(self, samples_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty(shape=(self.batch_size, 6))
        inds = self.sample_index[samples_batch]
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)
        db = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db.cursor()
        sql_query = "SELECT orientation, {column} FROM data WHERE id in ({index})".            format(column=self.column, index=','.join(str(ind) for ind in inds))
        db_cursor.execute(sql_query)
        for i in range(self.batch_size):
            line = db_cursor.fetchone()
            y[i] = np.asarray([[np.sin(math.radians(angle)), np.cos(math.radians(angle))] for angle in line[0]]).reshape(6)
            x_tmp = line[1].reshape(self.dim)
            x_tmp -= x_tmp.min()
            x_tmp /= x_tmp.max()
            x[i, :] = x_tmp
        return x, y

    def adapt_array(self, arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)


# In[6]:


import keras.backend as K
import tensorflow as tf

def get_angle(x):
    return (tf.atan2(x[0], x[1]), tf.atan2(x[2], x[3]), tf.atan2(x[4], x[5]))

def get_sum_squares(x):
    return tf.pow(x[0][0] - x[1][0], 2) + tf.pow(x[0][1] - x[1][1], 2) + tf.pow(x[0][2] - x[1][2], 2)/3.0

def custom_loss(y_true, y_pred):
    true_angles = tf.map_fn(get_angle , y_true, dtype=(tf.float32, tf.float32, tf.float32))
    pred_angles = tf.map_fn(get_angle, y_pred, dtype=(tf.float32, tf.float32, tf.float32))
    return K.mean(tf.map_fn(get_sum_squares , (true_angles, pred_angles), (tf.float32)))
        
        


# 1.) Voxels. Denseand CNN

# In[7]:


voxel_training_generator = DataGenerator("/home/pawel/git/orientation_train.db", num_features=30*30*15, dim=(9, 9, 9, 1), batch_size=32, column="voxels")
voxel_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(9, 9, 9, 1), batch_size=32, column="voxels")


# Just quick peek on data from generator, to be sure that it works:

# In[7]:


print(len(voxel_validation_generator))
v = voxel_validation_generator[1][0][0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_voxels(v, resolution=9, point_size=70)


# a) Simple dense neural net

# In[8]:


model_dense_voxel = Sequential()
model_dense_voxel.add(Flatten(input_shape=(9, 9, 9, 1))) 
model_dense_voxel.add(Dense(1024,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_dense_voxel.add(Dense(256,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_dense_voxel.add(Dense(32,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_dense_voxel.add(Dense(6, activation="tanh"))


# In[9]:


model_dense_voxel.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = ['mse', 'mae', 'mape', 'cosine'])
model_dense_voxel.summary()


# In[12]:


#General callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)


# In[11]:


# Callbacks for dense
checkpoint_dense_voxel = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights_dense_voxel.hdf5', verbose=0, save_best_only=True)
store_result_dense_voxel = keras.callbacks.callbacks.CSVLogger("results_dense_voxel.csv", separator=',', append=False)


# In[12]:


# Using workers=2, use_multiprocessing=True sometimes resulted in freeze
result_dense_voxel = model_dense_voxel.fit_generator(voxel_training_generator, verbose=0, validation_data=voxel_validation_generator,
                                                     epochs=500, 
                                                     callbacks=[early_stopping, checkpoint_dense_voxel, store_result_dense_voxel, reduce_lr])
model_dense_voxel.save('dense_voxel_trained_model.model')


# b) Convolutional neural network

# In[8]:


# Build convolutional neural net

model_cnn_voxels = Sequential()

model_cnn_voxels.add(Conv3D(16,(5,5,5), padding = 'same', activation = 'relu', input_shape = (9,9,9,1)))
model_cnn_voxels.add(BatchNormalization())

model_cnn_voxels.add(MaxPooling3D(pool_size = (2,2,2)))
model_cnn_voxels.add(Dropout(0.1))

model_cnn_voxels.add(Conv3D(32, (5,5,5), padding = 'same', activation = 'relu'))
model_cnn_voxels.add(BatchNormalization())

model_cnn_voxels.add(MaxPooling3D(pool_size = (2,2,2)))

model_cnn_voxels.add(Dropout(0.1))

model_cnn_voxels.add(Flatten())

model_cnn_voxels.add(Dense(128, activation = 'relu'))

model_cnn_voxels.add(Dense(6, activation="tanh"))


# In[9]:


model_cnn_voxels.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = ['mae'])
model_cnn_voxels.summary()


# In[10]:


checkpoint_cnn_voxel = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights_cnn_voxel.hdf5', verbose=1, save_best_only=True)
store_result_cnn_voxel = keras.callbacks.callbacks.CSVLogger("results_cnn_voxel.csv", separator=',', append=False)


# In[ ]:


result_cnn_voxels = model_cnn_voxels.fit_generator(generator=voxel_training_generator, validation_data=voxel_validation_generator,
                                                   epochs=500, 
                                                   callbacks=[early_stopping, checkpoint_cnn_voxel, store_result_cnn_voxel, reduce_lr, early_stopping])


# In[ ]:


model_cnn_voxels.save('cnn_voxel_trained_model.model')


# 2.) Principal component analysis

# In[ ]:


pca_training_generator = DataGenerator("/home/pawel/git/orientation_train.db", num_features=30*30*15, dim=(3, 3, 1), batch_size=32, column="pca")
pca_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(3, 3, 1), batch_size=32, column="pca")


# In[ ]:


model_pca = Sequential()
model_pca.add(Flatten(input_shape=(3, 3, 1))) 
model_pca.add(Dense(512,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_pca.add(Dense(64,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_pca.add(Dense(6, activation="tanh"))


# In[ ]:


model_pca.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = ['mae'])
model_pca.summary()


# In[ ]:


# Callbacks for dense
checkpoint_pca = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights_dense_pca.hdf5', verbose=0, save_best_only=True)
store_result_pca = keras.callbacks.callbacks.CSVLogger("results_dense_pca.csv", separator=',', append=False)


# In[ ]:


result_pca = model_pca.fit_generator(pca_training_generator, verbose=0, validation_data=pca_validation_generator,
                                           epochs=500, 
                                           callbacks=[early_stopping, checkpoint_pca, store_result_pca, reduce_lr])
model_pca.save('pca_trained_model.model')


# 3.) Projection on plane

# In[ ]:


plane_training_generator = DataGenerator("/home/pawel/git/orientation_train.db", num_features=30*30*15, dim=(128, 128, 1), batch_size=32, column="plane")
plane_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(128, 128, 1), batch_size=32, column="plane")


# In[ ]:


model_plane = Sequential()

model_plane.add(Conv2D(16,(5,5), padding = 'same', activation = 'relu', input_shape = (128,128,1)))
model_plane.add(BatchNormalization())

model_plane.add(MaxPooling2D(pool_size = (2,2)))
model_plane.add(Dropout(0.1))

model_plane.add(Conv2D(32, (5,5), padding = 'same', activation = 'relu'))
model_plane.add(BatchNormalization())

model_plane.add(MaxPooling2D(pool_size = (2,2)))

model_plane.add(Dropout(0.1))

model_plane.add(Flatten())

model_plane.add(Dense(128, activation = 'relu'))

model_plane.add(Dense(6, activation="tanh"))


# In[ ]:


model_plane.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = ['mae'])
model_plane.summary()


# In[ ]:


checkpoint_plane = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights_plane.hdf5', verbose=0, save_best_only=True)
store_result_plane = keras.callbacks.callbacks.CSVLogger("results_plane.csv", separator=',', append=False)


# In[ ]:


result_plane = model_plane.fit_generator(plane_training_generator, verbose=0, validation_data=plane_validation_generator,
                                                 epochs=500, 
                                                 callbacks=[early_stopping, checkpoint_plane, store_result_plane, reduce_lr])
model_plane.save('plane_trained_model.model')


# b) Dense network

# In[ ]:


checkpoint_dense_plane = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights_dense_plane.hdf5', verbose=0, save_best_only=True)
store_result_dense_plane = keras.callbacks.callbacks.CSVLogger("results_dense_plane.csv", separator=',', append=False)

model_dense_plane = Sequential()
model_dense_plane.add(Flatten(input_shape=(128, 128, 1))) 
model_dense_plane.add(Dense(2048,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_dense_plane.add(Dense(512,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_dense_plane.add(Dense(64,  kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
model_dense_plane.add(Dense(6, activation="tanh"))

model_dense_plane.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = ['mae'])

result_dense_plane = model_dense_plane.fit_generator(plane_training_generator, verbose=0, validation_data=plane_validation_generator,
                                                 epochs=500, 
                                                 callbacks=[early_stopping, checkpoint_dense_plane, store_result_dense_plane, reduce_lr])
model_dense_plane.save('dense_plane_trained_model.model')


# 4.) Projection on cylinder

# In[ ]:


cylinder_training_generator = DataGenerator("/home/pawel/git/orientation_train.db", num_features=30*30*15, dim=(128, 128, 1), batch_size=32, column="cylinder")
cylinder_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(128, 128, 1), batch_size=32, column="cylinder")

model_cylinder = Sequential()
model_cylinder.add(Conv2D(16,(5,5), padding = 'same', activation = 'relu', input_shape = (128,128,1)))
model_cylinder.add(BatchNormalization())
model_cylinder.add(MaxPooling2D(pool_size = (2,2)))
model_cylinder.add(Dropout(0.1))
model_cylinder.add(Conv2D(32, (5,5), padding = 'same', activation = 'relu'))
model_cylinder.add(BatchNormalization())
model_cylinder.add(MaxPooling2D(pool_size = (2,2)))
model_cylinder.add(Dropout(0.1))
model_cylinder.add(Flatten())
model_cylinder.add(Dense(128, activation = 'relu'))
model_cylinder.add(Dense(6, activation="tanh"))

model_cylinder.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = ['mae'])
model_cylinder.summary()

checkpoint_cylinder = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights_cylinder.hdf5', verbose=0, save_best_only=True)
store_result_cylinder = keras.callbacks.callbacks.CSVLogger("results_cylinder.csv", separator=',', append=False)

result_cylinder = model_cylinder.fit_generator(cylinder_training_generator, verbose=0, validation_data=cylinder_validation_generator,
                                                 epochs=500, 
                                                 callbacks=[early_stopping, checkpoint_cylinder, store_result_cylinder, reduce_lr])
model_cylinder.save('cylinder_trained_model.model')


# 5.) Projection on sphere

# In[ ]:


sphere_training_generator = DataGenerator("/home/pawel/git/orientation_train.db", num_features=30*30*15, dim=(128, 128, 1), batch_size=32, column="sphere")
sphere_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(128, 128, 1), batch_size=32, column="sphere")

model_sphere = Sequential()
model_sphere.add(Conv2D(16,(5,5), padding = 'same', activation = 'relu', input_shape = (128,128,1)))
model_sphere.add(BatchNormalization())
model_sphere.add(MaxPooling2D(pool_size = (2,2)))
model_sphere.add(Dropout(0.1))
model_sphere.add(Conv2D(32, (5,5), padding = 'same', activation = 'relu'))
model_sphere.add(BatchNormalization())
model_sphere.add(MaxPooling2D(pool_size = (2,2)))
model_sphere.add(Dropout(0.1))
model_sphere.add(Flatten())
model_sphere.add(Dense(128, activation = 'relu'))
model_sphere.add(Dense(6, activation="tanh"))

model_sphere.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = ['mae'])
model_sphere.summary()

checkpoint_sphere = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights_sphere.hdf5', verbose=0, save_best_only=True)
store_result_sphere = keras.callbacks.callbacks.CSVLogger("results_sphere.csv", separator=',', append=False)

result_sphere = model_sphere.fit_generator(sphere_training_generator, verbose=0, validation_data=sphere_validation_generator,
                                                 epochs=500, 
                                                 callbacks=[early_stopping, checkpoint_sphere, store_result_sphere, reduce_lr])
model_sphere.save('sphere_trained_model.model')


# 6.) Compare results

# In[ ]:


import sqlite3
import math
import time

voxel_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(9, 9, 9, 1), batch_size=1, column="voxels", shuffle=False)
pca_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(3, 3, 1), batch_size=1, column="pca", shuffle=False)
plane_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(128, 128, 1), batch_size=1, column="plane", shuffle=False)
cylinder_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(128, 128, 1), batch_size=1, column="cylinder", shuffle=False)
sphere_validation_generator = DataGenerator("/home/pawel/git/orientation_test.db", num_features=13*13*6, dim=(128, 128, 1), batch_size=1, column="sphere", shuffle=False)

def to_angle(arr):
    
    angles = math.atan2(arr[1], arr[0]), math.atan2(arr[3], arr[2]), math.atan2(arr[5], arr[4])
    return [math.degrees(x) for x in angles]

real_angles = pca_validation_generator[0][1][0]
print(to_angle(real_angles))

for saved_model, generator in [('dense_voxel_trained_model.model', voxel_validation_generator), ('cnn_voxel_trained_model.model', voxel_validation_generator), ('pca_trained_model.model', pca_validation_generator), ('plane_trained_model.model', plane_validation_generator), ('dense_plane_trained_model.model', plane_validation_generator), ('cylinder_trained_model.model', cylinder_validation_generator), ('sphere_trained_model.model', sphere_validation_generator)]:
    model = keras.models.load_model(saved_model)  #nesting & function argument 
    
    t0 = time.time()
    pred = model.predict(generator[0][0])
    
    print(saved_model + ": ", to_angle(pred[0]))


# In[ ]:


import csv
import pandas as pd

results = [
    "results_sphere.csv",
    "results_cylinder.csv",
    "results_dense_plane.csv",
    "results_plane.csv",
    "results_dense_pca.csv",
    "results_cnn_voxel.csv",
    "results_dense_voxel.csv"
    ]

all_results_list = dict()
handles = list()
for result in  results:
    name = result.replace("results_", "").split(".")[0]
    results_dict = {
        "loss": list(), 
        "val_loss": list()
    }
    with open(result, newline='') as csvfile:
        for num, row in enumerate(csv.reader(csvfile, delimiter=',', quotechar='|')):
            if num == 0:
                continue
            results_dict["loss"].append(float(row[1]))
            results_dict["val_loss"].append(float(row[3]))
    all_results_list[name] = pd.DataFrame.from_dict(results_dict)
    p = plt.plot(all_results_list[name]["val_loss"], label=name)

