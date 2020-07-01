from keras.preprocessing.image import ImageDataGenerator


def preprocess(x):
    return (x * 2.0 / 255) - 1


def load_dataset(data_dir, target_size, batch_size=32, validation_split=0.0):
    datagen = ImageDataGenerator(validation_split=validation_split, preprocessing_function=preprocess)
    training_data = datagen.flow_from_directory(data_dir,
                                                batch_size=batch_size,
                                                subset='training',
                                                shuffle=True,
                                                target_size=(target_size, target_size),
                                                interpolation="lanczos")
    validation_data = datagen.flow_from_directory(data_dir,
                                                  batch_size=batch_size,
                                                  subset='validation',
                                                  shuffle=True,
                                                  target_size=(target_size, target_size),
                                                  interpolation="lanczos")
    return training_data, validation_data
