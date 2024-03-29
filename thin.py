from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
import keras
import numpy as np
from keras.layers import Dense, Dropout, Activation, InputLayer
from keras.optimizers import Adam
from keras.optimizers import Nadam
model5 = Sequential([
InputLayer(input_shape=(1,28,28)),
keras.layers.Flatten(),
Dropout(0.4),
keras.layers.LeakyReLU(alpha=0.1),
Dense(500),
keras.layers.LeakyReLU(alpha=0.1),
Dense(290),
keras.layers.LeakyReLU(alpha=0.1),
Dense(10,activation='softmax')
])
from keras.models import Model, Sequential
model5 = Sequential([
InputLayer(input_shape=(1,28,28)),
keras.layers.Flatten(),
Dropout(0.4),
keras.layers.LeakyReLU(alpha=0.1),
Dense(500),
keras.layers.LeakyReLU(alpha=0.1),
Dense(290),
keras.layers.LeakyReLU(alpha=0.1),
Dense(10,activation='softmax')
])
model5.compile(optimizer='nadam', loss='categorical_crossentropy',metrics=['accuracy'])
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train/256.0
x_train = x_train.reshape(60000,1,28,28)
x_test = x_test/256.0
x_test = x_test.reshape(10000,1,28,28)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
datagen = ImageDataGenerator(rotation_range=15,height_shift_range=3,width_shift_range=3,data_format='channels_first')
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=15,height_shift_range=3,width_shift_range=3,data_format='channels_first')
model5.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
calbak = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=25, verbose=10, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
model5.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
x1 = Input(shape=(1,28,28))
x2 = Dense(32)(x1)
x2 = Flatten()(x1)
from keras.layers import Flatten, Concatenate
x2 = Flatten()(x1)
x2 = Dense(200)(x2)
x2 = Flatten()(x1)
x1 = Flatten()(x1)
x2 = Dense(200)(x2)
x3 = Concatenate()([x1,x2])
x4 = Dense(100)(x3)
out = Dense(10,activation='softmax')(x4)
model6 = Model(inputs=x1,outputs=out)
inp = Input(shape=(1,28,28))
x1 = Flatten()(inp)
model6 = Model(inputs=x1,outputs=out)
model6 = Model(inputs=inp,outputs=out)
x1 = Flatten()(inp)
x2 = Dense(200)(x1)
x3 = Concatenate()([x1,x2])
x4 = Dense(100)(x3)
x5 = Concatenate([x2,x4])
x5 = Concatenate()([x2,x4])
out = Dense(10,activation='softmax')(x5)
model6 = Model(inputs=inp,outputs=out)
model6.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model6.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
inp = Input(shape=(1,28,28))
x1 = Flatten()(inp)
x11 = Dropout(0.4)(x1)
x1o = LeakyReLU(alpha=0.1)(x11)
x2 = Dense(200)(x1o)
x2o = LeakyReLU(alpha=0.1)(x2)
x3 = Concatenate()([x1,x2])
x3o = LeakyReLU(alpha=0.1)(x3)
x3 = Concatenate()([x1o,x2o])
x3o = LeakyReLU(alpha=0.1)(x3)
x4 = Dense(100)(x3o)
x2 = Dense(500)(x1o)
x4 = Dense(290)(x3o)
x4o = LeakyReLU(alpha=0.1)(x4)
x5 = Concatenate()([x2o,x4o])
x5o = LeakyReLU(alpha=0.1)(x5)
out = Dense(10,activation='softmax')(x5o)
model6 = Model(inputs=inp,outputs=out)
model6.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model6.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
inp = Input(shape=(1,28,28))
x1 = Flatten()(inp)
x11 = Dropout(0.4)(x1)
x1o = LeakyReLU(alpha=0.1)(x11)
x2 = Dense(500)(x1o)
x2o = LeakyReLU(alpha=0.1)(x2)
x3 = Dense(290)(x2o)
x4 = Concatenate()([x2,x3])
x4o = LeakyReLU(alpha=0.1)(x4)
out = Dense(10,activation='softmax')(x4o)
model6 = Model(inputs=inp,outputs=out)
model6.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model6.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
x2 = Dense(300)(x1o)
x3 = Dense(190)(x2o)
x2x3 = Concatenate()([x2,x3])
x3o = LeakyReLU(alpha=0.1)(x3)
x4 = Dense(100)(x3o)
x2x3o = LeakyReLU(alpha=0.1)(x2x3)
x4o = LeakyReLU(alpha=0.1)(x4)
out = Dense(10,activation='softmax')(x4o)
out1 = Dense(10,activation='softmax')(x4o)
out2 = Dense(10,activation='softmax')(x3o)
model7 = Model(inputs=inp,outputs=[out1,out2])
model6.compile(optimizer='adam', loss='categorical_crossentropy',loss_weights=[1.0,0.1],metrics=['accuracy'])
model7.compile(optimizer='adam', loss='categorical_crossentropy',loss_weights=[1.0,0.1],metrics=['accuracy'])
model7.fit_generator(datagen.flow(x_train,[y_train,y_train],batch_size=3000),validation_data=(x_test,[y_test,y_test]),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
model7.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
model7.fit_generator(datagen.flow(x_train,[y_train,y_train],batch_size=3000),validation_data=(x_test,[y_test,y_test]),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
model7.fit_generator(datagen.flow([x_train],[y_train,y_train],batch_size=3000),validation_data=([x_test],[y_test,y_test]),steps_per_epoch=20,epochs=10000, callbacks=[calbak])
datagen
datagen = ImageDataGenerator(rotation_range=15,height_shift_range=3,width_shift_range=3,data_format='channels_first')
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=60000)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=600000)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=60000)
td1 = datagen.flow(x_train,y_train,batch_size=600000)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=60000)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=6000)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=600)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=60)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=6)
(x_t1, y_t1) = datagen.flow(x_train,y_train,batch_size=6000)
x_t1, y_t1 = datagen.flow(x_train,y_train,batch_size=6000)
datagen.flow(x_train,y_train,batch_size=6000)
(x_t1, y_t1) = list(datagen.flow(x_train,y_train,batch_size=6000))
