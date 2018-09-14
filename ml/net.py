from keras.models import Sequential
from keras.layers import (Dense, Activation, Conv2D, SeparableConv2D,
                            BatchNormalization, Activation, AveragePooling2D,
                            Flatten, Reshape, Dropout)
from keras.activations import relu, softmax
from keras import optimizers

model = Sequential()

def addConv(filters,kernel_size,strides):
    model.add(Conv2D(
        filters = filters,
        kernel_size =  kernel_size,
        strides = strides,
        padding = "same",
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

def addConvDW(filters,kernel_size,strides):
    model.add(SeparableConv2D(
        filters = filters,
        kernel_size =  kernel_size,
        strides = strides,
        padding = "same",
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

#Architecture
model.add(Conv2D(
    filters = 32,
    kernel_size = (3,3),
    strides = 2,
    padding = "same",
    input_shape = (224,224,3)
))
model.add(BatchNormalization())
model.add(Activation('relu'))
addConvDW(32, (3,3), 1)
addConv(64, (1,1), 1)
addConvDW(64, (3,3), 2)
addConv(128, (1,1), 1)
addConvDW(128, (3,3), 1)
addConv(128, (1,1), 1)
addConvDW(128, (3,3), 2)
addConv(256, (1,1), 1)
addConvDW(256, (3,3), 1)
addConv(256, (1,1), 1)
addConvDW(256, (3,3), 2)
addConv(512, (1,1), 1)
for _ in range(5):
    addConvDW(512, (3,3), 1)
    addConv(512, (1,1), 1)
addConvDW(512, (3,3), 2)
addConv(1024, (1,1), 1)
addConvDW(1024, (3,3), 1)
addConv(1024, (1,1), 1)
model.add(AveragePooling2D(
    pool_size = (7,7),
    strides = 1,
    padding = "valid"
))
model.add(Flatten())
model.add(Dense(
    units = 102
))
model.add(Activation('softmax'))
adam = optimizers.Adam(lr=0.001)
model.compile(
    optimizer = adam,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
