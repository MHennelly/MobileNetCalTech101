from keras.preprocessing.image import ImageDataGenerator

trainGen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 60,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = 0.5
).flow_from_directory(
    directory = './101_ObjectCategories',
    target_size = (224,224),
    batch_size = 32
)

trainLabels = trainGen.class_indices

testGen = ImageDataGenerator(
    rescale = 1./255
).flow_from_directory(
    directory = './101_ObjectCategories',
    target_size = (224,224),
    batch_size = 1
)

testLabels = testGen.class_indices

# secretGen = ImageDataGenerator(
#     rescale = 1./255,
#     rotation_range = 45
# ).flow_from_directory(
#     directory = './secret_test',
#     target_size = (224,224),
#     class_mode = None,
#     batch_size = 1
# )

def dictIterator(dict, index):
    for key in dict:
        if dict[key] == index:
            return key
