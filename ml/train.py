from net import model
from generators import trainGen, testGen, trainLabels, dictIterator
import keras, os

def filecount(num):
    directory = dictIterator(trainLabels, num)
    return len(os.listdir('./101_ObjectCategories/'+directory))

tensorboard = keras.callbacks.TensorBoard(
    log_dir='./Graph',
    histogram_freq=0,
    write_graph=True,
    write_images=True,
    batch_size=32
)

try:
    for x,y in model.fit_generator(trainGen,epochs=350,callbacks=[tensorboard],validation_data=testGen):
        model.train_on_batch(x,y)
except:
    pass
choice = input("\nSave model (y/n):")
if choice.lower() == "y":
    model.save('./models/model.h5')
    print("Model saved")
