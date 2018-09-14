from keras.models import load_model
from generators import testGen, testLabels, dictIterator #, secretGen
import numpy as np
import matplotlib.pyplot as plt

model = load_model('./models/model.h5')

def testImage(x,y,dict):
    prediction = dictIterator(dict, np.argmax(model.predict(x), -1)[0])
    answer = dictIterator(dict, np.argmax(y, -1)[0])
    print("Prediction:",prediction)
    print("Answer:",answer,"\n")
    plt.imshow(x.reshape(224,224,3))
    plt.show()

try:
    # count = 0
    # for x in secretGen:
    #     plt.imshow(x.reshape(224,224,3))
    #     plt.show()
    #     num = np.argmax(model.predict(x), -1)[0]
    #     prediction = dictIterator(testLabels,num)
    #     print("Prediction:",prediction)
    #     count += 1
    #     if count == len(secretGen):
    #         break
    for x,y in testGen:
        testImage(x,y,testLabels)
except Exception as e: print(e)
print("Done testing")
