from skimage import io
import h5py
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/model_Dora.h5'
model = keras.models.load_model(modelPath, compile=True)


frames = [0, 27043]  # [7704, 9358, 17657, 28428, 27043]
filePath = ('//lebnas1.epfl.ch/microsc125/Watchdog/Model/Drp1.h5')
fileHandle = h5py.File(filePath, 'r')
item = list(fileHandle.keys())[0]
print(item)
drp1 = fileHandle.get(item)

drp1_frames = drp1[frames]
del(drp1)

filePath = ('//lebnas1.epfl.ch/microsc125/Watchdog/Model/Mito.h5')
fileHandle = h5py.File(filePath, 'r')
item = list(fileHandle.keys())[0]
print(item)
mito = fileHandle.get(item)

mito_frames = mito[frames]
del(mito)

print(mito_frames.shape)

for frame in range(mito_frames.shape[0]):
    mito_img = mito_frames[frame]
    drp1_img = drp1_frames[frame]
    mito_img = mito_img.reshape(1, mito_img.shape[0], mito_img.shape[0], 1)
    drp1_img = drp1_img.reshape(1, drp1_img.shape[0], drp1_img.shape[0], 1)
    inputData = inputData = np.stack((mito_img, drp1_img), 3)
    print(inputData.shape)
    outputPredict = model.predict_on_batch(inputData)
    plt.imshow(outputPredict[0, :, :, 0], vmin=0, vmax=100)
    io.imsave("C:/Users/stepp/Desktop/Pred.tif", outputPredict[0, :, :, 0])
    plt.show()