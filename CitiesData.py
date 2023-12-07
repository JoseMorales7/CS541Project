# %%
import torch
from torch.utils.data import Dataset, DataLoader

from skimage import io

import numpy as np
import torch
import os


# %%
class CitiesData(Dataset):
    def __init__(self, dataParentFolder: str, dataIdxs: list, transform = None, batch_size=128):
        self.dataParentFolder = dataParentFolder
        self.transform = transform

        imagePaths = []
        for city in os.listdir(dataParentFolder):
            path = os.path.join(dataParentFolder, city)
            imagePaths.extend([os.path.join(path, imageFile).replace("\\", "/") for imageFile in os.listdir(path)])
            
        self.imagePaths = np.array(imagePaths)[dataIdxs]

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagePath = self.imagePaths[idx]
        pathSplits = imagePath.split("/")
        city = pathSplits[2]
        city = self.city_to_vector(city)
        longitude, latitude = pathSplits[3].split(",")
        latitude = latitude.split(".jpg")[0]

        image = io.imread(imagePath)
        if self.transform:
            image = self.transform(image)
    
        return image, city, float(longitude), float(latitude)
    
    def city_to_vector(self, city):
        output = np.zeros(10)
        for i in range(len(city)):
            if city == 'Atlanta':
                output[0] = 1
            elif city == 'Austin':
                output[1] = 1
            elif city == 'Boston':
                output[2] = 1
            elif city == 'Chicago':
                output[3] = 1
            elif city == 'LosAngeles':
                output[4] = 1
            elif city == 'Miami':
                output[5] = 1
            elif city == 'NewYork':
                output[6] = 1
            elif city == 'Phoenix':
                output[7] = 1
            elif city == 'SanFrancisco':
                output[8] = 1
            elif city == 'Seattle':
                output[9] = 1
        return torch.tensor(output)

# %%
def getCitiesDataLoader(dataParentFolder: str, batchSize: int = 128, transforms = None):
    cityIdxs = [0]
    totalPoints = 0
    for city in os.listdir(dataParentFolder):
        totalPoints += len(os.listdir(os.path.join(dataParentFolder, city)))
        cityIdxs.append(totalPoints)

    trainIdxs = []
    testIdxs = []
    for i in range(len(cityIdxs) - 1):
        start = cityIdxs[i]
        stop = cityIdxs[i + 1]

        num_train = int(np.round((stop - start) / 100 * 90))
        # Shuffle all training stimulus images
        idxs = np.arange(start, stop)

        np.random.shuffle(idxs)

        # Assign 90% of the shuffled stimulus images for each city to the training partition,
        # and 10% to the test partition
        trainIdxs.extend(idxs[:num_train])
        testIdxs.extend(idxs[num_train:])

    trainData = CitiesData(dataParentFolder, trainIdxs, transform=transforms)
    testData = CitiesData(dataParentFolder, testIdxs, transform=transforms)

    trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)

    return trainDataLoader, testDataLoader
    