# %%
import torch
from torch.utils.data import Dataset, DataLoader

from skimage import io

import numpy as np
import torch
import os
from PIL import Image, ImageOps, ImageFilter
import random
import time



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
        pathSplits = imagePath.split(self.dataParentFolder)[1].split("/")
        city = pathSplits[0]
        city =self.city_to_vector(city)
        longitude, latitude = pathSplits[1].split(",")
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
    validIdxs = []
    testIdxs = []
    for i in range(len(cityIdxs) - 1):
        start = cityIdxs[i]
        stop = cityIdxs[i + 1]

        num_train = int(np.round((stop - start) / 100 * 90))
        num_valid = int(round(num_train * 0.025))

        num_train -= num_valid
        num_valid += num_train

        # Shuffle all training stimulus images
        idxs = np.arange(start, stop)

        np.random.shuffle(idxs)

        # Assign 90% of the shuffled stimulus images for each city to the training partition,
        # and 10% to the test partition
        trainIdxs.extend(idxs[:num_train])
        validIdxs.extend(idxs[num_train:num_valid])
        testIdxs.extend(idxs[num_valid:])

    trainData = CitiesData(dataParentFolder, trainIdxs, transform=transforms)
    validData = CitiesData(dataParentFolder, validIdxs, transform=transforms)
    testData = CitiesData(dataParentFolder, testIdxs, transform=transforms)

    trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    validDataLoader = DataLoader(validData, batch_size=batchSize, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)

    return trainDataLoader, validDataLoader, testDataLoader
    

# %%
def getBalancedCitiesDataLoader(dataParentFolder: str, augmentedParentFolder: str, batchSize: int = 128, transforms = None, balanced_size=200000):
    # Generate a dataset of 10k images per class - 100k images total
    class_balanced_size = int(balanced_size / 10)
    start = time.time()
    imagePaths = np.empty(balanced_size, dtype='object')
    count = 0
    for city in os.listdir(dataParentFolder):
        path = os.path.join(dataParentFolder, city)
        city_size = len(os.listdir(path))
        if city_size >= class_balanced_size:
            print("here")
            # Chop Data
            path_list = os.listdir(path)
            for i in range(class_balanced_size):
                imageFile = path_list[i]
                imagePaths[count] = os.path.join(path, imageFile).replace("\\", "/")
                count += 1
            end = time.time()
            print(end-start)
            print(class_balanced_size)
        else:
            # First add original data
            path_list = os.listdir(path)
            for i in range(city_size):
                imageFile = path_list[i]
                imagePaths[count] = os.path.join(path, imageFile).replace("\\", "/")
                count += 1 

            # Generate new data
            augmented_path = os.path.join(augmentedParentFolder, city)
            if len(os.listdir(augmented_path)) < class_balanced_size - city_size:
                data_generator(dataParentFolder, augmentedParentFolder, city, class_balanced_size - city_size)

            augmented_path_list = os.listdir(augmented_path)
            for i in range(class_balanced_size - city_size):
                imageFile = augmented_path_list[i]
                imagePaths[count] = os.path.join(augmented_path, imageFile).replace("\\", "/")
                count += 1

    trainIdxs = []
    validIdxs = []
    testIdxs = []
    
    for i in range(10):
        start = i * class_balanced_size
        stop = (i+1) * class_balanced_size
        
        num_train = int(np.round((stop - start) / 100 * 90))
        num_valid = int(round(num_train * 0.025))

        num_train -= num_valid
        num_valid += num_train

        # Shuffle all training stimulus images
        idxs = np.arange(start, stop)

        np.random.shuffle(idxs)

        # Assign 90% of the shuffled stimulus images for each city to the training partition,
        # and 10% to the test partition
        trainIdxs.extend(idxs[0:num_train])
        validIdxs.extend(idxs[num_train:num_valid])
        testIdxs.extend(idxs[num_valid:])

    train_imagePaths = np.array(imagePaths)[trainIdxs]
    valid_imagePaths = np.array(imagePaths)[validIdxs]
    test_imagePaths = np.array(imagePaths)[testIdxs]

    trainData = AugmentedCitiesData(dataParentFolder, augmentedParentFolder, train_imagePaths, transform=transforms)
    validData = AugmentedCitiesData(dataParentFolder, augmentedParentFolder, valid_imagePaths, transform=transforms)
    testData = AugmentedCitiesData(dataParentFolder, augmentedParentFolder, test_imagePaths, transform=transforms)

    trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    validDataLoader = DataLoader(validData, batch_size=batchSize, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)

    return trainDataLoader, validDataLoader, testDataLoader


# %%
class AugmentedCitiesData(Dataset):
    def __init__(self, dataParentFolder: str, augmentedParentFolder: str, imagePaths, transform = None, batch_size=128):
        self.dataParentFolder = dataParentFolder
        self.augmentedParentFolder = augmentedParentFolder
        self.transform = transform
        self.imagePaths = imagePaths
        

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagePath = self.imagePaths[idx]
        pathSplits = - 1
        if len(imagePath.split(self.dataParentFolder)) >1:
            pathSplits = imagePath.split(self.dataParentFolder)[1].split("/")
        else:
            pathSplits = imagePath.split(self.augmentedParentFolder)[1].split("/")

        city = pathSplits[0]
        city =self.city_to_vector(city)
        longitude, latitude = pathSplits[1].split(",")
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
def data_generator(original_data_folder, augmented_data_folder, city, amount):
    
    path = os.path.join(original_data_folder, city)
    count = 0
    while count < amount:
        for imageFile in os.listdir(path):
            imagePath = os.path.join(path, imageFile).replace("\\", "/")

            # Transform image - slight rotations and 
            image = Image.open(imagePath)

            rand = random.randint(0,9)
            if rand <= 5:
                rand2= random.randint(-3,3)
                image = image.rotate(rand2)
            else:
                image = image.filter(ImageFilter.GaussianBlur(1))


            #Write Image to AugmentedData
            image.save(os.path.join(os.path.join(augmented_data_folder, city), str(count) + ".0,1.0.jpg").replace("\\", "/"))

            count += 1
            if count >= amount:
                break


