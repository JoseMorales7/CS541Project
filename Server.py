# %%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import os
from PIL import Image, ImageOps, ImageFilter
import random
import time
import torchvision
from torchvision import models
from torchvision.transforms import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        image = Image.open(imagePath)
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
        image = Image.open(imagePath)
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
# The inference transforms are available at ViT_B_16_Weights.IMAGENET1K_V1.transforms and perform the following preprocessing operations: Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. 
# The images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. 
# Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
#models.ViT_B_16_Weights.IMAGENET1K_V1

model_name = "Inception_pretrained_balanced"
model_image_size = 224


# %%
class CityInception(torch.nn.Module):
    def __init__(self, numClasses: int, softmax:bool = True):
        super(CityInception, self).__init__()
        self.inceptionBase = torchvision.models.inception_v3(weights='DEFAULT')
        self.inceptionBase.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 216),
            torch.nn.ReLU(),
            torch.nn.Linear(216, 10)
        )
        # for param in list(self.inceptionBase.parameters())[:-1]:
        #     param.requires_grad = False
        # for param in self.inceptionBase.parameters():
        #     print(param.requires_grad)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x):
        # print(x.shape)
        logits = self.inceptionBase(x)
        # print(type(logits))
        # print(logits)
        # print(logits.shape)
        probs = self.softmax(logits.logits)
        return probs
# %%


model = CityInception(10).to(device)
# print(*list(model.children())[:-1])
# %%
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
# %%
batch_size = 64
transform = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
trainDataLoader, validDataLoader, testDataLoader = getBalancedCitiesDataLoader("/home/vislab-001/Documents/fmriMQP/data/", "./AugmentedData", transforms = transform, batchSize=batch_size)
# %%
print(len(trainDataLoader))
print(len(validDataLoader))
print(len(testDataLoader))

for i in trainDataLoader:
    image, cities, _, _ = i
    print(image.shape)
    break
# %%


from fvcore.nn import FlopCountAnalysis
valid_image = 0
for i in validDataLoader:
    valid_image, cities, _, _ = i
    valid_image = valid_image.to(device)
    break


flops = FlopCountAnalysis(model, valid_image)
print(str(flops.total()) + " flops")
# Ignore Reds
# %%


def evaluate_on_data(vit, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        total_loss = 0
        num_correct = 0.0
        num_samples = 0.0
        for data in dataloader:
            image, city, _, _ = data
            city = city.to(device)
            image = image.to(device)
            outputs = vit(image)
            loss = criterion(outputs, city)
            total_loss += loss.item()
            for i in range(len(city)):
                model_vote = 0
                answer = 0
                for j in range(len(outputs[i])):
                    if outputs[i][j] > outputs[i][model_vote]:
                        model_vote = j
                    if city[i][j] == 1:
                        answer = j
                if answer == model_vote:
                    num_correct += 1
                num_samples += 1         
    return total_loss / len(dataloader), num_correct / num_samples
# %%


num_epochs = 30
count = 0
valid_loss_array = np.zeros(num_epochs)
valid_acc_array = np.zeros(num_epochs)
train_loss_array = np.zeros(num_epochs)
for epoch in range(num_epochs):
    start = time.time()
    temp = 0
    for data in trainDataLoader:
        image, city, _, _ = data
        city = city.to(device)
        image = image.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, city)
        loss.backward()
        optimizer.step()
        end = time.time()
        count += 1
        print(str(int(end-start)) + " sec " + str(count * batch_size) + " images " + str(loss.item()) + " loss", end='\x1b\r')
    valid_loss, valid_acc = evaluate_on_data(model, validDataLoader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Valid Loss: {valid_loss}, Valid ACC: {valid_acc}')
    valid_loss_array[epoch] = valid_loss
    train_loss_array[epoch] = loss.item()
    valid_acc_array[epoch] = valid_acc
    

# %%
with open(model_name + '_valid.npy', 'wb') as f:
    np.save(f, valid_loss_array)
    

with open(model_name + '_valid_acc.npy', 'wb') as f:
    np.save(f, valid_acc_array)


with open(model_name + '_train.npy', 'wb') as f:
    np.save(f, train_loss_array)


print(evaluate_on_data(model, testDataLoader))
