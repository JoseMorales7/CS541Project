# %%
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import os
import torchvision
from torchvision import models
from torchvision.transforms import transforms
import torch
import numpy as np
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import time
from fvcore.nn import FlopCountAnalysis


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


# The inference transforms are available at ViT_B_16_Weights.IMAGENET1K_V1.transforms and perform the following preprocessing operations: Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. 
# The images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. 
# Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

#models.ViT_B_16_Weights.IMAGENET1K_V1
model_name = "InceptionV3"
model_image_size = 224

class CityInception(torch.nn.Module):
    def __init__(self, numClasses: int, softmax:bool = True):
        super(CityInception, self).__init__()
        self.inceptionBase = torchvision.models.inception_v3(weights=None)
        self.inceptionBase.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=numClasses),
            torch.nn.ReLU()
        )
        for param in list(self.inceptionBase.parameters())[:-1]:
            param.requires_grad = False
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


model = CityInception(10).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

batch_size = 512
transform = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
trainDataLoader, testDataLoader = getCitiesDataLoader("/home/vislab-001/Documents/fmriMQP/data/", transforms = transform, batchSize=batch_size)

test_image = 0
for i in testDataLoader:
    test_image, cities, _, _ = i
    test_image = test_image.to(device)
    break


flops = FlopCountAnalysis(model, test_image)
print(str(flops.total()) + " flops")


def evaluate_on_data(model, dataloader, criterion):
    with torch.no_grad():
        total_loss = 0
        num_correct = 0.0
        num_samples = 0.0
        for data in dataloader:
            image, city, _, _ = data
            city = city.to(device)
            image = image.to(device)
            outputs = model(image)
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


num_epochs = 15
count = 0
test_loss_array = np.zeros(num_epochs)
test_acc_array = np.zeros(num_epochs)
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
    test_loss, test_acc = evaluate_on_data(model, testDataLoader, criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss}, Test ACC: {test_acc}')
    test_loss_array[epoch] = test_loss
    train_loss_array[epoch] = loss.item()
    test_acc_array[epoch] = test_acc
    

with open(model_name + '_test.npy', 'wb') as f:
    np.save(f, test_loss_array)
    

with open(model_name + '_test_acc.npy', 'wb') as f:
    np.save(f, test_acc_array)
    

with open(model_name + '_train.npy', 'wb') as f:
    np.save(f, train_loss_array)


# Ignore Reds
    

# import torch
# import torchvision
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
# import os
# # %%
# class CitiesData(Dataset):
#     def __init__(self, dataParentFolder: str, dataIdxs: list, transform = None, batch_size=128):
#         self.dataParentFolder = dataParentFolder
#         self.transform = transform
#         imagePaths = []
#         for city in os.listdir(dataParentFolder):
#             path = os.path.join(dataParentFolder, city)
#             imagePaths.extend([os.path.join(path, imageFile).replace("\\", "/") for imageFile in os.listdir(path)])
#         self.imagePaths = np.array(imagePaths)[dataIdxs]
#     def __len__(self):
#         return len(self.imagePaths)
#     def __getitem__(self, idx: int):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         imagePath = self.imagePaths[idx]
#         pathSplits = imagePath.split(self.dataParentFolder)[1].split("/")
#         city = pathSplits[0]
#         longitude, latitude = pathSplits[1].split(",")
#         latitude = latitude.split(".jpg")[0]
#         image = Image.open(imagePath)
#         if self.transform:
#             image = self.transform(image)
#         return image, city, float(longitude), float(latitude)
# # %%

# def getCitiesDataLoader(dataParentFolder: str, batchSize: int = 128, transforms = None):
#     cityIdxs = [0]
#     totalPoints = 0
#     for city in os.listdir(dataParentFolder):
#         totalPoints += len(os.listdir(os.path.join(dataParentFolder, city)))
#         cityIdxs.append(totalPoints)
#     trainIdxs = []
#     testIdxs = []
#     for i in range(len(cityIdxs) - 1):
#         start = cityIdxs[i]
#         stop = cityIdxs[i + 1]
#         num_train = int(np.round((stop - start) / 100 * 90))
#         # Shuffle all training stimulus images
#         idxs = np.arange(start, stop)
#         np.random.shuffle(idxs)
#         # Assign 90% of the shuffled stimulus images for each city to the training partition,
#         # and 10% to the test partition
#         trainIdxs.extend(idxs[:num_train])
#         testIdxs.extend(idxs[num_train:])
#     trainData = CitiesData(dataParentFolder, trainIdxs, transform=transforms)
#     testData = CitiesData(dataParentFolder, testIdxs, transform=transforms)
#     trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
#     testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=True)
#     return trainDataLoader, testDataLoader


# class CityInception(torch.nn.Module):
#     def __init__(self, numClasses: int, softmax:bool = True):
#         super(CityInception, self).__init__()
#         self.inceptionBase = torchvision.models.inception_v3(weights='DEFAULT')
#         self.inceptionBase.fc = torch.nn.Linear(in_features=2048, out_features=numClasses)
#         for param in list(self.inceptionBase.parameters())[:-1]:
#             param.requires_grad = False
#         # for param in self.inceptionBase.parameters():
#         #     print(param.requires_grad)
#         self.softmax = torch.nn.Softmax(dim=-1)
#     def forward(self, x):
#         # print(x.shape)
#         logits = self.inceptionBase(x)
#         # print(type(logits))
#         # print(logits)
#         # print(logits.shape)
#         probs = self.softmax(logits.logits)
#         return probs

# def city_to_vector(city):
#     output = np.zeros(shape=(len(city), 10))
#     for i in range(len(city)):
#         if city[i] == 'Atlanta':
#             output[i][0] = 1
#         elif city[i] == 'Austin':
#             output[i][1] = 1
#         elif city[i] == 'Boston':
#             output[i][2] = 1
#         elif city[i] == 'Chicago':
#             output[i][3] = 1
#         elif city[i] == 'LosAngeles':
#             output[i][4] = 1
#         elif city[i] == 'Miami':
#             output[i][5] = 1
#         elif city[i] == 'NewYork':
#             output[i][6] = 1
#         elif city[i] == 'Phoenix':
#             output[i][7] = 1
#         elif city[i] == 'SanFrancisco':
#             output[i][8] = 1
#         elif city[i] == 'Seattle':
#             output[i][9] = 1
#     return torch.tensor(output)

# numEpochs = 1
# batchSize = 1024
# learningRate = 0.00001

# device = "cuda:1" if torch.cuda.is_available() else "cpu"

# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(342),
#     torchvision.transforms.CenterCrop(299),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
# ])

# trainDataLoader, testDataLoader = getCitiesDataLoader(batchSize=batchSize, dataParentFolder="/home/vislab-001/Documents/fmriMQP/data/", transforms=transforms)

# model = CityInception(10).to(device)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# lossPerEpoch = []
# for epoch in range(numEpochs):
#     lossPerBatch = []
#     for image, cities, _, _ in trainDataLoader:
#         image = image.to(device)
#         cities = city_to_vector(cities).to(device)
#         optimizer.zero_grad()
#         predicted = model(image)
#         loss = criterion(predicted, cities)
#         lossPerBatch.append(loss.item())
#         loss.backward()
#         optimizer.step()
#     avgLoss = sum(lossPerBatch) / len(lossPerBatch)
#     lossPerEpoch.append()
#     print("Loss for epoch {}: {}".format(epoch, avgLoss))


# lossPerEpoch = np.array(lossPerEpoch)
# np.save("./results/inception/lossPerEpoch.npy", lossPerEpoch)















        

