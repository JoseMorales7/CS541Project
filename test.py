class CityInception(torch.nn.Module):
    def __init__(self, numClasses: int, softmax:bool = True):
        super(CityInception, self).__init__()
        self.inceptionBase = torchvision.models.inception_v3(weights='DEFAULT')
        self.inceptionBase.fc = torch.nn.Linear(in_features=2048, out_features=numClasses)
        for param in list(self.inceptionBase.parameters())[:-1]:
            param.requires_grad = False
        # for param in self.inceptionBase.parameters():
        #     print(param.requires_grad)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x):
        # print(x.shape)
        logits = self.inceptionBase(x)
        probs = self.softmax(logits)
        return probs






model = CityInception(10).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(numEpochs):
    for image, cities, _, _ in trainDataLoader:
        image = image.to(device)
        cities = city_to_vector(cities).to(device)
        optimizer.zero_grad()
        predicted = model(image)
        loss = criterion(predicted, cities)
        print("Loss for epoch {}: {}".format(epoch, loss.item()))
        loss.backward()
        optimizer.step()
        

