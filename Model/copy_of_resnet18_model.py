




class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = image.float()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def train_test_split(all_data, test_portion = 0.3, seed = 0):
  training_size = int((1 - test_portion) * len(all_data))
  test_size = len(all_data) - training_size
  training_indices, test_indices = random_split(
                              range(len(all_data)), 
                              [training_size, test_size],
                              generator=torch.Generator().manual_seed(seed))
  training_data = Subset(all_data, training_indices)
  test_data = Subset(all_data, test_indices)
  return training_data, test_data

from google.colab import drive
drive.mount('/content/drive')

# Download training data from open datasets.

all_data = CustomImageDataset(
    annotations_file="/content/drive/MyDrive/Ai builders/annotations_file.csv", 
    img_dir="/content/drive/MyDrive/Ai builders/pig pics cleaned", 
    transform=Resize(size=128), 
    target_transform=None
)

training_data, test_data = train_test_split(all_data)

print("All data size: " + str(len(all_data)))
print("Training size: " + str(len(training_data)) + "\tTest size: " + str(len(test_data)))

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
from torchvision import models
model = models.resnet18(pretrained=True)
# model = models.resnet50(pretrained=True)
model.fc =nn.Linear(512,2)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # print(f"X.Shape: {X.shape}")
            pred = model(X)
            # print(f"pred: {pred}")
            # print(f"pred shape: {pred.shape}")
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "resnet_50_model_128.pth")
print("Saved PyTorch Model State to model.pth")

from sklearn.metrics import classification_report
def get_incorrect_predictions(model, test_dataloader):
  out1 = []
  out2 = []
  
  for i in range(len(test_data)):
    model.eval()
    x, y = test_data[i][0], test_data[i][1]
    
    # Turn it into 4D tensor [N, C, H, W] just like in dataloader
    x = x.unsqueeze(0)
    
    with torch.no_grad():
      
      pred = model(x)
      predicted, actual = pred[0].argmax(0), y
   
      out1.append(predicted.item())
      out2.append(y)
       
  return out1, out2
y_true,y_pred = get_incorrect_predictions(model,test_dataloader)
print(classification_report(y_true,y_pred))
