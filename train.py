from create_metadata import load_data_from_folder
import config
import custom_data_loader
from torch.utils.data import random_split
import torch
from model import AudioClassifier
from training_loop import training
from config import BATCH_SIZE

# Take relevant columns

df= load_data_from_folder(config.path)



from torch.utils.data import random_split

myds = custom_data_loader.SoundDS(df)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_test = num_items - num_train


(trainData, testData) = random_split(myds,
	[num_train, num_test],
	generator=torch.Generator().manual_seed(42))

old_num_train = num_train
num_train = round(old_num_train * 0.8)
num_val = old_num_train - num_train

(trainData, valData) = random_split(trainData,
	[num_train, num_val],
	generator=torch.Generator().manual_seed(42))


# initialize the train, validation, and test data loaders
trainDataLoader = torch.utils.data.DataLoader(trainData, shuffle=True,
	batch_size=2)
valDataLoader = torch.utils.data.DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = torch.utils.data.DataLoader(testData, batch_size=BATCH_SIZE)


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device
num_epochs = 10  # Just for demo, adjust this higher.
training(myModel, trainDataLoader,valDataLoader,testDataLoader,testData, num_epochs)
# Run inference on trained model with the validation set
#inference(myModel, val_dl)