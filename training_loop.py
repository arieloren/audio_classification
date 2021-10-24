import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from config import BATCH_SIZE

# ----------------------------
# Training Loop
# ----------------------------
def training(model, trainDataLoader,valDataLoader,testDataLoader,testData, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Loss Function, Optimizer and Scheduler
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(trainDataLoader)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }


    # Repeat for each epoch
    for epoch in range(num_epochs):

        # set the model in training mode
        model.train()


        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(trainDataLoader):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = torch.tensor(data[0]).to(device), torch.tensor(data[1]).to(device)
            labels = labels.type(torch.LongTensor)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()


            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (outputs.argmax(1) == labels).type(
                torch.float).sum().item()

            # switch off autograd for evaluation
            with torch.no_grad():
                # set the model in evaluation mode
                model.eval()

                # loop over the validation set
                for (x, y) in valDataLoader:
                    # send the input to the device
                    (x, y) = (x.to(device), y.to(device))

                    # make the predictions and calculate the validation loss
                    outputs = model(x)
                    y = y.type(torch.LongTensor)
                    totalValLoss = criterion(outputs, y)

                    # calculate the number of correct predictions
                    valCorrect += (outputs.argmax(1) == y).type(
                        torch.float).sum().item()

            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps

            # calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(trainDataLoader.dataset)
            valCorrect = valCorrect / len(valDataLoader.dataset)

            # update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["val_acc"].append(valCorrect)

            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, num_epochs))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                avgValLoss, valCorrect))


        # we can now evaluate the network on the test set
        print("[INFO] evaluating network...")

        # turn off autograd for testing evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()

            # initialize a list to store our predictions
            preds = []

            # loop over the test set
            for (x, y) in testDataLoader:
                # send the input to the device
                x = x.to(device)

                # make the predictions and add them to the list
                outputs = model(x)
                preds.extend(outputs.argmax(axis=1).cpu().numpy())

        # generate a classification report
        print(classification_report(testData.dataset.df['classID'].loc[testData.indices].to_numpy(),
                                    np.array(preds), target_names=list(
                testData.dataset.df['label'].drop_duplicates())))  # need to add the clasees column

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label="train_loss")
        plt.plot(H["val_loss"], label="val_loss")
        plt.plot(H["train_acc"], label="train_acc")
        plt.plot(H["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig('output/plot.png')

        # serialize the model to disk
        torch.save(model,'output/model.pth')





    print('Finished Training')


