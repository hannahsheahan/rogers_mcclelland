"""
Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 19/11/2020
Notes: N/A
Issues: N/A
"""
import numpy as np
import sys
import constants as const

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import random
from datetime import datetime


class CreateDataset(Dataset):
    """A class to hold a dataset."""

    def __init__(self, dataset, transform=None):
        """
        Args:
            datafile (string): name of numpy datafile
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.index = dataset['index']
        self.label = dataset['label']
        self.input_item = dataset['input_item']
        self.input_context = dataset['input_context']
        self.words = dataset['words']
        self.domains = dataset['domains']
        self.data = {'index':self.index, 'label':self.label,  'input_item':self.input_item, 'input_item':self.input_context}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # for retrieving either a single sample of data, or a subset of data
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'index':self.index[idx], 'label':self.label[idx].flatten(), 'input_item':self.input_item[idx], 'input_context':self.input_context[idx]}
        return sample


def batch_to_torch(originalimages):
    """Convert the input batch to a torch tensor"""
    #originalimages = originalimages.unsqueeze(1)   # change dim for the convnet
    originalimages = originalimages.type(torch.FloatTensor)  # convert torch tensor data type
    return originalimages


def train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    """Train a neural network on the training set."""

    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
        input_item, input_context, labels = batch_to_torch(data['input_item']), batch_to_torch(data['input_context']), data['label'].type(torch.FloatTensor)
        output = model(input_item, input_context)
        output = np.squeeze(output, axis=1)

        loss = criterion(output, labels)
        loss.backward()         # passes the loss backwards to compute the dE/dW gradients
        optimizer.step()        # update our weights

        # evaluate performance
        train_loss += loss.item()
        if (torch.abs(output-labels) < args.correct_threshold).all():
            correct += 1

    train_loss /= len(train_loader.dataset)
    accuracy = (correct / len(train_loader.dataset)) * 100
    return train_loss, accuracy


def test(args, model, device, test_loader, criterion, printOutput=True):
    """
    Test a neural network on the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
            input_item, input_context, labels = batch_to_torch(data['input_item']), batch_to_torch(data['input_context']), data['label'].type(torch.FloatTensor)
            output = model(input_item, input_context)
            output = np.squeeze(output, axis=1)
            test_loss += criterion(output, labels).item()

            if (torch.abs(output-labels) < args.correct_threshold).all():
                correct += 1

    test_loss /= len(test_loader.dataset)
    accuracy = (correct / len(test_loader.dataset)) * 100
    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


class RM_Net(nn.Module):
    """
    This is a network with the architecture shown in Rogers/McClelland 2008.
    """
    def __init__(self, D_item_in, D_context_in, D_out, D_h_item, D_h_context, D_h_combined):
        super(RM_Net, self).__init__()
        self.h_item_size = D_h_item
        self.h_context_size = D_h_context
        self.h_combined_size = D_h_combined

        self.fc_item_to_hitem = nn.Linear(D_item_in, self.h_item_size)  # size input, size output
        self.fc_context_to_hcontext = nn.Linear(D_context_in, self.h_context_size)
        self.fc_hitem_to_combined = nn.Linear(self.h_item_size, self.h_combined_size)
        self.fc_hcontext_to_combined = nn.Linear(self.h_context_size, self.h_combined_size)
        self.fc_combined_to_out = nn.Linear(self.h_combined_size, D_out)

    def forward(self, x_item, x_context):

        # standard relu throughout case (use with MSE loss)
        #self.hitem_activations = F.relu(self.fc_item_to_hitem(x_item))
        #self.hcontext_activations = F.relu(self.fc_context_to_hcontext(x_context))
        #self.combined_activations = F.relu(self.fc_hitem_to_combined(self.hitem_activations) + self.fc_hcontext_to_combined(self.hcontext_activations))
        #self.output = F.relu(self.fc_combined_to_out(self.combined_activations))

        # alternative linear + sigmoid at end case (use with BCE loss)
        self.hitem_activations = self.fc_item_to_hitem(x_item)
        self.hcontext_activations = self.fc_context_to_hcontext(x_context)
        self.combined_activations = self.fc_hitem_to_combined(self.hitem_activations) + self.fc_hcontext_to_combined(self.hcontext_activations)
        self.output = torch.sigmoid(self.fc_combined_to_out(self.combined_activations))

        # alternative relu + sigmoid on final layer (use with BCE loss)
        #self.hitem_activations = F.relu(self.fc_item_to_hitem(x_item))
        #self.hcontext_activations = F.relu(self.fc_context_to_hcontext(x_context))
        #self.combined_activations = F.relu(self.fc_hitem_to_combined(self.hitem_activations) + self.fc_hcontext_to_combined(self.hcontext_activations))
        #self.output = torch.sigmoid(self.fc_combined_to_out(self.combined_activations))

        return self.output

    def get_activations(self, x_item, x_context):
        self.forward(x_item, x_context)  # update the activations with the particular input
        return self.combined_activations, self.hcontext_activations, self.hitem_activations


def init_weights(m):
    """Initialise network weights with custom settings.
      - Keep gain small to ensure small and low variance weights.
      """
    if type(m) == nn.Linear:
        gain = 1.0
        torch.nn.init.xavier_uniform(m.weight, gain)


def print_progress(i, numiter):
    """This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""
    j = i/numiter
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()


def log_performance(writer, epoch, train_loss, test_loss, train_accuracy, test_accuracy):
    """ Write out the training and testing performance for this epoch to tensorboard.
          - 'writer' is a SummaryWriter instance
    """
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)


def get_activations(args, trained_model, test_loader):
    """ This will determine the hidden unit activations for each input pair in the train/test set."""

    #  pass each input through the network and see what happens to the hidden layer activations
    item_activations = np.zeros((args.n_unique, args.D_h_item))
    context_activations = np.zeros((args.n_unique, args.D_h_context))
    combined_activations = np.zeros((args.n_unique, args.D_h_combined))
    trained_model.eval()
    with torch.no_grad():
        for sample_idx, data in enumerate(test_loader):
            input_item, input_context, labels = batch_to_torch(data['input_item']), batch_to_torch(data['input_context']), data['label'].type(torch.FloatTensor)
            combined_act, context_act, item_act = trained_model.get_activations(input_item, input_context)
            item_activations[sample_idx] = item_act
            context_activations[sample_idx] = context_act
            combined_activations[sample_idx] = combined_act

    activations = [item_activations, context_activations, combined_activations]
    return activations


def get_model_name(args):
    """Determine the correct name for the model and analysis files."""
    hiddensizes = '_' + str(args.D_h_item) + '_' + str(args.D_h_context) + '_' + str(args.D_h_combined)
    model_name = const.MODEL_DIRECTORY + str(args.id) + '_model' + hiddensizes + '.pth'
    analysis_name = const.ANALYSIS_DIRECTORY + str(args.id) + '_model_analysis' + hiddensizes + '.npy'
    return model_name, analysis_name


def train_network(args, device, trainset, testset):
    """
    This function performs the train/test loop for training the Rogers/McClelland '08 analogy model
    """

    print("Network training conditions: ")
    print(args)
    print("\n")

    # Define a model for training
    model = RM_Net(args.D_item_in, args.D_context_in, args.D_out, args.D_h_item, args.D_h_context, args.D_h_combined).to(device)
    model.apply(init_weights)

    #criterion = nn.MSELoss()   # mean squared error loss
    criterion = nn.BCELoss() # binary cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Define our dataloaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Train/test loop
    n_epochs = args.epochs
    printOutput = False

    # Log the model on TensorBoard and label it with the date/time and some other naming string
    now = datetime.now()
    date = now.strftime("_%d-%m-%y_%H-%M-%S")
    comment = "_lr-{}_epoch-{}_hitem{}_hcxt{}_hcomb{}".format(args.lr, args.epochs, args.D_h_item, args.D_h_context, args.D_h_combined)
    writer = SummaryWriter(log_dir=const.TB_LOG_DIRECTORY + 'record_' + date + comment)
    print("Open tensorboard in another shell to monitor network training (hannahsheahan$  tensorboard --logdir=training_records/tensorboard)")

    train_loss_record, test_loss_record, train_accuracy_record, test_accuracy_record = [[] for i in range(4)]

    print("Training network...")
    for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

        # train network
        train_loss, train_accuracy = train(args, model, device, trainloader, optimizer, criterion, epoch, printOutput)

        # assess network
        test_loss, test_accuracy = test(args, model, device, testloader, criterion, printOutput)

        # log performance
        log_performance(writer, epoch, train_loss, test_loss, train_accuracy, test_accuracy)
        if epoch % args.log_interval == 0:
            print('loss: {:.10f}, accuracy: {:.1f}%'.format(train_loss, train_accuracy))
            print_progress(epoch, n_epochs)

        train_accuracy_record.append(train_accuracy)
        test_accuracy_record.append(test_accuracy)
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

    record = {"train_loss":train_loss_record, "test_loss":test_loss_record, "train_accuracy":train_accuracy_record, "test_accuracy":test_accuracy_record, "args":vars(args) }
    randnum = str(random.randint(0,10000))
    args.id = randnum
    model_name, _ = get_model_name(args)
    dat = json.dumps(record)
    f = open(const.TRAININGRECORDS_DIRECTORY + randnum + date + comment + ".json","w")
    f.write(dat)
    f.close()

    writer.close()
    print("Training complete.")

    print('\nSaving trained model...')
    print(model_name)
    torch.save(model, model_name)

    return model, randnum
