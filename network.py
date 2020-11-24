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
import argparse


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
        if np.all(np.abs(output-labels) < args.correct_threshold):
            correct += 1

    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
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

            print(output)
            if np.all(np.abs(output-labels) < args.correct_threshold):
                correct += 1

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
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
        self.hitem_activations = F.relu(self.fc_item_to_hitem(x_item))
        self.hcontext_activations = F.relu(self.fc_context_to_hcontext(x_context))
        self.combined_activations = F.relu(self.fc_hitem_to_combined(self.hitem_activations) + self.fc_hcontext_to_combined(self.hcontext_activations))
        self.output = torch.sigmoid(self.fc_combined_to_out(self.combined_activations))
        return self.output

    def get_activations(self, x_item, x_context):
        self.forward(x_item, x_context)  # update the activations with the particular input
        return self.combined_activations, self.hcontext_activations, self.hitem_activations

    def init_weights(m):
        if type(m) == nn.Linear:
            gain = 0.1  # keep initial weights small and low variance
            torch.nn.init.xavier_uniform(m.weight, gain)
            m.bias.data.fill_(-0.01)  # initialize to start all units 'off'


def define_hyperparams():
    """
    This will enable us to take different network training settings/hyperparameters in when we call main.py from the command line.
    e.g. python3 main.py --batch-size=12 --epochs=20 --save-model
    Or if you want it to execute train/test across multiple combinations of hyperparameters, you can define multiple as follows:
    e.g. python3 main.py --batch-size-multi 12 24 48 --lr-multi 0.01 0.001
    If you are running this from a notebook and not the command line, just adjust the params specified in the class argparser()
    """
    """
    This will enable us to take different network training settings/hyperparameters in when we call main.py from the command line.
    e.g. python3 main.py --batch-size=12 --epochs=20 --save-model
    Or if you want it to execute train/test across multiple combinations of hyperparameters, you can define multiple as follows:
    e.g. python3 main.py --batch-size-multi 12 24 48 --lr-multi 0.01 0.001
    If you are running this from a notebook and not the command line, just adjust the params specified in the class argparser()
    """
    use_cuda = False  # use CPU on my local computer
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    parser = argparse.ArgumentParser(description='PyTorch network settings')

    # task config
    parser.add_argument('--n_items', default=8, type=int, help='number of input items (default: 8)')
    parser.add_argument('--n_contexts', default=2, type=int, help='number of context inputs (default: 2)')
    parser.add_argument('--n_domains', default=3,  type=int, help='number of domains (default: 3)')
    parser.add_argument('--n_attributes', default=15, type=int, help='number of attributes (default: 15)')     # True: task is like Fabrice's with filler trials; False: solely compare trials

    # network architecture
    parser.add_argument('--D_h_item', type=int, default=100, help='hidden size for hidden item representation (default: 100)')
    parser.add_argument('--D_h_context', type=int, default=100, help='hidden size for hidden context representation (default: 100)')
    parser.add_argument('--D_h_combined', type=int, default=200, help='hidden size for hidden combined representation (default: 200)')

    # network training hyperparameters
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=30000, metavar='N', help='number of epochs to train (default: 30,000 as used in Rogers/McClelland 08)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 0.05 as used in Rogers/McClelland 08)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many epochs to wait before printing training status')
    parser.add_argument('--weight_decay', type=int, default=0.0000, metavar='N', help='weight-decay for l2 regularisation (default: 0)')
    parser.add_argument('--save-model', action='store_true', help='For saving the current model')
    parser.add_argument('--correct_threshold', type=float, default=0.01, help='how close each output value needs to be to label to be classified as correct (default: 0.01 = within 1% as used in Rogers/McClelland 08)')
    args = parser.parse_args()

    # set remaining defaults
    args.n_outputs = args.n_attributes * args.n_domains * args.n_contexts
    args.n_unique = args.n_domains * args.n_contexts * args.n_items
    args.D_item_in = args.n_items * args.n_domains
    args.D_context_in = args.n_contexts * args.n_domains
    args.D_out = args.n_attributes * args.n_domains * args.n_contexts

    return args, device


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
    model_name = const.MODEL_DIRECTORY + 'rogers_mcclelland_model' + hiddensizes + '.pth'
    analysis_name = const.ANALYSIS_DIRECTORY + 'rogers_mcclelland_model_analysis' + hiddensizes + '.npy'
    return model_name, analysis_name


def train_network(args, device, trainset, testset):
    """
    This function performs the train/test loop for training the Rogers/McClelland '08 analogy model
    """
    model_name, _ = get_model_name(args)

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
            print('loss: {:.10f}, accuracy: {:.2f}'.format(train_loss, train_accuracy))
            print_progress(epoch, n_epochs)

        train_accuracy_record.append(train_accuracy)
        test_accuracy_record.append(test_accuracy)
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

    record = {"train_loss":train_loss_record, "test_loss":test_loss_record, "train_accuracy":train_accuracy_record, "test_accuracy":test_accuracy_record, "args":vars(args) }
    randnum = str(random.randint(0,10000))
    dat = json.dumps(record)
    f = open(const.TRAININGRECORDS_DIRECTORY + randnum + date + comment + ".json","w")
    f.write(dat)
    f.close()

    writer.close()
    print("Training complete.")

    print('\nSaving trained model...')
    print(str(randnum) + '_' + model_name)
    torch.save(model, model_name)

    return model
