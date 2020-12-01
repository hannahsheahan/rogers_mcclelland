"""
Configuration file for network architecture and training settings
Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 25/11/2020
"""
import argparse
import torch


def parse_config():
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
    parser.add_argument('--model-type', default='linearsigmoid-wgain1.0', help='string type label for model (default: "none")')

    # task config
    parser.add_argument('--n_items', default=8, type=int, help='number of input items (default: 8)')
    parser.add_argument('--n_contexts', default=2, type=int, help='number of context inputs (default: 2)')
    parser.add_argument('--n_domains', default=3,  type=int, help='number of domains (default: 3)')
    parser.add_argument('--n_attributes', default=15, type=int, help='number of attributes (default: 15)')     # True: task is like Fabrice's with filler trials; False: solely compare trials

    # network architecture
    parser.add_argument('--D_h_item', type=int, default=1500, help='hidden size for hidden item representation (default: 100)')
    parser.add_argument('--D_h_context', type=int, default=1500, help='hidden size for hidden context representation (default: 100)')
    parser.add_argument('--D_h_combined', type=int, default=2000, help='hidden size for hidden combined representation (default: 200)')

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
    config = parser.parse_args()

    # set remaining defaults
    config.n_outputs = config.n_attributes * config.n_domains * config.n_contexts
    config.n_unique = config.n_domains * config.n_contexts * config.n_items
    config.D_item_in = config.n_items * config.n_domains
    config.D_context_in = config.n_contexts * config.n_domains
    config.D_out = config.n_attributes * config.n_domains * config.n_contexts

    return config, device


def get_config():
    return config, device


# Parse arguments when this script is called
config, device = parse_config()
