"""
Rogers/McClelland 2008 model of analogy
Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 05/12/2021
Notes: N/A
Issues: N/A
"""

from config import get_config
import rogers_mcclelland as rm
import network as net

def main():
    args, device = get_config()

    # set up the task
    context_setup = 'rm08'  # 'rm08' 'assym_mix'
    lookup, inputs, words = rm.setup_inputs(args, context_setup)
    attributes = rm.setup_outputs(args, lookup, context_setup)

    # define train and test sets
    dataset = {'index':list(range(args.n_unique)),'input_item':inputs[0],'input_context':inputs[1],'label':attributes, 'words':words, 'domains':inputs[2]}
    trainset = net.CreateDataset(dataset)
    testset = net.CreateDataset(dataset)  # Note that for now train and test are same dataset. As in Rogers/McClelland

    # train and test network
    #model, id = net.train_network(args, device, trainset, testset)
    args.id = 1139

    # analyse trained network hidden activations and training trajectory
    rm.analyse_network(args, trainset, testset, lookup, context_setup)


if __name__ == '__main__':
    main()
