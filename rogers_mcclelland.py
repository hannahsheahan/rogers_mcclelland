"""
Rogers/McClelland 2008 model of analogy
Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 19/11/2020
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import numpy as np
import sys
from config import get_config
import network as net
import matplotlib.pyplot as plt
import scipy.io as spio
import constants as const
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import matplotlib.patches as patches
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox


def int_to_onehot(integer, maxSize):
    """This function will take as input an interger and output a one hot representation of that integer up to a max of maxSize."""
    oneHot = np.zeros((maxSize,))
    oneHot[integer-1] = 1
    return oneHot


def onehot_to_int(onehot):
    """This function will take as input a one hot representation and determine the integer interpretation"""
    integer = np.nonzero(onehot)[0]
    return integer+1  # because we are starting counting from 1 not 0


def make_hierarchy(n_items,n_attributes):
    """Construct covariance between items and attributes within a given domain.
    i.e. a context correlation structure.
    - Note HRS have totally hacked/hardcoded the hierarchy.
    """
    cov = np.zeros((n_attributes, n_items))
    i = 1
    j = n_items // 2

    start_ind = [0,0,4,0,2,4,6,0,1,2,3,4,5,6,7] # HRS hack
    stop_ind = [7,3,7,1,3,5,7,0,1,2,3,4,5,6,7] # HRS hack
    count = 0
    for ind in range(n_attributes):
        i = start_ind[ind]
        j = stop_ind[ind]+1
        cov[count,i:j] = 1
        count +=1

    return cov


def make_unbalanced_hierarchy(n_items,n_attributes):
    """Construct covariance between items and attributes within a given domain.
    i.e. a context correlation structure.
    - Note HRS have totally hacked/hardcoded the hierarchy.
    - make the hierarchy asymmetric in orde to replicate Rogers/McClelland '08
    """
    cov = np.zeros((n_attributes, n_items))
    i = 1
    j = n_items // 2

    start_ind = [0,0,0,4,4,0,1,2,3,4,5,6,7,6,7] # HRS hack
    stop_ind = [7,3,3,7,5,0,1,2,3,4,5,6,7,6,7] # HRS hack
    count = 0
    for ind in range(len(start_ind)):
        i = start_ind[ind]
        j = stop_ind[ind]+1
        cov[count,i:j] = 1
        count +=1

    return cov


def make_context2_struct(n_items,n_attributes):
    """Construct covariance between items and attributes within a given domain.
    i.e. a context correlation structure.
    - make the hierarchy asymmetric in orde to replicate Rogers/McClelland '08
    """
    cov = np.zeros((n_attributes, n_items))
    i = 1
    j = n_items // 2

    start_ind = [0,0,4,4,0,1,2,3,6,7,6,7] # HRS hack
    stop_ind = [7,3,5,5,0,1,2,3,6,7,6,7] # HRS hack
    count = 0
    for ind in range(len(start_ind)):
        i = start_ind[ind]
        j = stop_ind[ind]+1
        cov[count,i:j] = 1
        count +=1

    return cov


def make_magnitude(n_items, n_attributes):
    """Produce a magnitude covariance matrix that shows how attributes can
     encode the similarity relations between items."""

    cov = np.zeros((n_attributes, n_items))
    j = n_items // 2
    stop_ind = list(range(n_items,-1,-1))

    count = 0
    for ind in range(1,n_items):
        j = stop_ind[ind]
        cov[count,-j:] = 1
        cov[n_attributes -count -2,:j] = 1
        count += 1

    return cov


def setup_inputs(args):
    """ Setup inputs and attributes.
    - On each trial we have both one-hot context input and one-hot item input
      each item 1:8 will be assessed in each of 4 different contexts."""

    max_itemsize = args.n_items*args.n_domains
    max_contextsize = args.n_contexts*args.n_domains

    # id/words for each input
    all_domain_words = ['company','mammals','monarchy', 'fourthone']
    all_context_words = ['human size','popularity','number of teeth', 'dietary preferences', 'friendliness', 'dog common ancestry','conA','conB']
    all_item_words = [ 'katie','michael','teresa','taika','Mr Perez','Ms Que','Mrs Wellie','Mr X', \
                        'lion','elephant','monkey','mouse','sheep','ferret','gazelle','pig', \
                        'chow chow','alaskan malamute','siberian husky','saluki','basenji','grey wolf','shar pei','akita inu',\
                        '1','2','3','4','5','6','7','8']

    # cycle through all possible inputs
    # preallocate space
    item_input = np.zeros((args.n_unique,max_itemsize))
    context_input = np.zeros((args.n_unique,max_contextsize))
    domains = np.zeros((args.n_unique,args.n_domains))

    context_words, item_words, domain_words = [[] for i in range(3)]
    item_int = np.zeros((args.n_unique,1))
    context_int = np.zeros((args.n_unique,1))
    domain_int = np.zeros((args.n_unique,1))
    lookup = np.zeros((args.n_domains, args.n_contexts, args.n_items))
    count = 0

    for domain_idx in range(args.n_domains):
        for item_idx in range(args.n_items):
            for context_idx in range(args.n_contexts):

                # set up one-hot input coding
                item_input[count,:] = int_to_onehot((domain_idx)*args.n_items + item_idx +1, max_itemsize)
                context_input[count,:] = int_to_onehot((domain_idx)*args.n_contexts +context_idx +1, max_contextsize)
                domains[count,:] = int_to_onehot(domain_idx + 1,args.n_domains)

                # set up words for each input
                context_words.append(all_context_words[onehot_to_int(context_input[count,:])[0]-1])
                domain_words.append(all_domain_words[onehot_to_int(domains[count,:])[0]-1])
                item_words.append(all_item_words[onehot_to_int(item_input[count,:])[0]-1])

                # set up integers for alternative lookup
                item_int[count] = item_idx
                context_int[count] = context_idx
                domain_int[count] = domain_idx
                lookup[domain_idx, context_idx, item_idx] = count

                count += 1

    # Check that unique inputs appropriately made
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(item_input)
    plt.title('Items')
    plt.ylabel('input id #')

    plt.subplot(1,3,2)
    plt.imshow(context_input)
    plt.title('Contexts')
    plt.ylabel('input id #')

    plt.subplot(1,3,3)
    plt.imshow(domains)
    plt.title('Domains')
    plt.ylabel('input id #')
    plt.savefig(const.FIGURE_DIRECTORY + 'Inputs_coding.pdf',bbox_inches='tight')

    inputs = [item_input, context_input, domains]
    words = [all_item_words, all_context_words, all_domain_words]

    return lookup, inputs, words


def setup_outputs(args, lookup):
    """Each item has a set of attributes pertaining to it in each context,
    and as with items and contexts, each domain has its own completely distinct set.
    - There are [n_attributes x n_domains x n_contexts]  outputs in total, so 15 for each
     context in each domain, and each item activates some subset of these."""

    context_cov_A = make_unbalanced_hierarchy(args.n_items, args.n_attributes);  # HRS note this is hacked together for now
    #context_cov_B = make_magnitude(args.n_items, args.n_attributes);
    context_cov_B = make_context2_struct(args.n_items, args.n_attributes);
    context_covs = [context_cov_A, context_cov_B]

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(context_cov_A)
    plt.title('hierarchy context')
    plt.xlabel('item #')
    plt.ylabel('attribute #')
    plt.subplot(1,2,2)
    plt.imshow(context_cov_B)
    plt.title('other context')
    plt.xlabel('item #')
    plt.ylabel('attribute #')
    plt.savefig(const.FIGURE_DIRECTORY + 'Item_attribute_covariance.pdf',bbox_inches='tight')

    # cycle through all possible outputs
    attributes = np.zeros((args.n_unique, args.n_domains,args.n_contexts,args.n_attributes))

    count = 0
    for domain_idx in range(args.n_domains):
        for item_idx in range(args.n_items):
            for context_idx in range(args.n_contexts):
                attribute_ind = domain_idx * args.n_contexts + context_idx + 1;
                context_cov = context_covs[context_idx]
                attributes[count, domain_idx, context_idx, :] = context_cov[:,item_idx] # intentional spatial separation: each element in dim2+ is different node
                count += 1

    total_attribute_activity = plot_output_rdm(args, attributes, lookup)
    plot_hierarchical_cluster(total_attribute_activity)
    return attributes


def plot_output_rdm(args, attributes, lookup):
    """Plot the structure in the outputs/attributes for each context
     use Euclidean distance matrix, imshow of [items for all domains]x[items for all domains]
     in output space. This is like Fig R3A in Rogers/McClelland 2008."""

    total_attribute_distance = np.zeros((args.n_contexts, args.n_domains*args.n_items, args.n_domains*args.n_items))
    total_attribute_activity = np.zeros((args.n_contexts, args.n_domains*args.n_items, args.n_outputs))

    plt.figure()
    for context_idx in range(args.n_contexts):
        attribute_activity = np.zeros((args.n_domains*args.n_items, args.n_outputs))
        for domain_idx in range(args.n_domains):
           for item_idx in range(args.n_items):
               count = int(lookup[domain_idx, context_idx, item_idx])
               item_attributes = attributes[count, :, :,:].flatten()  # flatten across all output units
               attr_idx = domain_idx * args.n_items + item_idx
               attribute_activity[attr_idx, :] = item_attributes

        # compute distance matrix over outputs
        attribute_distance = pairwise_distances(attribute_activity, metric='euclidean')
        np.fill_diagonal(np.asarray(attribute_distance), 0)
        total_attribute_activity[context_idx,:,:] = attribute_activity
        total_attribute_distance[context_idx,:,:] = attribute_distance

        plt.subplot(2,2,context_idx * args.n_contexts + 1)
        plt.imshow(attribute_activity)
        plt.colorbar
        plt.title('Output activity vectors')
        plt.xlabel('output units')
        plt.ylabel('items x domains')

        plt.subplot(2,2,context_idx * args.n_contexts + 2)
        plt.imshow(attribute_distance)
        plt.colorbar
        plt.title('Output RDM')
        plt.xlabel('items x domains')
        plt.ylabel('items x domains')

    plt.savefig(const.FIGURE_DIRECTORY + 'Outputs_RDMs_by_context.pdf',bbox_inches='tight')

    # plot output RDM averaged/collapsed over the two contexts
    mean_attr_dist = np.mean(total_attribute_distance,0)

    plt.figure()
    plt.imshow(mean_attr_dist)
    plt.colorbar
    plt.title('Output RDM, collapsed across contexts')
    plt.xlabel('items x domains')
    plt.ylabel('items x domains')
    plt.savefig(const.FIGURE_DIRECTORY + 'Outputs_RDM_contextcollapsed.pdf',bbox_inches='tight')
    return total_attribute_activity


def plot_hierarchical_cluster(total_attribute_activity, figure_modifier='Outputs'):
    """ Hierarchical cluster analysis of outputs (R3.B)"""

    total_attribute_activity = np.mean(total_attribute_activity,0)
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    dendrogram = sch.dendrogram(sch.linkage(total_attribute_activity, method='single'))
    yscale = np.max(np.asarray(dendrogram['dcoord']))
    plt.title('Hierarchical cluster analysis of output activity')
    plt.xlabel('\nitems x domains')
    plt.ylabel('Euclidean distance')
    xticks_to_icons(ax, yscale)
    plt.savefig(const.FIGURE_DIRECTORY + figure_modifier + '_hierarchical_cluster_analysis.pdf',bbox_inches='tight')


def xticks_to_icons(ax, yscale):
    """A mapping function that will replace particular xtick values with images of particular symbols.
    - uses same symbols as in 2008 Rogers/McClelland."""
    # set ticks where your images will be
    xloc, labels = plt.xticks()
    labels = [int(label.get_text()) for label in labels]
    TICKYPOS = -yscale/10

    # remove numeric tick labels
    #ax.get_xaxis().set_ticklabels([])

    # mapping from item number to icon label
    icons_map = {label:'blank' for label in labels}
    domain_colours = ['black', 'darkgrey', 'lightgrey', 'white']
    for domain in range(len(domain_colours)):
        for i in range(domain*8, domain*8+4):
            icons_map[i] = domain_colours[domain] + '-circle'
        for i in range(domain*8+4,domain*8+6):
            icons_map[i] = domain_colours[domain] + '-square'
        for i in range(domain*8+6,domain*8+8):
            icons_map[i] = domain_colours[domain] + '-star'

    # insert icons on xaxis
    for i,x in enumerate(xloc):
        x = x-26 -0.28*x  # hack to fit x-axis
        lowerCorner = ax.transData.transform((x-2.5,TICKYPOS+yscale/40))
        upperCorner = ax.transData.transform((x+2.5,TICKYPOS+yscale/20))

        bbox_image = BboxImage(Bbox([lowerCorner, upperCorner]), norm=None, origin=None, clip_on=False)
        icon = icons_map[labels[i]] + '.png'

        bbox_image.set_data(plt.imread(os.path.join(const.ICONS_DIRECTORY,icon)))
        ax.add_artist(bbox_image)


def plot_learning_curve(record_name):
    """Get the record of training and plot loss and accuracy over time."""

    with open(os.path.join(const.TRAININGRECORDS_DIRECTORY, record_name)) as record:
        data = json.load(record)

    plt.figure()
    plt.plot(data['train_loss'])
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.savefig(const.FIGURE_DIRECTORY + 'traintraj_' + record_name[:-5] + '.pdf', bbox_inches='tight')


def analyse_network(args, trainset, testset, lookup):
    """Analyse the hidden unit activations for each unique input in each context.
    """
    model_name, analysis_name, record_name = net.get_model_name(args)

    # load an existing dataset
    try:
        data = np.load(analysis_name, allow_pickle=True)
        MDS_dict = data.item()
        preanalysed = True
        print('\nLoading existing network analysis...')
    except:
        preanalysed = False
        print('\nAnalysing trained network...')

    if not preanalysed:
        # plot training record and save it
        plot_learning_curve(record_name)

        # load the trained model and the datasets it was trained/tested on
        trained_model = torch.load(model_name)

        # Assess the network activations on test set
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)
        all_activations = net.get_activations(args, trained_model, test_loader)

        item_activations, context_activations, combined_activations = all_activations
        layers = ['items','contexts','combined']
        context_texts = ['hierarchy','hierarchy']
        hidden_sizes = [args.D_h_item, args.D_h_context, args.D_h_combined]

        # combined layer for now
        for layer_idx in range(3):
            layer = layers[layer_idx]
            activations = all_activations[layer_idx]
            hdim = hidden_sizes[layer_idx]

            # format into RDM
            total_hidden_distance = np.zeros((args.n_contexts, args.n_domains*args.n_items, args.n_domains*args.n_items))
            total_hidden_activity = np.zeros((args.n_contexts, args.n_domains*args.n_items, hdim))

            plt.figure(figsize=(10,10.5))
            for context_idx in range(args.n_contexts):
                context_text = context_texts[context_idx]
                hidden_activity = np.zeros((args.n_domains*args.n_items, hdim))
                for domain_idx in range(args.n_domains):
                   for item_idx in range(args.n_items):
                       count = int(lookup[domain_idx, context_idx, item_idx])
                       attr_idx = domain_idx * args.n_items + item_idx
                       hidden_activity[attr_idx, :] = activations[count, :].flatten()

                # compute distance matrix over hidden activations
                hidden_distance = pairwise_distances(hidden_activity, metric='euclidean')
                np.fill_diagonal(np.asarray(hidden_distance), 0)
                total_hidden_activity[context_idx,:,:] = hidden_activity
                total_hidden_distance[context_idx,:,:] = hidden_distance

                plt.subplot(2,2,context_idx * args.n_contexts + 1)
                plt.imshow(hidden_activity)
                plt.colorbar
                plt.title(layer + 'hidden layer')
                plt.xlabel('hidden units')
                plt.ylabel('items x domains')

                plt.subplot(2,2,context_idx * args.n_contexts + 2)
                plt.imshow(hidden_distance)
                plt.colorbar
                plt.title(layer + 'hidden RDM: ' + context_text)
                plt.xlabel('items x domains')
                plt.ylabel('items x domains')

            plt.savefig(const.FIGURE_DIRECTORY + layer + 'hidden_activity_RDMs_by_context_'+model_name[7:-4]+'.pdf',bbox_inches='tight')

            plt.figure()
            mean_total_hidden_distance = np.mean(total_hidden_distance,0)
            plt.imshow(mean_total_hidden_distance)
            plt.colorbar
            plt.title(layer + 'hidden RDM: both contexts')
            plt.xlabel('items x domains')
            plt.ylabel('items x domains')
            plt.savefig(const.FIGURE_DIRECTORY + layer + 'hidden_activity_RDMs_meancontext_'+model_name[7:-4]+'.pdf',bbox_inches='tight')

            plot_hierarchical_cluster(total_hidden_distance, 'hiddencombined')



def main():
    args, device = get_config()

    # load in the inputs and outputs I built in matlab (before realising I really want to train this in pytorch)
    lookup, inputs, words = setup_inputs(args)
    attributes = setup_outputs(args, lookup)

    # define train and test sets using our Dataset-inherited class
    dataset = {'index':list(range(args.n_unique)),'input_item':inputs[0],'input_context':inputs[1],'label':attributes, 'words':words, 'domains':inputs[2]}
    trainset = net.CreateDataset(dataset)
    testset = net.CreateDataset(dataset)  # HRS note that, for now, train and test are same dataset. As in Rogers/McClelland

    # train and test network
    #model, id = net.train_network(args, device, trainset, testset)
    args.id = 3544
    # analyse trained network hidden activations and training trajectory
    analyse_network(args, trainset, testset, lookup)


main()
