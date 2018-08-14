import numpy as np
from src.plot_tools import Plotter
import matplotlib.pyplot as plt
import os

path = '../results/'


class Saver:
    def __init__(self, exp_name, job_name):
        self.path = path + exp_name + '/' + job_name + '/'

    def store_act_hists(self, name, hists, epochs):
        for idx, epoch_hists in enumerate(hists):
            for layer_idx, layer_hist in enumerate(epoch_hists):
                file_path = self.path + name + '_epoch' + str(epochs[idx]) + '_layer' + str(layer_idx + 1)
                file_handle = open(file=file_path + '.act', mode='wb')
                np.save(file_handle, layer_hist, allow_pickle=False)

    def store_sequence(self, name, sequence, epochs):
        sequence = np.reshape(np.asarray(sequence), newshape=(-1, 1))
        epochs = np.reshape(epochs, newshape=(-1, 1))
        data_to_store = np.concatenate((epochs, sequence), axis=1)

        file_path = self.path + name
        file_handle = open(file=file_path + '.seq', mode='wb')
        np.save(file_handle, data_to_store, allow_pickle=False)

    def store_connections(self, name, w_vals):
        for layer_idx in range(len(w_vals)):
            file_path = self.path + name + str(layer_idx) + '.vals'
            file_handle = open(file=file_path, mode='wb')
            np.save(file_handle, w_vals(layer_idx), allow_pickle)


class Loader:
    def __init__(self, exp_name, job_name):
        self.path = path + exp_name + '/' + job_name + '/'

    def load_act_hists(self, name):
        hists = []
        epochs = []
        layers = []

        for file in os.listdir(self.path):
            if file.endswith('.act') and file.startswith(name):
                file_handle = open(self.path + file, mode='rb')
                hists.append(np.load(file_handle))
                epochs.append(extract_first_number(file[(file.rfind('epoch') + 5):]))
                layers.append(extract_first_number(file[(file.rfind('layer') + 5):]))

        epochs = np.asarray(epochs)
        layers = np.asarray(layers)
        hist_list = []
        epoch_list = []

        for epoch in np.unique(epochs):
            epoch_hist = []
            for layer in np.unique(layers):
                epoch_idxs = np.squeeze(np.argwhere(epochs == epoch))
                layer_idxs = np.squeeze(np.argwhere(layers == layer))
                hist_idx = np.squeeze(np.intersect1d(epoch_idxs, layer_idxs))
                epoch_hist.append(hists[hist_idx])
            hist_list.append(epoch_hist)
            epoch_list.append(epoch)

        return hist_list, epoch_list

    def load_sequence(self, name):
        file_handle = open(self.path + name + '.seq', mode='rb')
        return np.load(file_handle)

    def generate_plots(self):
        sequences = []
        seq_names = []
        for file in os.listdir(self.path):
            if file.endswith('.act'):
                file_handle = open(self.path + file, mode='rb')
                histogram = np.load(file_handle)
                epoch = extract_first_number(file[(file.rfind('epoch') + 5):])
                layer = extract_first_number(file[(file.rfind('layer') + 5):])
                plt.figure()
                plt.bar(histogram[:, 0], histogram[:, 1], width=0.9, align='center')
                plt.title('Activation Histogram: Epoch ' + str(epoch) + ', Layer ' + str(layer + 1))
                plt.xlabel('activation')
                plt.ylabel('amount')
                plt.savefig(self.path + file[:file.find('.act')] + '.png', bbox_inches='tight')
            elif file.endswith('.seq'):
                file_handle = open(self.path + file, mode='rb')
                sequences.append(np.load(file_handle))
                seq_names.append(file[:file.find('.seq')])

        plt.figure()
        legend = []
        for idx, seq in enumerate(sequences):
            plt.plot(seq[:, 0], seq[:, 1])
            legend.append(seq_names[idx])
        plt.legend(tuple(legend))
        plt.xlabel('epochs')
        plt.ylabel('misclassification rate')
        plt.title('Misclassification rate')
        plt.show()
        plt.savefig(self.path + 'misclassification', bbox_inches='tight')


def extract_first_number(text):
    number_string = ""
    for idx, char in enumerate(text):
        if char.isdigit():
            number_string += char
        else:
            break
    return int(number_string)





