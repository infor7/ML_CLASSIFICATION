import xml.etree.ElementTree as ET
import importlib
import matplotlib.pyplot as plt
import os
import sys
import lib.tools as tools

import plots.Bayes.naive_bayes as nb
import numpy as np


class PlotML():
    def __init__(self):
        os.chdir(os.path.dirname(sys.argv[0]))

    def start_plotting(self):
        plots = self.get_plots_config()
        number_of_plots = len(plots)
        list_of_plots = [None] * number_of_plots
        if number_of_plots > 1:
            fig, list_of_plots = plt.subplots(nrows=number_of_plots)

        for index, plot in enumerate(plots):
            print(index)
            try:
                script = importlib.import_module(self.convert_from_path_to_module(plot['location']))
                script.execute(list_of_plots[index], **plot['parameters'])
            except Exception as e:
                print(e)
            list_of_plots[index].set_title(plot['title'])
            if "x_label" in plot['parameters'] and "y_label" in plot['parameters']:
                list_of_plots[index].set_xlabel(plot['parameters']['x_label'])
                list_of_plots[index].set_ylabel(plot['parameters']['y_label'])
        plt.show()

    def plot_total_comparison(self):
        datasets = self.get_datasets()
        fig, ax = plt.subplots()
        x = np.arange(len(datasets))  # the label locations
        num_of_methods = 4
        width = 0.7
        accuracies = np.zeros([num_of_methods, len(datasets)])
        accuracy_labels = ["Multinomial", "Multinomial-sk", "Gaussian", "Gaussian-sk"]
        datasets_names = []
        for index, item in enumerate(datasets):
            if item.get('type') == 'sklearn':
                module = importlib.import_module('sklearn.datasets')

            else:
                with open(item['path'], 'r') as f:
                    if item.get('dtype') is None:
                        item['dtype'] = int
                    if item.get('label_index') is None:
                        item['label_index'] = None
                    else:
                        item['label_index'] = int(item['label_index'])
                    if item.get('labels_numeric') is None:
                        item['labels_numeric'] = True
                    datasets_names.append(item['name'])
                    dataset, labels = tools.load_text_file(f, label_index=item['label_index'], dtype=item['dtype'],
                                                           labels_numeric=bool(item['labels_numeric']))
                    data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=3)
                    print("Multinomial:")
                    acc_my, acc_skl = nb.accuracy_of_multinomial(data_split, labels_split)
                    accuracies[0, index] = np.mean(acc_my)
                    accuracies[1, index] = np.mean(acc_skl)
                    print("Gaussian:")
                    acc_my, acc_skl = nb.accuracy_of_gaussian(data_split, labels_split)
                    accuracies[2, index] = np.mean(acc_my)
                    accuracies[3, index] = np.mean(acc_skl)
        for i in range(0, len(accuracies)):
            ax.bar(x - width / 2 + i * width / num_of_methods, accuracies[i], width / num_of_methods,
                   label=accuracy_labels[i])
        ax.set_ylabel('Accuracies')
        ax.set_title('Total comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_names)
        ax.legend()
        plt.show()

    def generate_all_accuracies(self):
        from lib.naive_bayes import NaiveBayesGaussian, NaiveBayesMultinomial
        from sklearn.naive_bayes import GaussianNB, MultinomialNB
        algorithms_list = [NaiveBayesGaussian, NaiveBayesMultinomial]
        sklearn_list = [GaussianNB, MultinomialNB]
        accuracy_labels = ["Gaussian", "Gaussian-sk", "Multinomial", "Multinomial-sk"]
        algorithms_kwargs = [{}, {}]
        sklearn_kwargs = [{}, {'alpha':0.0001, 'fit_prior':True}]
        accuracies, dataset_names = self.run_for_sklearn_datasets(algorithms_list, algorithms_kwargs, sklearn_list, sklearn_kwargs)
        self.plot_general_plot(accuracies, accuracy_labels, dataset_names)

    def plot_general_plot(self, accuracies, accuracy_labels, datasets_names):
        x = np.arange(len(accuracies))  # the label locations
        num_of_methods = len(accuracies[0])
        width = 0.7
        fig, ax = plt.subplots()
        for i in range(0, len(accuracies)):
            ax.bar(x - width / 2 + i * width / num_of_methods, accuracies[:, i], width / num_of_methods,
                   label=accuracy_labels[i])
        ax.set_ylabel('Accuracies')
        ax.set_title('Total comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_names)
        ax.legend()
        plt.show()

    def run_for_sklearn_datasets(self, algorithm_classes, algorithm_kwargs, sklearn_classes, sklearn_kwargs):
        """
        Loads predefined sklear datasets and calculate accuracies of all different algorithms, which are provided in two lists

        :param list algorithm_classes: list of unintialized algorithms classes
        :param list sklearn_classes: list of unintialized sklearn classes
        :return:
        """
        from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
        accuracies = np.zeros([4, 2 * len(algorithm_classes)])
        dataset = load_iris()
        accuracies[0] = self.accuracies_of_different_methods(dataset.data, dataset.target, algorithm_classes,
                                                             algorithm_kwargs, sklearn_classes, sklearn_kwargs)
        dataset = load_digits()
        accuracies[1] = self.accuracies_of_different_methods(dataset.data, dataset.target, algorithm_classes,
                                                             algorithm_kwargs, sklearn_classes, sklearn_kwargs)
        dataset = load_breast_cancer()
        accuracies[2] = self.accuracies_of_different_methods(dataset.data, dataset.target, algorithm_classes,
                                                             algorithm_kwargs, sklearn_classes, sklearn_kwargs)
        dataset = load_wine()
        accuracies[3] = self.accuracies_of_different_methods(dataset.data, dataset.target, algorithm_classes,
                                                             algorithm_kwargs, sklearn_classes, sklearn_kwargs)

        dataset_names = ["Iris", "Digits", "Cancer", "Wines"]
        return accuracies, dataset_names

    def accuracies_of_different_methods(self, dataset, labels, algorithm_classes, algorithm_kwargs, sklearn_classes, sklearn_kwargs):
        """
        Returns np array of accuracies of different methods for one dataset, dataset cannot be already splitted, labels have to be separated
        algorithm_classes - not_initiated class
        algorithm_kwargs - arguments for initiation
        sklearn_classes - not initiated sklearning classes

        :param dataset:
        :param labels:
        :param algorithm_classes:
        :param algorithm_kwargs:
        :param sklearn_classes:
        :param sklearn_kwargs:
        :return:
        """
        data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=3)
        accuracies_for_dataset = np.zeros(2 * len(algorithm_classes))
        counter = 0
        for index, algorithm in enumerate(algorithm_classes):
            alg = algorithm(**algorithm_kwargs[index])
            skl = sklearn_classes[index](**sklearn_kwargs[index])
            acc_my, acc_skl = tools.accuracy_of_method(data_split, labels_split, alg,
                                                       sklearn_class=skl)
            accuracies_for_dataset[counter] = np.mean(acc_my)
            accuracies_for_dataset[counter + 1] = np.mean(acc_skl)
            counter += 2
        return accuracies_for_dataset

    def convert_from_path_to_module(self, path):
        path = str(path).replace(".py", "")
        return path.replace("/", ".")

    def get_plots_config(self):
        plots_config = []
        root = self.read_xml()
        for plot in root:
            title = self.get_title_from_et(plot)
            script_location = self.get_script_location_from_et(plot)
            parameters = self.get_parameters_from_et(plot)
            plots_config.append({
                "title": title,
                "location": script_location,
                "parameters": parameters,
            })
        return plots_config

    def get_datasets(self):
        root = self.read_xml()
        datasets_config = []
        for dataset in root:
            if dataset.tag == "DATASET":
                # name = self.get_attribute_from_et(dataset, "name")
                # path = self.get_attribute_from_et(dataset, "path")
                # dtype = self.get_attribute_from_et(dataset, "dtype")
                # labels_numeric = self.get_attribute_from_et(dataset, "labels_numeric")
                # labels_index = self.get_attribute_from_et(dataset, "labels_index")
                datasets_config.append(self.get_all_attributes_from_et(dataset))
                # datasets_config.append({
                #     "name": name,
                #     "path": path,
                #     "dtype": dtype,
                #     "labels_numeric": labels_numeric,
                #     "labels_index": labels_index,
                # })
        return datasets_config

    def read_xml(self):
        tree = ET.parse('config/config.xml')
        root = tree.getroot()
        return root

    def get_title_from_et(self, et):
        for title in et.findall("TITLE"):
            return title.text.strip()
        raise Exception("Title is not defined")

    def get_script_location_from_et(self, et):
        for script_path in et.findall("SCRIPT_PATH"):
            return script_path.text.strip()
        raise Exception("Script location is not defined")

    def get_all_attributes_from_et(self, et):
        parameters = {}
        for item in et:
            parameters[item.tag] = item.text.strip()
        return parameters

    def get_attribute_from_et(self, et, attribute: str):
        for item in et.findall(attribute):
            return item.text.strip()
        raise Exception(attribute + " is not defined")

    def get_parameters_from_et(self, et):
        parameters = {}
        for main_branch in et.findall("PARAMETERS"):
            for additional_parameter in main_branch:
                parameters[additional_parameter.tag] = additional_parameter.text.strip()
        return parameters


if __name__ == "__main__":
    plot_ml = PlotML()
    plot_ml.generate_all_accuracies()
    wait = input("PRESS ENTER TO CONTINUE.")
