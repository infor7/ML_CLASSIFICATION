import xml.etree.ElementTree as ET
import importlib
import matplotlib.pyplot as plt
import os
import sys
import lib.tools as tools
import ast

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
        methods = self.get_methods()

        # accuracies = np.zeros([num_of_methods, len(datasets)])
        accuracy_labels = []
        algorithm_list = []
        algorithm_kwargs = []
        for index, item in enumerate(methods):
            mod = __import__(self.convert_from_path_to_module(item['PATH']), fromlist=[item["CLASS"]])
            klass = getattr(mod, item["CLASS"])
            algorithm_list.append(klass)
            accuracy_labels.append(item.get("NAME"))
            if item.get("KWARGS") is not None:
                algorithm_kwargs.append(self.get_all_attributes_from_et(item.get("KWARGS")))
            else:
                algorithm_kwargs.append({})
        accuracies, dataset_names = self.generate_all_accuracies(algorithm_list, accuracy_labels, algorithm_kwargs)
        self.plot_general_plot(accuracies, accuracy_labels, dataset_names)

        # else:
        #     with open(item['path'], 'r') as f:
        #         if item.get('dtype') is None:
        #             item['dtype'] = int
        #         if item.get('label_index') is None:
        #             item['label_index'] = None
        #         else:
        #             item['label_index'] = int(item['label_index'])
        #         if item.get('labels_numeric') is None:
        #             item['labels_numeric'] = True
        #         datasets_names.append(item['name'])
        #         dataset, labels = tools.load_text_file(f, label_index=item['label_index'], dtype=item['dtype'],
        #                                                labels_numeric=bool(item['labels_numeric']))
        #         data_split, labels_split = tools.cross_validation_split(dataset=dataset, labels=labels, folds=3)
        #         print("Multinomial:")
        #         acc_my, acc_skl = nb.accuracy_of_multinomial(data_split, labels_split)
        #         accuracies[0, index] = np.mean(acc_my)
        #         accuracies[1, index] = np.mean(acc_skl)
        #         print("Gaussian:")
        #         acc_my, acc_skl = nb.accuracy_of_gaussian(data_split, labels_split)
        #         accuracies[2, index] = np.mean(acc_my)
        #         accuracies[3, index] = np.mean(acc_skl)

    def generate_all_accuracies(self, algorithms_list, accuracy_labels, algorithms_kwargs):
        accuracies, dataset_names = self.run_for_all_datasets(algorithms_list, algorithms_kwargs)
        return accuracies, dataset_names

    def plot_general_plot(self, accuracies, accuracy_labels, datasets_names):
        x = np.arange(len(accuracies))  # the label locations
        num_of_methods = len(accuracies[0])
        width = 0.7
        fig, ax = plt.subplots()
        for i in range(0, len(accuracies[0])):
            ax.bar(x - width / 2 + i * width / num_of_methods, accuracies[:, i], width / num_of_methods,
                   label=accuracy_labels[i])
        ax.set_ylabel('Accuracies')
        ax.set_title('Total comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_names)
        ax.legend()
        plt.show()

    def run_for_all_datasets(self, algorithm_classes, algorithm_kwargs):
        datasets = self.get_datasets()
        accuracies = np.zeros([len(datasets), len(algorithm_classes)])
        datasets_names = []
        for index, item in enumerate(datasets):
            if item.get("label") is not None:
                datasets_names.append(item["label"])
                print("Dataset: ", item.get("label"))
            else:
                datasets_names.append(item["name"])
                print("Dataset: ", item.get("name"))
            if item.get("type") == "sklearn":
                mod = __import__(self.convert_from_path_to_module(item['path']), fromlist=[item["name"]])
                klass = getattr(mod, item["name"])
                dataset = klass()
                accuracies[index] = self.accuracies_of_different_methods(dataset.data, dataset.target,
                                                                         algorithm_classes,
                                                                         algorithm_kwargs)
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
                    dataset, labels = tools.load_text_file(f, label_index=item['label_index'], dtype=item['dtype'],
                                                           labels_numeric=bool(item['labels_numeric']))
                    accuracies[index] = self.accuracies_of_different_methods(dataset, labels, algorithm_classes,
                                                                             algorithm_kwargs)
        return accuracies, datasets_names

    def accuracies_of_different_methods(self, dataset, labels, algorithm_classes, algorithm_kwargs):
        """
        Returns np array of accuracies of different methods for one dataset, dataset cannot be already splitted, labels have to be separated
        algorithm_classes - not_initiated class
        algorithm_kwargs - arguments for initiation

        :param dataset:
        :param labels:
        :param algorithm_classes:
        :param algorithm_kwargs:
        :return:
        """
        data_split, labels_split = tools.cross_validation_split(dataset=abs(dataset), labels=labels, folds=3)
        accuracies_for_dataset = np.zeros(len(algorithm_classes))
        for index, algorithm in enumerate(algorithm_classes):
            alg = algorithm(**algorithm_kwargs[index])
            acc_my, _ = tools.accuracy_of_method(data_split, labels_split, alg)
            accuracies_for_dataset[index] = np.mean(acc_my)
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
                datasets_config.append(self.get_all_attributes_from_et(dataset))
        return datasets_config

    def get_methods(self):
        root = self.read_xml()
        datasets_config = []
        for dataset in root:
            if dataset.tag == "METHOD":
                datasets_config.append(self.get_all_attributes_from_et(dataset))
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

    def get_val(self, string):
        return ast.literal_eval(string)


if __name__ == "__main__":
    plot_ml = PlotML()
    plot_ml.plot_total_comparison()
    wait = input("PRESS ENTER TO CONTINUE.")
