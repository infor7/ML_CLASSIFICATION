import xml.etree.ElementTree as ET
import importlib
import matplotlib.pyplot as plt
import os
import sys

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

        plt.show()

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

    def get_parameters_from_et(self, et):
        parameters = {}
        for main_branch in et.findall("PARAMETERS"):
            for additional_parameter in main_branch:
                parameters[additional_parameter.tag] = additional_parameter.text.strip()
        return parameters


if __name__ == "__main__":
    plot_ml = PlotML()
    plot_ml.start_plotting()
