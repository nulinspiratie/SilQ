import json
from functools import partial

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
import os
from PyQt5.QtWidgets import QApplication
import silq
from . import start_gui


def frequencies_gui(thread=True):
    start_gui(FreqenciesGui, 'frequencies_gui', use_thread=thread)


class FreqenciesGui(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QFormLayout()
        self.data = {}
        self.backup_data = {}
        self.init_ui()

    def init_ui(self):
        # self.setWindowIcon(QtGui.QIcon(r''))
        self.setWindowTitle("Frequencies")
        root_dir = r'C:\Users\LabUser\Documents\GitHub\Experiments\Flip-flop\config' # Directory for the gui data
        os.chdir(root_dir)
        properties_file = root_dir + r'\properties.json'
        backup_file = r'C:\Users\LabUser\Desktop\backup.json' # Directory for the backup data
        with open(properties_file, 'r') as f:
            self.data = json.load(f)
        with open(backup_file, 'r') as g:
            self.backup_data = json.load(g)

        self.entries = self._generate_fields()
        self.setLayout(self.layout)

    def _generate_fields(self):
    # Creates the boxes for each item of the file

        lines = []
        for key, val in self.data.items():
            if isinstance(val, (int, float)):
                line = QLineEdit('{:.2f}'.format(val))
            else:
                line = QLineEdit(val)
            # line.setValidator(QDoubleValidator(val, 99.99, 2))
            line.key = key  # Manually add key label to the textbox
            line.returnPressed.connect(partial(self._update_entry, key, val, line)) # connects the enter key to saving the new values in the parameter file
                                                                                    # and the old values in a backup file. It also config.refreshes.
            self.layout.addRow(key, line)
            lines.append(line)

        return lines

    def _update_entry(self, key, val, line):
        new_value = line.text()
        try:
            new_value = float(new_value)
        except:
            pass

        if new_value != self.data[key]:
            write_file('backup.json', self.data)
            self.data[key] = new_value
            write_file('properties.json', self.data)
        silq.config.refresh()

def write_file(filename, data):
    if filename:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = frequencies_gui()
    window.show()
    sys.exit(app.exec_())