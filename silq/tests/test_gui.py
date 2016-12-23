import time
from threading import Thread
from winsound import Beep
import sys
from PyQt4 import QtGui

from qcodes.process.server import BaseServer


class GUIServer(BaseServer):
    def __init__(self, query_queue, response_queue, extras=None):
        super().__init__(query_queue, response_queue, extras)

        app = QtGui.QApplication(sys.argv)
        self.gui = ExampleGui(query_queue, response_queue)
        sys.exit(app.exec_())

    def handle_add_text(self, text):
        self.example_gui.add_text(text)

class ExampleGui(QtGui.QMainWindow):
    def __init__(self, query_queue, response_queue):
        super().__init__()

        self.initUI()
        self.textEdit.append('hi')

    def initUI(self):
        textEdit = QtGui.QTextEdit()
        self.setCentralWidget(textEdit)
        self.textEdit = textEdit

        exitAction = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAction)

        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('Main window')
        self.show()

    def add_text(self, text):
        self.textEdit.append(text)

def create_gui(*args, **kwargs):
    app = QtGui.QApplication(sys.argv)
    example_gui = ExampleGui(*args, **kwargs)

    sys.exit(app.exec_())