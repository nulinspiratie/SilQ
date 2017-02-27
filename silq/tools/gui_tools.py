import sys
from datetime import datetime, timedelta
from queue import Empty
from traceback import format_exc
import logging
from winsound import Beep

from PyQt4 import QtGui
from PyQt4.QtCore import QThread


from qcodes.process.server import BaseServer



class GUIServer(BaseServer):
    def __init__(self, query_queue, response_queue, extras=None):
        super().__init__(query_queue, response_queue, extras)

        app = QtGui.QApplication(sys.argv)
        # ex = GUIServer(query_queue=query_queue,
        #                response_queue=response_queue)
        ex = GUIClass(query_queue=query_queue, response_queue=response_queue)
        sys.exit(app.exec_())


class GUIClass(QtGui.QMainWindow):
    def __init__(self, query_queue, response_queue):
        super().__init__()

        self.initUI()
        event_server = EventLoopServer(query_queue=query_queue,
                                       response_queue=response_queue,
                                       gui=self)
        event_server.start()
        Beep(2000,2000)
        print('started')


    def initUI(self):
        self.text_box = QtGui.QTextEdit()
        self.setCentralWidget(self.text_box)

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

    def handle_append_text(self, text):
        self.text_box.append(text)

    def handle_print(self, text):
        print(text)


class EventLoopServer(QThread, BaseServer):

    def __init__(self, query_queue, response_queue, gui, extras=None):
        QThread.__init__(self)
        BaseServer.__init__(self, query_queue, response_queue, extras)

        self.query_queue = query_queue
        self.response_queue = response_queue

        self.gui = gui

    def __del__(self):
        self.wait()

    def __getattr__(self, attr):
        return getattr(self.gui, attr)

    def run(self):
        self.run_event_loop()
