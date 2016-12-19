import sys
from datetime import datetime, timedelta
from queue import Empty
from traceback import format_exc
import logging

from PyQt4 import QtGui
from PyQt4.QtCore import QThread


from qcodes.process.server import BaseServer



class PipeServer(BaseServer):
    def __init__(self, query_queue, response_queue, extras=None):
        super().__init__(query_queue, response_queue, extras)

        app = QtGui.QApplication(sys.argv)
        # ex = GUIServer(query_queue=query_queue,
        #                response_queue=response_queue)
        ex = GUIServer(query_queue=query_queue, response_queue=response_queue)
        sys.exit(app.exec_())


class GUIServer(BaseServer, QtGui.QMainWindow):
    def __init__(self, query_queue, response_queue):
        QtGui.QMainWindow.__init__(self)
        BaseServer.__init__(self, query_queue, response_queue)

        self.initUI()
        self.run_event_loop()

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


    def handle_append(self, text):
        self.text_box.append(text)


class EventLoopThread(QThread):

    def __init__(self, query_queue, response_queue):
        QThread.__init__(self)
        self.query_queue = query_queue
        self.response_queue = response_queue

    def __del__(self):
        self.wait()

    def run_event_loop(self):
        self.running = True
        next_store_ts = datetime.now()
        next_monitor_ts = datetime.now()

        while self.running:
            read_timeout = self._storage_period / self.queries_per_store
            try:
                query = self._query_queue.get(timeout=read_timeout)
                self.process_query(query)
            except Empty:
                pass

            try:
                now = datetime.now()

                if self._measuring and now > next_store_ts:
                    td = timedelta(seconds=self._storage_period)
                    next_store_ts = now + td
                    self._data.write()

                if now > next_monitor_ts:
                    td = timedelta(seconds=self._monitor_period)
                    next_monitor_ts = now + td
                    # TODO: update the monitor data storage

            except:
                logging.error(format_exc())

    def run(self):
        # your logic here