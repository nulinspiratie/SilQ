import threading
import sys
from PyQt5.QtWidgets import QApplication


def start_gui(window, gui_name, *args, use_thread=True):
    if use_thread:
        if any(t.name == gui_name for t in threading.enumerate()):
            raise RuntimeError(f'GUI {gui_name} already exists. Exiting')
        t = threading.Thread(target=start_gui, name=gui_name,
                             args=(window, gui_name, *args),
                             kwargs={'use_thread': False})
        t.start()
        return t
    else:
        qApp = QApplication(sys.argv)
        aw = window(*args)
        aw.show()
        qApp.exec_()
        return qApp