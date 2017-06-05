import traceback
from functools import partial
import json
import pyperclip
import logging

from PyQt5.QtGui import QPalette, QFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QEvent

from silq.tools.general_tools import get_exponent, get_first_digit
from . import start_gui

logger = logging.getLogger(__name__)

states = ['up_down', 'left_right', 'none']


def sim_gui(mini=False):
    # Must set sim_gui.voltage_parameters
    start_gui(SIMControlDialog, 'sim_gui',
              parameters=sim_gui.voltage_parameters,
              mini=mini)


class SIM928Dialog(QFrame):
    state_change = pyqtSignal()

    def __init__(self, parameter, idx=1, mini=False):
        super().__init__()

        self.parameter = parameter
        self.idx = idx
        self.mini = mini

        self._state = 'none'
        self.min_step = 0.0001
        self.max_step = 0.05
        self.modified_val = False

        self.setMaximumWidth(170)

        self.initUI()

        self.name_label.mousePressEvent = self.cycle_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state not in ['none', 'up_down', 'left_right']:
            logger.debug(f'State {state} not recognized')

        else:
            # Create palette for name color customization
            name_palette = QPalette()

        if state == 'none':
            name_palette.setColor(QPalette.Foreground, Qt.black)
        elif state == 'up_down':
            name_palette.setColor(QPalette.Foreground, Qt.blue)
        elif state == 'left_right':
            name_palette.setColor(QPalette.Foreground, Qt.darkGreen)
        self._state = state
        self.name_label.setPalette(name_palette)

        # Emit signal that state has changed
        self.state_change.emit()

    def cycle_state(self, *args, **kwargs):
        logger.debug(f'Cycling gate {self.parameter.name}')
        if self.state == 'none':
            self.state = 'up_down'
        elif self.state == 'up_down':
            self.state = 'left_right'
        else:
            self.state = 'none'

    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        self.setLayout(layout)

        # Set parameter name
        self.name_label = QLabel(f'{self.idx}: {self.parameter.name}')
        self.name_label.setAlignment(Qt.AlignCenter)

        if not self.mini:
            self.name_label.setFont(QFont("Times", 16, QFont.Bold))
            layout.addWidget(self.name_label)

        # Add editable voltage
        val_hbox = QHBoxLayout()
        layout.addLayout(val_hbox)

        if not self.mini:
            val_descr_label = QLabel('Val:')
            val_descr_label.setFont(QFont("Times", 12, QFont.Bold))
            val_hbox.addWidget(val_descr_label)
        else:
            self.name_label.setFont(QFont("Times", 12, QFont.Bold))
            val_hbox.addWidget(self.name_label)

        self.val_textbox = QLineEdit('{:.5g}'.format(
            self.parameter.get_latest()))
        self.val_textbox.setAlignment(Qt.AlignCenter)
        self.val_textbox.setFont(QFont("Times", 12, QFont.Bold))
        self.val_textbox.returnPressed.connect(
            lambda: self.set_voltage(self.val_textbox.text()))
        val_hbox.addWidget(self.val_textbox)

        val_units_label = QLabel('V')
        val_units_label.setFont(QFont("Times", 12, QFont.Bold))
        val_hbox.addWidget(val_units_label)


        # Add current val
        self.current_val_label = QLabel(f'')
        self.current_val_label.setAlignment(Qt.AlignCenter)
        self.current_val_label.setFont(QFont("Times", 12))
        layout.addWidget(self.current_val_label)

        self.val_textbox.textChanged.connect(self._val_textbox_changed)

        # Add voltage buttons
        if not self.mini:
            self.val_grid = QGridLayout()
            self.val_grid.setHorizontalSpacing(0)
            self.val_grid.setVerticalSpacing(0)
            layout.addLayout(self.val_grid)
            self.val_buttons = {}
            for column_idx, scale in enumerate([100, 10, 1]):
                for row_idx, sign in enumerate([1, -1]):
                    val = sign * scale
                    button = QPushButton(f"{'+' if sign == 1 else '-'}{scale} mV")
                    self.val_grid.addWidget(button, row_idx, column_idx + 1)
                    width = button.fontMetrics().boundingRect(
                        f"+100 mV").width() + 7
                    button.setMaximumWidth(width)
                    button.clicked.connect(
                        partial(self.increase_voltage, val / 1000))
                    self.val_buttons[val] = button

            # Add scale
            self.scale_label = QLabel(f'Scale: {self.parameter.scale}')
            self.scale_label.setAlignment(Qt.AlignCenter)
            self.scale_label.setFont(QFont("Times", 12))
            layout.addWidget(self.scale_label)

            # Add SIM voltage
            self.raw_val_label = QLabel('SIM val: {:.3g} V'.format(
                self.parameter.get_latest() * self.parameter.scale))
            self.raw_val_label.setAlignment(Qt.AlignCenter)
            self.raw_val_label.setFont(QFont("Times", 12))
            layout.addWidget(self.raw_val_label)

    def _val_textbox_changed(self):
        try:
            textbox_val = float(self.val_textbox.text())
            self.modified_val = (textbox_val != self.parameter.get_latest())
        except:
            self.modified_val = True
        if self.modified_val:
            self.val_textbox.setStyleSheet("color: rgb(255, 0, 0);")
            if self.current_val_label.text() == '':
                self.current_val_label.setText(
                    f'Current val: {self.parameter.get_latest()}')
        else:
            self._reset_val_textbox()

    def set_voltage(self, val):
        try:
            val = round(float(val), 6)
            self.parameter(val)
        except:
            pass
        self._reset_val_textbox()

    def _reset_val_textbox(self):
        self.modified_val = False
        self.val_textbox.setText('{:.5g}'.format(self.parameter.get_latest()))
        self.current_val_label.setText('')
        self.val_textbox.setStyleSheet("color: rgb(0, 0, 0);")

        if not self.mini:
            self.raw_val_label.setText('SIM val: {:.3g} V'.format(
                self.parameter.get_latest() * self.parameter.scale))

    def increase_voltage(self, increment):
        self.set_voltage(self.parameter.get_latest() + increment)


class SIMConfigDialog(QFrame):
    def __init__(self, mini=False):
        super().__init__()

        self.mini = mini

        self.step = {'up_down': 0.001, 'left_right': 0.001}
        self.min_step = 0.0001
        self.max_step = 0.05

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add up, down, left, right color indication
        if not self.mini:
            key_grid = QGridLayout()
            key_grid.setVerticalSpacing(0)
            for k, (color, keys) in enumerate([
                ('blue', [('up', (0, 1)), ('down', (1, 1))]),
                ('darkGreen', [('left', (1, 0)), ('right', (1, 2))])]):
                for (key, pos) in keys:
                    key_label = QLabel(key)
                    key_label.setAlignment(Qt.AlignCenter)
                    key_label.setFont(QFont("Times", 12, QFont.Bold))

                    key_palette = QPalette()
                    key_palette.setColor(QPalette.Foreground, getattr(Qt, color))
                    key_label.setPalette(key_palette)

                    key_grid.addWidget(key_label, *pos)
            self.layout.addLayout(key_grid)

            self.layout.addStretch(1)

            # Add step sizes

            step_label = QLabel('Step:')
            step_label.setAlignment(Qt.AlignCenter)
            step_label.setFont(QFont("Times", 12, QFont.Bold))
            self.layout.addWidget(step_label)

        step_hbox = QHBoxLayout()
        self.layout.addLayout(step_hbox)

        self.step_textbox = {}
        for k, (state, color) in enumerate([('up_down', 'blue'),
                                            ('left_right', 'darkGreen')]):
            step_hbox.addStretch(0.8)
            self.step_textbox[state] = QLineEdit(str(self.step[state]))
            self.step_textbox[state].setMaximumWidth(55)
            step_hbox.addWidget(self.step_textbox[state])
            self.step_textbox[state].returnPressed.connect(
                lambda: self.set_step(state, self.step_textbox.text()))
            step_unit_label = QLabel('V')
            if k == 0:
                step_unit_label.setContentsMargins(0, 0, 20, 0)
            step_hbox.addWidget(step_unit_label)

            for widget in [step_unit_label, self.step_textbox[state]]:
                widget.setFont(QFont("Times", 10, QFont.Bold))
                widget.setStyleSheet(f"color: {color};")

        self.layout.addStretch(1)

        # Add ramp buttons
        ramp_hbox = QHBoxLayout()
        self.ramp_button = QPushButton('Ramp')
        self.ramp_zero_button = QPushButton('Ramp to zero')
        ramp_hbox.addWidget(self.ramp_button)
        ramp_hbox.addWidget(self.ramp_zero_button)
        self.layout.addLayout(ramp_hbox)


        self.layout.addStretch(1)

    def set_step(self, state, val):
        try:
            val = round(float(val), 6)
            if val > self.max_step:
                val = self.max_step
            elif val < self.min_step:
                val = self.min_step
            self.step[state] = val
            self.step_textbox[state].setText(str(val))
            self._clear_focus()
        except:
            traceback.print_exc()

    def increase_step(self, state):
        current_val = self.step[state]
        exponent = get_exponent(current_val)
        first_digit = get_first_digit(current_val)
        if first_digit == 1:
            self.set_step(state, 2 * 10 ** exponent)
        elif first_digit < 4:
            self.set_step(state, 5 * 10 ** exponent)
        else:
            self.set_step(state, 1 * 10 ** (exponent + 1))

    def decrease_step(self, state):
        current_val = self.step[state]
        exponent = get_exponent(current_val)
        first_digit = get_first_digit(current_val)
        if first_digit == 1:
            self.set_step(state, 5 * 10 ** (exponent - 1))
        elif first_digit < 4:
            self.set_step(state, 1 * 10 ** exponent)
        else:
            self.set_step(state, 2 * 10 ** exponent)

    def _clear_focus(self):
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()


class Separator(QFrame):
    def __init__(self, mode='vertical'):
        super().__init__()
        if mode is 'vertical':
            frame_shape = QFrame.VLine
        else:
            frame_shape = QFrame.HLine

        self.setFrameShape(frame_shape)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setLineWidth(2)


class SIMControlDialog(QDialog):
    def __init__(self, parameters, mini=False):
        if len(parameters) > 10:
            raise SyntaxError('Can use at most 10 gates')
        super().__init__(
            flags=Qt.WindowMinimizeButtonHint|Qt.WindowCloseButtonHint)
        self.parameters = parameters
        self.mini = mini

        self.index_keys = {}
        self.state_parameters = {state: [] for state in states}

        # Dict of {parameter.name: SIM928Dialog}
        self.SIM928_dialogs = {}

        self.initUI()
        self.move(1400, 350)

    def initUI(self):
        if not self.mini:
            self.layout = QHBoxLayout()
        else:
            self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        self.config_widget = SIMConfigDialog(mini=self.mini)
        self.config_widget.ramp_button.clicked.connect(
            lambda clicked: self.ramp_voltages())
        self.config_widget.ramp_button.clicked.connect(self._clear_focus)
        self.config_widget.ramp_zero_button.clicked.connect(
            lambda clicked: self.ramp_voltages(0))
        self.config_widget.ramp_zero_button.clicked.connect(self._clear_focus)

        self.config_widget.copy_button = QPushButton('Copy from clipboard')
        self.config_widget.copy_button.clicked.connect(
            self._copy_from_clipboard)
        self.config_widget.copy_button.clicked.connect(self._clear_focus)
        self.config_widget.layout.addWidget(self.config_widget.copy_button)


        self.layout.addWidget(self.config_widget)

        for k, parameter in enumerate(self.parameters):
            idx = (k+1) % 10
            self.layout.addWidget(Separator())
            SIM928_dialog = SIM928Dialog(parameter, idx=idx, mini=self.mini)
            self.layout.addWidget(SIM928_dialog)
            Qt_index_key = getattr(Qt, f'Key_{idx}')
            self.index_keys[Qt_index_key] = SIM928_dialog
            self.SIM928_dialogs[parameter.name] = SIM928_dialog

            SIM928_dialog.state_change.connect(self.update_parameters)
            self.config_widget.ramp_button.clicked.connect(
                SIM928_dialog._reset_val_textbox)
            self.config_widget.ramp_zero_button.clicked.connect(
                SIM928_dialog._reset_val_textbox)

    def keyPressEvent(self, event):
        try:
            if event.key() in self.index_keys:
                # Change state of SIM928 dialog
                self.index_keys[event.key()].cycle_state()
            elif event.key() == Qt.Key_Escape:
                # Clear focus
                self._clear_focus()
            elif event.key() == Qt.Key_Up:
                # Increase voltage for blue (up_down) dialogs
                self.increase_voltages(self.state_parameters['up_down'],
                                       self.config_widget.step['up_down'])
            elif event.key() == Qt.Key_Down:
                # Decrease voltage for blue (up_down) dialogs
                self.increase_voltages(self.state_parameters['up_down'],
                                       -self.config_widget.step['up_down'])
            elif event.key() == Qt.Key_Right:
                # Increase voltage for green (left_right) dialogs
                self.increase_voltages(self.state_parameters['left_right'],
                                       self.config_widget.step['left_right'])
            elif event.key() == Qt.Key_Left:
                # Decrease voltage for green (left_right) dialogs
                self.increase_voltages(self.state_parameters['left_right'],
                                       -self.config_widget.step['left_right'])
            elif event.key() == Qt.Key_W:
                self.config_widget.decrease_step('up_down')
            elif event.key() == Qt.Key_S:
                self.config_widget.increase_step('up_down')
            elif event.key() == Qt.Key_A:
                self.config_widget.increase_step('left_right')
            elif event.key() == Qt.Key_D:
                self.config_widget.decrease_step('left_right')
        except:
            traceback.print_exc()

    def increase_voltages(self, parameters, val):
        for parameter in parameters:
            self.SIM928_dialogs[parameter.name].increase_voltage(val)

    def update_parameters(self):
        logger.debug('updating parameters')
        # Clear parameters from self.state_parameters
        self.state_parameters = {state: [] for state in states}

        # Iterate through SIM928 dialogs and add them to state_parameters
        for dialog in self.SIM928_dialogs.values():
            self.state_parameters[dialog.state].append(dialog.parameter)

    def _copy_from_clipboard(self, *args):
        clipboard_dict = json.loads(pyperclip.paste())
        for parameter_name, val in clipboard_dict.items():
            self.SIM928_dialogs[parameter_name].val_textbox.setText(str(val))

    def _clear_focus(self):
        logger.debug('clearing focus')
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()

    def ramp_voltages(self, voltage=None, parameters=None):
        if voltage is not None:
            if parameters is None:
                parameters = self.parameters

            for parameter in parameters:
                parameter(voltage)
        else:
            for SIM928_dialog in self.SIM928_dialogs.values():
                if SIM928_dialog.modified_val:
                    SIM928_dialog.set_voltage(SIM928_dialog.val_textbox.text())

    def changeEvent(self, event):
        # Correctly determines it is unminimized, but cannot clear focus
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() == Qt.WindowNoState:
                self._clear_focus()
                for SIM928_dialog in self.SIM928_dialogs.values():
                    SIM928_dialog._reset_val_textbox()
