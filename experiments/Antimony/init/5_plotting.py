from qcodes.plots.qcmatplotlib import MatPlot
from matplotlib import rcParams
rcParams['figure.max_open_warning'] = 80
plt.ion()

from silq.tools.notebook_tools import *
from qcodes.plots.qcmatplotlib import MatPlot
from matplotlib import rcParams

rcParams['figure.max_open_warning'] = 80

set_gates_txt = """\
{x_label}({x_val:.5f})
{y_label}({y_val:.5g})"""
measure_single_txt = """\
{param}_parameter.single_settings(samples={samples_measure}, silent=False)
qc.Measure({param}_parameter).run(name="{param}_measure", quiet=True)"""

from silq.tools.notebook_tools import *
from functools import partial
class PlotAction:
    key = None
    action_keys = []

    def __init__(self, plot, key=None):
        if key is not None:
            self.key = key
        self.plot = plot

    def txt_to_clipboard(self, txt):
        pyperclip.copy(txt)

    def key_press(self, event):
        self.plot.copy = (event.key[:4] == 'ctrl')
        self.plot.execute = (event.key[-1].isupper())

    def button_press(self, event):
        if self.plot.copy:
            self.txt_to_clipboard(self.txt)
        else:
            create_cell(self.txt, execute=self.plot.execute, location='below')


class SetGates(PlotAction):
    key = 'alt+g'

    def __init__(self, plot, key=None):
        super().__init__(plot=plot, key=key)

    def key_press(self, event):
        super().key_press(event)

    def button_press(self, event):
        self.txt = set_gates_txt.format(x_label=self.plot.x_label,
                                        y_label=self.plot.y_label,
                                        x_val=event.xdata, y_val=event.ydata)
        super().button_press(event)


class MeasureSingle(PlotAction):
    key = 'alt+s'

    def __init__(self, plot, key=None):
        super().__init__(plot=plot, key=key)

    def key_press(self, event):
        super().key_press(event)

    def button_press(self, event):
        txt = set_gates_txt + '\n\n' + measure_single_txt
        self.txt = txt.format(x_label=self.plot.x_label,
                              y_label=self.plot.y_label,
                              x_val=event.xdata, y_val=event.ydata,
                              param=self.plot.measure_parameter,
                              samples_measure=self.plot.samples_measure)
        super().button_press(event)


class MoveGates(PlotAction):
    key = 'alt+m'
    action_keys = ['alt+' + key for key in
                   ['up', 'down', 'left', 'right', '-', '+']]
    def __init__(self, plot, key=None):
        self.delta = 0.001
        super().__init__(plot=plot, key=key)

    def key_press(self, event):
        if event.key == self.key:
            self.point = self.plot[0].plot(self.plot.x_gate(),
                                           self.plot.y_gate(), 'o')[0]
        elif event.key in ['alt+up', 'alt+down']:
            val = self.plot.y_gate()
            delta = self.delta * (1 if event.key == 'alt+up' else -1)
            self.plot.y_gate(val + delta)
            self.point.set_ydata(val + delta)
        elif event.key in ['alt+left', 'alt+right']:
            val = self.plot.x_gate()
            delta = self.delta * (1 if event.key == 'alt+right' else -1)
            self.plot.x_gate(val + delta)
            self.point.set_xdata(val + delta)
        elif event.key == 'alt++':
            self.delta /= 1.5
        elif event.key == 'alt+-':
            self.delta *= 1.5

    def button_press(self, event):
        pass


class InteractivePlot(MatPlot):
    gates = {}
    def __init__(self, dataset, subplots, figsize=None, interval=1, nticks=6,
                 timeout=60, **kwargs):

        super().__init__(subplots=subplots, figsize=figsize, interval=interval,
                         **kwargs)
        self.dataset = dataset
        self.timeout = timeout
        self.cid = {}

        self.key = 'contrast'
        self.plot_data(nticks=nticks)

        self.nticks = nticks
        self.t_previous = None
        self.last_key = None
        self.last_action = None
        self.copy = False
        self.execute = False

        self.connect_event('key_press_event', self.handle_key_press)
        self.connect_event('button_press_event', self.handle_button_press)

    @property
    def t_elapsed(self):
        if self.t_previous is None:
            return timeout + 1
        else:
            return time() - self.t_previous

    @property
    def action_keys(self):
        return {action.key: action for action in self.actions}

    @property
    def x_label(self):
        return getattr(self.dataset, self.key).set_arrays[1].name

    @property
    def y_label(self):
        return getattr(self.dataset, self.key).set_arrays[0].name

    @property
    def x_gate(self):
        return self.gates[self.x_label]

    @property
    def y_gate(self):
        return self.gates[self.y_label]

    def get_action(self, key=None):
        if key is None:
            action = self.last_action
            if self.last_action is not None and self.t_elapsed < self.timeout:
                return self.last_action
            else:
                return None

        # Ignore shift
        key = key.lower()

        if key[:4] == 'ctrl':
            # Ignore ctrl
            key = key[5:]

        if key in self.action_keys:
            return self.action_keys[key]
        else:
            return None

    def connect_event(self, event, action):
        if event in self.cid:
            self.fig.canvas.mpl_disconnect(self.cid[event])

        cid = self.fig.canvas.mpl_connect(event, action)
        self.cid[event] = cid

    def handle_key_press(self, event):
        if self.get_action(event.key) is not None:
            self.t_previous = time()
            self.last_key = event.key
            action = self.get_action(event.key)
            action.key_press(event)
            self.last_action = action
        elif self.last_action is not None \
                and event.key in self.last_action.action_keys:
            self.last_action.key_press(event)
        else:
            pass

    def handle_button_press(self, event):
        action = self.get_action()
        if action is not None:
            action.button_press(event)

class CalibrationPlot(InteractivePlot):
    measure_parameter = 'adiabatic_ESR'
    samples_measure =200
    samples_scan = 100

    def __init__(self, dataset, **kwargs):
        subplots = 3 if 'voltage_difference' in dataset.arrays else 2
        super().__init__(dataset=dataset, subplots=subplots, **kwargs)

        self.key = 'contrast'
        self.plot_data(nticks=self.nticks)

        self.actions = [SetGates(self), MeasureSingle(self), MoveGates(self)]

    def plot_data(self, nticks=6):
        self.add(self.dataset.contrast, subplot=0, nticks=nticks)
        self.add(self.dataset.dark_counts, subplot=1, nticks=nticks)
        if 'voltage_difference' in self.dataset.arrays:
            self.add(self.dataset.voltage_difference, subplot=2, nticks=nticks)

class DCPlot(InteractivePlot):
    def __init__(self, dataset,  **kwargs):
        super().__init__(dataset=dataset, subplots=1, **kwargs)

        self.key = 'DC_voltage'
        self.plot_data(nticks=self.nticks)

        self.actions = [SetGates(self), MoveGates(self)]

    def plot_data(self, nticks=6):
        self.add(self.dataset.DC_voltage)

if 'TGAC' in locals():
    InteractivePlot.gates = {'TGAC': TGAC, 'DF_DS': DF_DS, 'TG': TG, 'DF': DF, 'DS': DS}