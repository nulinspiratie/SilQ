from qcodes.plots.qcmatplotlib import MatPlot
from matplotlib import rcParams
rcParams['figure.max_open_warning'] = 80
plt.ion()

if hasattr(qc, 'MatPlot'):
    qc.MatPlot.plot_2D_kwargs = {'cmap': 'inferno'}

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
        elif self.plot.execute:
            cell_create_below_execute(self.txt)
        else:
            cell_create_below_select(self.txt)


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


class CalibrationPlot(MatPlot):
    measure_parameter = 'adiabatic_ESR'
    samples_measure =200
    samples_scan = 100
    def __init__(self, dataset, figsize=(13, 4), interval=5, nticks=6,
                 timeout=15, **kwargs):
        super().__init__(subplots=(1, 3), figsize=figsize, interval=interval,
                         **kwargs)
        self.dataset = dataset
        self.timeout = timeout
        self.cid = {}

        self.extract_gates('contrast')
        self.plot_data(nticks=nticks)

        self.t_previous = None
        self.last_key = None
        self.copy = False
        self.execute = False

        self.connect_event('key_press_event', self.handle_key_press)
        self.connect_event('button_press_event', self.handle_button_press)

        self.actions = [SetGates(self), MeasureSingle(self)]

    @property
    def t_elapsed(self):
        if self.t_previous is None:
            return timeout + 1
        else:
            return time() - self.t_previous

    @property
    def action_keys(self):
        return {action.key: action for action in self.actions}

    def get_action(self, key=None):
        if key is None:
            key = self.last_key

        # Ignore shift
        key = key.lower()

        if key[:4] == 'ctrl':
            # Ignore ctrl
            key = key[5:]

        if key in self.action_keys:
            return self.action_keys[key]
        else:
            return None

    def extract_gates(self, key):
        self.x_label = getattr(self.dataset, key).set_arrays[1].name
        self.y_label = getattr(self.dataset, key).set_arrays[0].name
        return self.x_label, self.y_label

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
        else:
            pass

    def handle_button_press(self, event):
        action = self.get_action()
        if action is not None and self.t_elapsed < self.timeout:
            action.button_press(event)

    def plot_data(self, nticks=6):
        self.add(self.dataset.contrast, subplot=1, nticks=nticks)
        self.add(self.dataset.dark_counts, subplot=2, nticks=nticks)
        self.add(self.dataset.voltage_difference, subplot=3, nticks=nticks)

        # if hasattr(self.dataset, 'fidelity_empty'):
        #     self.plot.add(self.dataset.fidelity_load, subplot=3, nticks=nticks)
        #     self.plot.add(self.dataset.fidelity_empty, subplot=4, nticks=nticks)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
