from qcodes.plots.qcmatplotlib import MatPlot
from silq.tools.notebook_tools import *
import pyperclip
from time import time
import numpy as np

from qcodes.station import Station

set_gates_txt = """\
{x_label}({x_val:.5f})
{y_label}({y_val:.5g})"""
measure_single_txt = """\
{param}_parameter.single_settings(samples={samples_measure}, silent=False)
qc.Measure({param}_parameter).run(name="{param}_measure", quiet=True)"""


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
                   ['up', 'down', 'left', 'right', '-', '+', '=']]
    def __init__(self, plot, key=None):
        self.delta = 0.001
        super().__init__(plot=plot, key=key)

    def key_press(self, event):
        if event.key == self.key:
            self.point = self.plot[0].plot(self.plot.x_gate(),
                                           self.plot.y_gate(), 'or', )[0]
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
        elif event.key in ['alt++', 'alt+=']:
            self.delta /= 1.5
        elif event.key == 'alt+-':
            self.delta *= 1.5

    def button_press(self, event):
        self.plot.txt += '\nreceived'
        if event.guiEvent['altKey']:
            self.plot.txt += ' alt'
            self.plot.x_gate(event.xdata)
            self.point.set_xdata(event.xdata)
            self.plot.y_gate(event.ydata)
            self.point.set_ydata(event.ydata)


class InteractivePlot(MatPlot):
    def __init__(self, subplots, dataset=None, figsize=None,
                 nticks=6, timeout=60, **kwargs):
        super().__init__(subplots=subplots, figsize=figsize,
                         **kwargs)
        self.station = Station.default

        if hasattr(self.station, 'layout'):
            self.layout = self.station.layout
        self.timeout = timeout
        self.cid = {}

        self.nticks = nticks
        self.t_previous = None
        self.last_key = None
        self.last_action = None
        self.copy = False
        self.execute = False

        if dataset:
            self.load_dataset(dataset)

        self._event_key = None
        self._event_button = None

        self.connect_event('key_press_event', self.handle_key_press)
        self.connect_event('button_press_event', self.handle_button_press)

    @property
    def t_elapsed(self):
        if self.t_previous is None:
            return self.timeout + 1
        else:
            return time() - self.t_previous

    @property
    def action_keys(self):
        return {action.key: action for action in self.actions}

    def load_dataset(self, dataset):
        self.dataset = dataset
        self.x_label = getattr(self.dataset, self.key).set_arrays[1].name
        self.y_label = getattr(self.dataset, self.key).set_arrays[0].name

        if hasattr(self.station, self.x_label):
            self.x_gate = getattr(self.station, self.x_label)
        if hasattr(self.station, self.x_label):
            self.y_gate = getattr(self.station, self.y_label)

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
        self._event_key = event
        if self.get_action(event.key) is not None:
            self.t_previous = time()
            self.last_key = event.key
            action = self.get_action(event.key)
            action.key_press(event)
            self.last_action = action
        elif self.last_action is not None \
                and event.key in self.last_action.action_keys:
            self.t_previous = time()
            self.last_action.key_press(event)
        else:
            pass

    def handle_button_press(self, event):
        self._event_button = event
        self.txt = f'x:{plot._event_button.xdata}, y:{plot._event_button.ydata}, alt: {plot._event_button.guiEvent["altKey"]}'
        action = self.get_action()
        if action is not None:
            self.t_previous = time()
            self.txt += '\nSent'
            action.button_press(event)

    def plot_data(self, **kwargs):
        raise NotImplementedError(
            'plot_data should be implemented in a subclass')

class CalibrationPlot(InteractivePlot):
    measure_parameter = 'adiabatic_ESR'
    samples_measure =200
    samples_scan = 100

    def __init__(self, dataset, **kwargs):
        subplots = 3 if 'voltage_difference' in dataset.arrays else 2
        self.key = 'contrast'
        super().__init__(subplots=subplots, dataset=dataset, **kwargs)

        self.plot_data(nticks=self.nticks)

        self.actions = [SetGates(self), MeasureSingle(self), MoveGates(self)]

    def plot_data(self, nticks=6):
        self.add(self.dataset.contrast, subplot=0, nticks=nticks)
        self.add(self.dataset.dark_counts, subplot=1, nticks=nticks)
        if 'voltage_difference' in self.dataset.arrays:
            self.add(self.dataset.voltage_difference, subplot=2, nticks=nticks)


class DCPlot(InteractivePlot):
    def __init__(self, dataset,  **kwargs):
        self.key = 'DC_voltage'
        super().__init__(dataset=dataset, subplots=1, **kwargs)

        self.plot_data(nticks=self.nticks)

        self.actions = [SetGates(self), MoveGates(self)]

    def plot_data(self, nticks=6):
        self.add(self.dataset.DC_voltage)


class ScanningPlot(InteractivePlot):
    def __init__(self, parameter, interval=0.01, auto_start=False, **kwargs):
        super().__init__(**kwargs)
        self.timer = self.fig.canvas.new_timer(interval=interval * 1000)
        self.timer.add_callback(self.scan)

        self.parameter = parameter

        self.parameter.continuous = auto_start
        self.scan(initialize=True, setup=True, start=True,
                  stop=(not auto_start))

        if auto_start:
            # Already started during acquire
            self.start(setup=False, start=False)

    @property
    def interval(self):
        return self.timer.interval / 1000

    @interval.setter
    def interval(self, interval):
        if hasattr(self, 'timer'):
            self.timer.interval = interval * 1000

    def start(self, setup=True, start=True):
        if setup:
            self.parameter.setup(start=start)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.layout.stop()

    def scan(self, initialize=False, setup=False, start=False, stop=False):
        from winsound import Beep
        self.results = self.parameter.acquire(start=start, stop=stop,
                                              setup=setup)
        self.update_plot(initialize=initialize)


class DCSweepPlot(ScanningPlot):
    def __init__(self, parameter, **kwargs):
        if parameter.trace_pulse.enabled:
            subplots = (2, 1)
        else:
            subplots = 1
        super().__init__(parameter, subplots=subplots, **kwargs)

        if parameter.trace_pulse.enabled:
            self[1].set_ylim(-0.1, 1.3)

        self[0].plot([0], [0], 'or', ms=5)

    def update_plot(self, initialize=False):
        for k, result in enumerate(self.results):
            if initialize:
                setpoints = self.parameter.setpoints[k]
                setpoint_names = self.parameter.setpoint_names[k]
                name = self.parameter.names[k]
                if len(setpoints) == 2:
                    self[k].add(result, x=setpoints[0], y=setpoints[1],
                                xlabel=setpoint_names[0],
                                ylabel=setpoint_names[1],
                                zlabel=name)
                else:
                    self[k].add(result, x=setpoints[0],
                                xlabel=setpoint_names[0],
                                ylabel=name)

            else:
                result_config = self.traces[k]['config']
                if 'z' in result_config:
                    result_config['z'] = result
                else:
                    result_config['y'] = result
        super().update_plot()