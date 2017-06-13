from functools import partial
import matplotlib as mpl
from qcodes.plots.qcmatplotlib import MatPlot
from silq.tools.notebook_tools import *
import pyperclip
from time import time
import numpy as np
import logging
from winsound import Beep

from qcodes.station import Station

logger = logging.getLogger(__name__)

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
        raise NotImplementedError

    def handle_code(self, code, copy=False, execute=False, new_cell=True):
        if copy:
            logger.debug('Copying code to clipboard')
            self.txt_to_clipboard(self.txt)

        if new_cell:
            logger.debug(f'Adding code to new cell below, execute: {execute}')
            create_cell(self.txt, execute=execute, location='below')


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
        self.handle_code(self.txt, copy=self.plot.copy,
                         execute=self.plot.execute)


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
        self.handle_code(self.txt, copy=self.plot.copy,
                         execute=self.plot.execute)


class MoveGates(PlotAction):
    key = 'alt+m'
    action_keys = ['alt+' + key for key in
                   ['up', 'down', 'left', 'right', '-', '+', '=']]
    def __init__(self, plot, key=None):
        self.delta = 0.001
        super().__init__(plot=plot, key=key)

    def key_press(self, event):
        if event.key == self.key:
            if self.plot.point is None:
                self.plot.point = self.plot[0].plot(self.plot.x_gate(),
                                               self.plot.y_gate(), 'ob', )[0]
            else:
                self.plot.point.set_xdata(self.plot.x_gate())
                self.plot.point.set_ydata(self.plot.y_gate())

        elif event.key in ['alt+up', 'alt+down']:
            val = self.plot.y_gate()
            delta = self.delta * (1 if event.key == 'alt+up' else -1)
            self.plot.y_gate(val + delta)
            self.plot.point.set_ydata(val + delta)
        elif event.key in ['alt+left', 'alt+right']:
            val = self.plot.x_gate()
            delta = self.delta * (1 if event.key == 'alt+right' else -1)
            self.plot.x_gate(val + delta)
            self.plot.point.set_xdata(val + delta)
        elif event.key in ['alt++', 'alt+=']:
            self.delta /= 1.5
        elif event.key == 'alt+-':
            self.delta *= 1.5

    def button_press(self, event):
        logger.debug('MoveGates button pressed')
        if event.guiEvent['altKey']:
            logger.debug(f'Moving to gates ({event.xdata}, {event.ydata})')
            self.plot.x_gate(event.xdata)
            self.plot.point.set_xdata(event.xdata)
            self.plot.y_gate(event.ydata)
            self.plot.point.set_ydata(event.ydata)
        else:
            logger.info('Alt key not pressed, not moving gates')


class InteractivePlot(MatPlot):
    def __init__(self, *args, nticks=6, timeout=600, **kwargs):
        super().__init__(*args, **kwargs)
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

    def load_data_array(self, data_array):
        set_arrays = data_array.set_arrays
        labels = []
        gates = []
        for set_array in data_array.set_arrays:
            labels.append(set_array.name)
            gates.append(getattr(self.station, set_array.name, None))
        return {'set_arrays': set_arrays,
                'labels': labels,
                'gates': gates}

    def get_action(self, key=None):
        if key is None:
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
            logger.debug(f'Enabling action {action} with key {event.key}')
            try:
                action.key_press(event)
            except Exception as e:
                logger.error(f'Performing action {action}: {e}')
            self.last_action = action
        elif self.last_action is not None \
                and event.key in self.last_action.action_keys:

            logger.debug(f'Using last action {self.last_action} '
                        f'with key {event.key}')
            self.t_previous = time()
            try:
                self.last_action.key_press(event)
            except Exception as e:
                logger.error(f'Performing action {action}: {e}')
        else:
            pass

    def handle_button_press(self, event):
        self._event_button = event
        logger.debug(f'Clicked (x:{event.xdata:.6}, '
                     f'y:{event.ydata:.6}), '
                     f'alt: {event.guiEvent["altKey"]}')
        action = self.get_action()
        if action is not None:
            self.t_previous = time()
            try:
                action.button_press(event)
            except Exception as e:
                logger.error(f'Button action {action}: {e}')


class DoubleSlider2DPlot(InteractivePlot):
    """
    Used to slide through 2D images of a 4D dataset
    """
    def __init__(self, data_array, **kwargs):
        self.data_array = data_array
        super().__init__(**kwargs)
        self.fig.tight_layout(rect=[0, 0.15, 1, 0.95])

        results = self.load_data_array(data_array)
        self.set_arrays = results['set_arrays']

        self.plot_idx = (0, 0)

        self.add(self.data_array[self.plot_idx],
                 **self.plot_kwargs)

        # Add sliders
        self.slideraxes = [self.fig.add_axes([0.13, 0.02, 0.6, 0.05],
                                             facecolor='yellow'),
                           self.fig.add_axes([0.13, 0.06, 0.6, 0.05],
                                             facecolor='yellow')]
        self.sliders = []
        self.set_vals = []
        for k, sliderax in enumerate(self.slideraxes):
            set_idx = -3 - k
            set_vals = self.set_arrays[set_idx]
            if set_vals.ndim == 2:
                set_vals = set_vals[0]
            slider = mpl.widgets.Slider(ax=sliderax,
                                        label=self.set_arrays[set_idx].name,
                                        valmin=set_vals[0],
                                        valmax=set_vals[-1],
                                        valinit=set_vals[0])
            self.set_vals.append(set_vals)
            slider.on_changed(partial(self.update_slider, k))
            slider.drawon = False
            self.sliders.append(slider)

    @property
    def plot_kwargs(self):
        return {'x': self.set_arrays[-1][self.plot_idx][0],
                'y': self.set_arrays[-2][self.plot_idx],
                'xlabel': self.set_arrays[-1].label,
                'ylabel': self.set_arrays[-2].label,
                'zlabel': self.data_array.label,
                'xunit': self.set_arrays[-1].unit,
                'yunit': self.set_arrays[-2].unit,
                'zunit': self.data_array.unit}

    def update_slider(self, idx, value=None):
        if value is not None:
            # Check if value is one of the set values
            set_idx = -3 - idx
            logger.debug(f'Updating slider {idx} to {value}')
            slider_idx = np.argmin(abs(self.set_vals[idx] - value))

            if idx == 0:
                self.plot_idx = (slider_idx, self.plot_idx[1])
            else:
                self.plot_idx = (self.plot_idx[0], slider_idx)

            logger.debug(f'set_idx: {set_idx}, slider_idx: {slider_idx}, '
                         f'plot_idx: {self.plot_idx}')
            set_value = self.set_vals[idx][slider_idx]
            logger.debug(f'val {value} set_value {set_value}')

            if value != set_value:
                logger.debug(f'val {value} not equal to set_value {set_value}')
                if not np.isnan(value):
                    self.sliders[idx].set_val(set_value)
                return

        # Update plot
        logger.debug(f'val {value} equal to set_value {set_value}')
        self.sliders[idx].valtext.set_text(
            f'{self.set_arrays[set_idx].name}: {value}')

        self[0].clear()
        self[0].add(self.data_array[self.plot_idx], **self.plot_kwargs)
        self.update()


class Slider2DPlot(InteractivePlot):
    """
    Used to slide through 2D images of a 3D dataset
    """
    def __init__(self, data_array, **kwargs):
        self.data_array = data_array
        super().__init__(**kwargs)
        self.fig.tight_layout(rect=[0, 0.1, 1, 0.95])

        results = self.load_data_array(data_array)
        self.set_arrays = results['set_arrays']

        self.plot_idx = 0

        self.add(self.data_array[self.plot_idx], **self.plot_kwargs)

        # Add slider
        self.sliderax = self.fig.add_axes([0.13, 0.02, 0.6, 0.05],
                                          facecolor='yellow')
        self.slider = mpl.widgets.Slider(self.sliderax,
                                         self.set_arrays[-3].name,
                                         float(np.nanmin(self.set_arrays[-3])),
                                         float(np.nanmax(self.set_arrays[-3])),
                                         valinit=self.set_arrays[-3][0])
        self.slider.on_changed(self.update_slider)
        self.slider.drawon = False

    @property
    def plot_kwargs(self):
        return {'x': self.set_arrays[-1][self.plot_idx,0],
                'y': self.set_arrays[-2][self.plot_idx],
                'xlabel': self.set_arrays[-1].label,
                'ylabel': self.set_arrays[-2].label,
                'zlabel': self.data_array.label,
                'xunit': self.set_arrays[-1].unit,
                'yunit': self.set_arrays[-2].unit,
                'zunit': self.data_array.unit}

    def update_slider(self, value):
        try:
            logger.debug(f'Updating slider to {value}')
            self.plot_idx = np.nanargmin(abs(self.set_arrays[-3].ndarray -
                                             value))
            set_value = self.set_arrays[-3][self.plot_idx]

            if value != set_value:
                logger.debug(f'val {value} not equal to set_value {set_value}')
                if not np.isnan(value):
                    self.slider.set_val(set_value)
            else:
                self.slider.valtext.set_text(f'{self.set_arrays[-3].name}: {value}')

                self[0].clear()
                self[0].add(self.data_array[self.plot_idx], **self.plot_kwargs)
                self.update()
        except Exception as e:
            logger.debug(f'Error: {e}')


class CalibrationPlot(InteractivePlot):
    measure_parameter = 'adiabatic_ESR'
    samples_measure = 200
    samples_scan = 100

    def __init__(self, data_set, **kwargs):
        self.data_set = data_set
        if 'voltage_difference' in data_set.arrays:
            super().__init__(data_set.contrast, data_set.dark_counts,
                             data_set.voltage_difference,
                             **kwargs)
        else:
            super().__init__(data_set.contrast, data_set.dark_counts, **kwargs)

        results = self.load_data_array(self.data_set.contrast)
        self.y_gate, self.x_gate = results['gates']
        self.y_label, self.x_label = results['labels']

        self.actions = [SetGates(self), MeasureSingle(self), MoveGates(self)]


class DCPlot(InteractivePlot):
    def __init__(self, data_set,  **kwargs):
        self.data_set = data_set
        super().__init__(data_set.DC_voltage, **kwargs)

        results = self.load_data_array(data_set.DC_voltage)
        self.y_gate, self.x_gate = results['gates']
        self.y_label, self.x_label = results['labels']

        self.actions = [SetGates(self), MoveGates(self)]


class ScanningPlot(InteractivePlot):
    def __init__(self, parameter, interval=0.01, auto_start=False, **kwargs):
        super().__init__(**kwargs)
        self.update_idx = 0
        self.update_start_idx = 1
        self.t_start = None

        self.timer = self.fig.canvas.new_timer(interval=interval * 1000)
        self.timer.add_callback(self.scan)
        self.connect_event('close_event', self.stop)

        self.parameter = parameter

        self.parameter.continuous = auto_start
        if auto_start:
            self.parameter.setup(start=False)
        self.scan(initialize=True, start=True,
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

    @property
    def update_interval(self):
        if self.update_idx > 0:
            return (time() - self.t_start) / (self.update_idx -
                                              self.update_start_idx)

    def start(self, setup=True, start=True):
        self.parameter.continuous = True
        if setup:
            self.parameter.setup(start=start)
        self.timer.start()

        self.update_idx = 0

    def stop(self, *args):
        # *args are needed for if it is a callback
        logger.debug('Stopped')
        self.timer.stop()
        self.layout.stop()
        self.parameter.continuous = False

    def scan(self, initialize=False, start=False, stop=False):
        if self.update_idx == self.update_start_idx:
            self.t_start = time()

        self.results = self.parameter.acquire(start=start, stop=stop)
        self.update_plot(initialize=initialize)

        self.update_idx += 1


class TracePlot(ScanningPlot):
    def __init__(self, parameter, **kwargs):
        subplots = kwargs.pop('subplots', 1)
        if parameter.samples > 1:
            subplots = (len(self.layout.acquisition_outputs()), 1)
        else:
            subplots = 1
        super().__init__(parameter, subplots=subplots, **kwargs)

        # self.actions = [MoveGates(self)]

    def update_plot(self, initialize=False):
        for k, result in enumerate(self.results):
            if initialize:
                setpoints = self.parameter.setpoints[k]
                setpoint_names = self.parameter.setpoint_names[k]
                setpoint_units = self.parameter.setpoint_units[k]
                name = self.parameter.names[k]
                unit = self.parameter.units[k]

                if len(setpoints) == 2:
                    # import pdb; pdb.set_trace()
                    self[k].add(result, x=setpoints[1], y=setpoints[0],
                                xlabel=setpoint_names[1],
                                ylabel=setpoint_names[0],
                                xunit=setpoint_units[1],
                                yunit=setpoint_units[0],
                                zlabel=name,
                                zunit=unit)
                    self[k].y_label, self.x_label = setpoint_names
                    if hasattr(self.station, self.x_label) and \
                            hasattr(self.station, self.y_label):
                        self[k].x_gate = getattr(self.station, self.x_label)
                        self[k].y_gate = getattr(self.station, self.y_label)

                        self[k].plot([self.x_gate.get_latest()],
                                     [self.y_gate.get_latest()], 'ob', ms=5)
                else:
                    print(f'adding plot for {name}')
                    # import pdb; pdb.set_trace()
                    self.add(result[0], x=setpoints[0],
                                xlabel=setpoint_names[0],
                                ylabel=name,
                                xunit=setpoint_units[0],
                                yunit=unit)

            else:
                result_config = self.traces[k]['config']
                if 'z' in result_config:
                    result_config['z'] = result
                else:
                    result_config['y'] = result
        super().update_plot()


class DCSweepPlot(ScanningPlot):
    gate_mapping = {}
    def __init__(self, parameter, gate_mapping=None, **kwargs):
        if gate_mapping is not None:
            self.gate_mapping = gate_mapping

        if parameter.trace_pulse.enabled:
            subplots = {'nrows': 2, 'ncols': 1,
                        'gridspec_kw': {'height_ratios': [2,1]}}
            kwargs['figsize'] = kwargs.get('figsize', (6.5, 6))
        else:
            subplots = 1
        super().__init__(parameter, subplots=subplots, **kwargs)

        if parameter.trace_pulse.enabled:
            self[1].set_ylim(-0.1, 1.3)

        self.actions = [MoveGates(self)]

    def update_plot(self, initialize=False):
        for k, result in enumerate(self.results):
            if initialize:
                setpoints = self.parameter.setpoints[k]
                setpoint_names = self.parameter.setpoint_names[k]
                setpoint_units = self.parameter.setpoint_units[k]
                name = self.parameter.names[k]
                unit = self.parameter.units[k]
                if len(setpoints) == 2:
                    self[k].add(result, x=setpoints[1], y=setpoints[0],
                                xlabel=setpoint_names[1],
                                ylabel=setpoint_names[0],
                                xunit=setpoint_units[1],
                                yunit=setpoint_units[0],
                                zlabel=name,
                                zunit=unit)
                    self.x_label = self.gate_mapping.get(setpoint_names[1],
                                                         setpoint_names[1])
                    self.y_label = self.gate_mapping.get(setpoint_names[0],
                                                         setpoint_names[0])

                    if hasattr(self.station, self.x_label) and \
                            hasattr(self.station, self.y_label):
                        self.x_gate = getattr(self.station, self.x_label)
                        self.y_gate = getattr(self.station, self.y_label)

                        self.point = self[k].plot(self.x_gate.get_latest(),
                                                  self.y_gate.get_latest(),
                                                  'ob', ms=5)[0]
                else:
                    self[k].add(result, x=setpoints[0],
                                xlabel=setpoint_names[0],
                                ylabel=name,
                                xunit=setpoint_units[0],
                                yunit=unit)

            else:
                result_config = self.traces[k]['config']
                if 'z' in result_config:
                    result_config['x'] = self.parameter.setpoints[k][1]
                    result_config['y'] = self.parameter.setpoints[k][0]
                    result_config['z'] = result
                    if self.point is not None:
                        self.point.set_xdata(self.x_gate.get_latest())
                        self.point.set_ydata(self.y_gate.get_latest())
                else:
                    result_config['y'] = result

        super().update_plot()
