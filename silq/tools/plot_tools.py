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

__all__ = ['PlotAction', 'SetGates', 'MeasureSingle', 'MoveGates',
           'SwitchPlotIdx', 'InteractivePlot', 'SliderPlot', 'CalibrationPlot',
           'DCPlot', 'ScanningPlot', 'TracePlot', 'DCSweepPlot']

logger = logging.getLogger(__name__)


class PlotAction:
    enable_key = None

    def __init__(self, plot, timeout=None, enable_key=None):
        self.timeout = timeout
        self.t_enable_key_pressed = None

        if enable_key is not None:
            self.enable_key = enable_key

        self.plot = plot

    @property
    def enabled(self):
        if self.enable_key is None:
            # No enable key specified, so always enabled
            return True
        elif self.t_enable_key_pressed is None:
            # Enable key never pressed, so disabled
            return False
        elif self.timeout is None:
            # Enable key pressed, and no timeout, so enabled
            return True
        else:
            # Depends on if last enable_key press was within timeout seconds
            return time() - self.t_enable_key_pressed < self.timeout

    def txt_to_clipboard(self, txt):
        pyperclip.copy(txt)

    def key_press(self, event):
        if event.key == self.enable_key:
            logger.debug(f'Enabling action {self}')
            self.t_enable_key_pressed = time()

    def button_press(self, event):
        pass

    def handle_code(self, code, copy=False, execute=False, new_cell=True):
        if copy:
            logger.debug('Copying code to clipboard')
            self.txt_to_clipboard(self.txt)

        if new_cell:
            logger.debug(f'Adding code to new cell below, execute: {execute}')
            create_cell(self.txt, execute=execute, location='below')


class SetGates(PlotAction):
    enable_key = 'alt+g'

    def button_press(self, event):
        super().button_press(event)

        self.txt = f"{self.plot.x_label}({event.xdata:.5f})\n" \
                   f"{self.plot.y_label}({event.ydata:.5g})"
        self.handle_code(self.txt, copy=True, execute=False)


class MeasureSingle(PlotAction):
    enable_key = 'alt+s'

    def button_press(self, event):
        super().button_press(event)

        self.txt = f"{self.plot.x_label}({event.xdata:.5f}\n" \
                   f"{self.plot.y_label}({event.ydata:.5g})\n" \
                   f"{self.plot.measure_parameter}_parameter.single_settings" \
                   f"(samples={self.plot.samples_measure}, silent=False)\n"\
                   f"qc.Measure({param}_parameter).run" \
                   f"(name='{self.plot.measure_parameter}_measure', quiet=True)"
        self.handle_code(self.txt, copy=True, execute=False)


class MoveGates(PlotAction):
    enable_key = 'alt+m'
    delta = 0.001 # step to move when pressing a key

    def key_press(self, event):
        super().key_press(event)

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
        else:
            pass

    def button_press(self, event):
        super().button_press(event)

        logger.debug('MoveGates button pressed')
        if event.guiEvent['altKey']:
            logger.debug(f'Moving to gates ({event.xdata}, {event.ydata})')
            self.plot.x_gate(event.xdata)
            self.plot.point.set_xdata(event.xdata)
            self.plot.y_gate(event.ydata)
            self.plot.point.set_ydata(event.ydata)
        else:
            logger.info('Alt key not pressed, not moving gates')


class TuneCompensation(PlotAction):
    enable_key = 'alt+c'

    def __init__(self, *args, **kwargs):
        """ A tool to plot a line of compensated plunging/emptying on a DC scan.

        Key commands:
          'alt+c'            : enables this tool
          'alt+<arrow_key>'  : move the central (read) position of your line in that respective direction
          'alt+<+ or ->'     : increase/decrease the compensation angle with the horizontal
          'alt+<8 or 2>'     : increase/decrease the empty depth
          'alt+<6 or 4>'     : increase/decrease the plunge depth
          'alt+5'            : reset all parameters to their default values
        """
        super().__init__(*args, **kwargs)
        self.default_empty_depth = 10e-3
        self.default_plunge_depth = -10e-3
        self.default_theta = -45 # degrees
        self.default_y_read = np.nanmean(self.plot.data_set.DC_voltage.set_arrays[0])
        self.default_x_read = np.nanmean(self.plot.data_set.DC_voltage.set_arrays[1][0])
        self.plot_feats = []

        self.delta_v = 1e-3
        self.delta_t = 1

        self.initialize_parameters()

    def initialize_parameters(self):
        self.plunge_depth = self.default_plunge_depth
        self.empty_depth = self.default_empty_depth
        self.theta = self.default_theta
        self.x_read = self.default_x_read
        self.y_read = self.default_y_read

    def key_press(self, event):
        super().key_press(event)

        if event.key in ['alt+up', 'alt+down', 'alt+left', 'alt+right', 'alt+8',
                         'alt+6', 'alt+4', 'alt+2', 'alt+5', 'alt++', 'alt+-']:

            # Tune compensation
            if event.key == 'alt++':
                self.theta += self.delta_t
            elif event.key == 'alt+-':
                self.theta -= self.delta_t

            # Tune read position
            elif event.key == 'alt+up':
                self.y_read += self.delta_v
            elif event.key == 'alt+down':
                self.y_read -= self.delta_v
            elif event.key == 'alt+right':
                self.x_read += self.delta_v
            elif event.key == 'alt+left':
                self.x_read -= self.delta_v

            # Tune empty depth
            elif event.key == 'alt+8':
                self.empty_depth += self.delta_v
            elif event.key == 'alt+2':
                self.empty_depth -= self.delta_v

            # Tune plunge depth
            elif event.key == 'alt+6':
                self.plunge_depth += self.delta_v
            elif event.key == 'alt+4':
                self.plunge_depth -= self.delta_v

            # Reset
            elif event.key == 'alt+5':
                self.initialize_parameters()

        self.draw_features()

    def draw_features(self):
        # Remove old features before drawing new ones
        for plot_feat in self.plot_feats:
            f = plot_feat.pop(0)
            f.remove()
            del f

        set_vals = self.plot.data_set.DC_voltage.set_arrays[1][0]

        # Calculate compensation factor
        self.compensation = np.tan(np.deg2rad(self.theta))

        read_pt = self.plot[0].plot(self.x_read, self.y_read, 'ro',
                                    markeredgewidth=2, markerfacecolor='None')

        empty_pt = self.plot[0].plot(self.x_read + self.empty_depth / np.sqrt(1 + 1 / self.compensation ** 2),
                                     self.y_read + np.sign(self.compensation) *
                                     self.empty_depth / np.sqrt(self.compensation ** 2 + 1),
                                     'go', markeredgewidth=2, markerfacecolor='None')

        plunge_pt = self.plot[0].plot(self.x_read + self.plunge_depth / np.sqrt(1 + 1 / self.compensation ** 2),
                                      self.y_read + np.sign(self.compensation) *
                                      self.plunge_depth / np.sqrt(self.compensation ** 2 + 1),
                                      'yo', markeredgewidth=2, markerfacecolor='None')

        comp_line = self.plot[0].plot(set_vals,
                                      (set_vals - self.x_read) / self.compensation + self.y_read, 'c')

        self.plot_feats = [read_pt, empty_pt, plunge_pt, comp_line]

        self.plot.update()


class SwitchPlotIdx(PlotAction):
    def key_press(self, event):
        super().key_press(event)

        plot_idx = list(self.plot.plot_idx)
        if event.key in ['alt+left', 'alt+right']:
            set_vals = self.plot.set_vals[0]
            if event.key == 'alt+left':
                plot_idx[0] = max(plot_idx[0] - 1, 0)
            else:
                plot_idx[0] = min(plot_idx[0] + 1, len(set_vals) - 1)
            self.plot.plot_idx = tuple(plot_idx)
            self.plot.update_slider(0)
        elif event.key in ['alt+down', 'alt+up'] and len(plot_idx) > 1:
            set_vals = self.plot.set_vals[1]
            if event.key == 'alt+down':
                plot_idx[1] = max(plot_idx[1] - 1, 0)
            else:
                plot_idx[1] = min(plot_idx[1] + 1, len(set_vals) - 1)
            self.plot.plot_idx = tuple(plot_idx)
            self.plot.update_slider(1)


class InteractivePlot(MatPlot):
    def __init__(self, *args, actions=(), timeout=600, **kwargs):
        super().__init__(*args, **kwargs)
        self.station = Station.default
        self.layout = getattr(self.station, 'layout', None)

        self.actions = actions
        # setting timeout sets it for all actions
        self.timeout = timeout

        # cid is used to connect specific functions to event handlers
        self.cid = {}

        self._event_key = None
        self._event_button = None

        self.connect_event('key_press_event', self.handle_key_press)
        self.connect_event('button_press_event', self.handle_button_press)

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        self._timeout = timeout
        for action in self.actions:
            action.timeout = timeout

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

    def connect_event(self, event, action):
        if event in self.cid:
            self.fig.canvas.mpl_disconnect(self.cid[event])

        cid = self.fig.canvas.mpl_connect(event, action)
        self.cid[event] = cid

    def handle_key_press(self, event):
        self._event_key = event
        logger.debug(f'Key pressed: {event.key}')
        try:
            for action in self.actions:
                if action.enabled or event.key == action.enable_key:
                    action.key_press(event=event)
        except Exception as e:
            logger.error(f'key press: {e}')

    def handle_button_press(self, event):
        self._event_button = event
        logger.debug(f'Clicked (x:{event.xdata:.6}, y:{event.ydata:.6})')
        try:
            for action in self.actions:
                if action.enabled:
                    action.button_press(event)
        except Exception as e:
            logger.error(f'button press: {e}')


class SliderPlot(InteractivePlot):
    """
    Used to slide through 2D images of a 4D dataset
    """
    def __init__(self, data_array, ndim=2, **kwargs):
        self.ndim = ndim
        self.data_array = data_array
        super().__init__(actions=[SwitchPlotIdx(self)], **kwargs)
        self.fig.tight_layout(rect=[0, 0.15, 1, 0.95])

        results = self.load_data_array(data_array)
        self.set_arrays = results['set_arrays']

        self.num_sliders = len(self.set_arrays) - self.ndim

        self.plot_idx = tuple(0 for _ in self.set_arrays[:-self.ndim])

        self.add(self.data_array[self.plot_idx], **self.plot_kwargs)

        # Add sliders
        self.slideraxes = [self.fig.add_axes([0.13, 0.02 + 0.04*k, 0.6, 0.05],
                                             facecolor='yellow')
                           for k in range(self.num_sliders)]

        self.sliders = []
        self.set_vals = []
        for k, sliderax in enumerate(self.slideraxes):
            set_idx = -self.ndim - (k + 1)
            set_vals = self.set_arrays[set_idx]
            if set_vals.ndim == 2:
                # Make more general for > 2D
                set_vals = set_vals[0]
            slider = mpl.widgets.Slider(ax=sliderax,
                                        label=self.set_arrays[set_idx].name,
                                        valmin=np.nanmin(set_vals),
                                        valmax=np.nanmax(set_vals),
                                        valinit=set_vals[0])
            self.set_vals.append(set_vals)
            slider.on_changed(partial(self.update_slider, k))
            slider.drawon = False
            self.sliders.append(slider)

    @property
    def plot_kwargs(self):
        if self.ndim == 1:
            return {'x': self.set_arrays[-1][self.plot_idx][0],
                    'xlabel': self.set_arrays[-1].label,
                    'ylabel': self.data_array.label,
                    'xunit': self.set_arrays[-1].unit,
                    'yunit': self.data_array.unit,}
        elif self.ndim == 2:
            return {'x': self.set_arrays[-1][self.plot_idx][0],
                    'y': self.set_arrays[-2][self.plot_idx],
                    'xlabel': self.set_arrays[-1].label,
                    'ylabel': self.set_arrays[-2].label,
                    'zlabel': self.data_array.label,
                    'xunit': self.set_arrays[-1].unit,
                    'yunit': self.set_arrays[-2].unit,
                    'zunit': self.data_array.unit}
        else:
            raise NotImplementedError(f'{self.ndim} dims not supported')

    def update_slider(self, idx, value=None):
        if value is None:
            value = self.set_vals[idx][self.plot_idx[idx]]
            self.sliders[idx].set_val(value)
        elif value == self.set_vals[idx][self.plot_idx[idx]]:
            self.update()
        else:
            # Check if value is one of the set values
            logger.debug(f'Updating slider {idx} to {value}')
            slider_idx = np.nanargmin(abs(self.set_vals[idx] - value))

            self.plot_idx = tuple(val if k != idx else slider_idx
                                  for k, val in enumerate(self.plot_idx))
            value = self.set_vals[idx][self.plot_idx[idx]]
            self.sliders[idx].set_val(value)

    def update(self):
        # Update plot
        self[0].clear()
        self[0].add(self.data_array[self.plot_idx], **self.plot_kwargs)


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

        self.actions = [SetGates(self), MoveGates(self), TuneCompensation(self)]


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
        self.scan(initialize=True, stop=(not auto_start))

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

    def scan(self, initialize=False, stop=False):
        if self.update_idx == self.update_start_idx:
            self.t_start = time()

        self.parameter()
        if stop:
            self.layout.stop()
        self.update_plot(initialize=initialize)

        self.update_idx += 1


class TracePlot(ScanningPlot):
    def __init__(self, parameter, **kwargs):
        subplots = kwargs.pop('subplots', 1)
        average_mode = getattr(parameter, 'average_mode', 'none')
        if parameter.samples > 1 and average_mode == 'none':
            subplots = (len(self.layout.acquisition_outputs()), 1)
        else:
            subplots = 1
        super().__init__(parameter, subplots=subplots, **kwargs)

        # self.actions = [MoveGates(self)]

    def update_plot(self, initialize=False):
        for k, name in enumerate(self.parameter.names):
            result = self.parameter.results[name]
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

        self.point = None
        super().__init__(parameter, subplots=subplots, **kwargs)

        if parameter.trace_pulse.enabled:
            self[1].set_ylim(-0.1, 1.3)

        self.actions = [MoveGates(self)]

    def update_plot(self, initialize=False):
        for k, name in enumerate(self.parameter.names):
            result = self.parameter.results[name]
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
