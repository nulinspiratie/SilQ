from matplotlib import rcParams
rcParams['figure.max_open_warning'] = 80


if hasattr(qc, 'MatPlot'):
    qc.MatPlot.plot_2D_kwargs = {'cmap': 'inferno'}


class CalibrationPlot:
    def __init__(self, dataset, figsize=None, interval=5, nticks=6,
                 on_click=None):
        self.dataset = dataset
        self.interval = interval
        self.cid = None

        self.extract_gates('contrast')
        self.create_figure(figsize=figsize)
        self.plot_data(nticks=nticks)

        if on_click is not None:
            self.connect_event('gates_to_clipboard')

    def create_figure(self, figsize=None):
        if hasattr(self.dataset, 'fidelity_empty'):
            if figsize is None:
                figsize = (10, 8)
            self.plot = qc.MatPlot(subplots=(2, 2), figsize=figsize,
                                   interval=self.interval)
        else:
            if figsize is None:
                figsize = (12, 5)
            self.plot = qc.MatPlot(subplots=(1, 2), figsize=figsize,
                                   interval=self.interval)

    def plot_data(self, nticks=6):
        self.plot.add(self.dataset.contrast, subplot=1, nticks=nticks)
        self.plot.add(self.dataset.dark_counts, subplot=2, nticks=nticks)

        if hasattr(self.dataset, 'fidelity_empty'):
            self.plot.add(self.dataset.fidelity_load, subplot=3, nticks=nticks)
            self.plot.add(self.dataset.fidelity_empty, subplot=4, nticks=nticks)

        self.fig = self.plot.fig
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def extract_gates(self, key):
        self.x = getattr(self.dataset, key).set_arrays[1].name
        self.y = getattr(self.dataset, key).set_arrays[0].name

    def connect_event(self, event_label):
        self.event_label = event_label

        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)

        if event_label == 'gates_to_clipboard':
            self.fig.canvas.mpl_connect('button_press_event',
                                        self.gates_to_clipboard)
        else:
            raise SyntaxError(
                'event label {} not understood'.format(event_label))

    def gates_to_clipboard(self, event):
        Beep(2500, 200)
        pyperclip.copy('{x}({x_val})\n{y}({y_val})'.format(
            x=self.x, y=self.y,
            x_val=event.xydata[0], y_val=event.xydata[1]))
