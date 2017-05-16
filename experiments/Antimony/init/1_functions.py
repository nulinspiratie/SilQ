# General functions
def plot_traces(traces, ax=None, traces_AWG=None, threshold_voltage=None,
                plot1D=False):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    else:
        plt.sca(ax)

    cax = ax.pcolormesh(range(traces.shape[1]),
                        range(traces.shape[0] + 1), traces)
    if traces_AWG is not None:
        trace_AWG = traces_AWG[:1]
        trace_AWG /= (np.max(trace_AWG) - np.min(trace_AWG))
        trace_AWG -= np.min(trace_AWG)
        ax.pcolormesh(range(traces.shape[1]),
                      np.array([0, 1]) + traces.shape[0], trace_AWG)
    ax.set_xlim([0, traces.shape[1]])
    ax.set_ylim([0, traces.shape[0] + 1])
    ax.invert_yaxis()

    plt.colorbar(cax)

    if plot1D:
        fig, axes = plt.subplots(len(traces), sharex=True)
        for k, trace in enumerate(traces):
            axes[k].plot(trace)
            #         axes[k].plot(trace > 0.5)
            if traces_AWG is not None:
                trace_AWG = traces_AWG[k]
                trace_AWG /= (np.max(trace_AWG) - np.min(trace_AWG))
                trace_AWG -= np.min(trace_AWG)
                axes[k].plot(trace_AWG)
            if threshold_voltage is not None:
                axes[k].plot([threshold_voltage] * len(trace), 'r')
            axes[k].locator_params(nbins=2)


def try_close_instruments(
        instruments=['pulseblaster', 'arbstudio', 'SIM900', 'ATS',
                     'ATS_controller', 'pulsemaster'],
        reload=False):
    if isinstance(instruments, str):
        instruments = [instruments]
    for instrument_name in instruments:
        try:
            eval('{}.close()'.format(instrument_name))
        except:
            pass
        try:
            eval('reload({}_driver)'.format(instrument_name))
        except:
            pass

if in_notebook():
    @register_cell_magic
    def label(line, cell):
        # Add cell magic %label {lbl}, that can be executed later on
        global code_labels
        code_labels[line] = cell


def create_window(window, *args, use_thread=True):
    if use_thread:
        t = threading.Thread(target=create_window, name='gui',
                             args=(window, *args),
                             kwargs={'use_thread': False})
        t.start()
        return t
    else:
        qApp = QApplication(sys.argv)
        aw = window(*args)
        aw.show()
        qApp.exec_()
        return qApp


def sim_gui():
    from silq.gui.SIMGui import SIMControlDialog
    global SIM900_scaled_parameters
    create_window(SIMControlDialog, SIM900_scaled_parameters)

def parameter_info(self, *parameter_names, detailed=False):
    snapshot = self.snapshot()
    param_info = {}
    for parameter_name in parameter_names:
        param_snapshot = snapshot['station']['parameters'][parameter_name]
        if detailed:
            param_info[parameter_name] = param_snapshot
        else:
            param_info[parameter_name] = param_snapshot['value']
    return param_info
DataSet.parameter_info = parameter_info