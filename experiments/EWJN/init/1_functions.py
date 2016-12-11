from silq.functions.general_functions import partial_from_attr

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


def get_voltages(SIM900):
    voltages = {}

    for channel, name in SIM900.channels().items():
        voltages[name] = SIM900.parameters[name]()
    return voltages

def ramp_to_voltages(target_voltages, SIM900=None, channels=None):
    if channels is None:
        channels = [SIM900.parameters[ch_name] for ch_name in
                    SIM900.channels().values()]

    if isinstance(target_voltages, int):
        target_voltages = {channel.name: target_voltages for channel in
                           channels}

    initial_voltages = {channel.name: channel() for channel in channels}
    for ratio in np.linspace(0, 1, 11):
        for channel in channels:
            voltage = (1 - ratio) * initial_voltages[channel.name] + \
                      ratio * target_voltages[channel.name]
            channel(voltage)