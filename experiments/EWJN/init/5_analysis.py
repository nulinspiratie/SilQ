from silq.analysis import analysis, fit_toolbox

class T1_Measurement:
    def __init__(self, location, B0, skip=[]):
        self.B0 = B0
        self.location = location
        self.skip = skip

        self.dataset = qc.load_data(self.location)

        if 'up_proportion_3_0_0' in self.dataset.arrays.keys():
            self.T1_label = 'up_proportion_3_0_0'
        else:
            self.T1_label = 'up_proportion'

        self.T1_wait_times, self.T1_data, self.T1_data_idx = self.sort_T1_data(
            self.dataset.T1_wait_time_set[0],
            self.dataset.arrays[self.T1_label],
            skip=self.skip)

        self.T1_data_mean, self.T1_data_std, self.T1_data_std_mean = self.analyse_T1_data()
        self.T1_fit_result = self.fit_T1(self.T1_wait_times,
                                         self.T1_data_mean)

    def sort_T1_data(self, T1_wait_times_unsorted, T1_data_unsorted, skip=[]):
        T1_wait_times = np.sort(T1_wait_times_unsorted)
        T1_data_idx = np.argsort(T1_wait_times_unsorted)

        self.max_idx = np.sum(
            [~np.isnan(np.sum(datarow)) for datarow in T1_data_unsorted])
        print('Number of successful sweeps: {}'.format(self.max_idx))
        T1_data = T1_data_unsorted[:self.max_idx, T1_data_idx]

        if skip:
            T1_wait_times = np.delete(T1_wait_times, skip, 0)
            T1_data = np.delete(T1_data, skip, 1)
            T1_data_idx = np.delete(T1_data_idx, skip, 0)

        return T1_wait_times, T1_data, T1_data_idx

    def analyse_T1_data(self, T1_data=None):
        if T1_data is None:
            T1_data = self.T1_data
        T1_data_mean = np.mean(T1_data, axis=0)
        T1_data_std = np.std(T1_data, axis=0)
        T1_data_std_mean = T1_data_std / np.sqrt(len(T1_data))

        return T1_data_mean, T1_data_std, T1_data_std_mean

    def fit_T1(self, T1_wait_times=None, T1_data=None, weights='std'):
        #         pass
        if T1_wait_times is None:
            T1_wait_times = self.T1_wait_times
        if T1_data is None:
            T1_data = self.T1_data
        if weights == 'std':
            std = np.std(T1_data, axis=0) / len(T1_data)
            weights = 1 / std ** 2
        else:
            weights = [1] * len(T1_data)

        fit_model = fit_toolbox.ExponentialFit()
        fit_result = fit_model.perform_fit(T1_wait_times,
                                           T1_data,
                                           weights=weights)
        self.T1 = fit_result.params['tau'].value
        self.T1_std = fit_result.params['tau'].stderr
        return fit_result

    def plot_T1(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        # Plot individual datapoints
        for datarow in self.T1_data:
            ax.plot(self.T1_wait_times, datarow, 'ok', ms=3, alpha=0.2)

        # Plot mean datapoints with errorbars
        (_, caps, _) = ax.errorbar(self.T1_wait_times, self.T1_data_mean,
                                   yerr=self.T1_data_std_mean,
                                   marker='o', linestyle='', ms=10)

        # Choose correct widths for errorbars
        for cap in caps:
            cap.set_markeredgewidth(3)

        # Plot fit curve
        plt.plot(self.T1_wait_times, self.T1_fit_result.best_fit, 'r-')

        ax.set_xscale("log")
        ax.set_xlim([0.9 * self.T1_wait_times[0], 1.1 * self.T1_wait_times[-1]])
        ax.set_ylim([0, 0.55])

        ax.set_xlabel('Wait time (ms)')
        ax.set_ylabel('Up population')

        return ax