from silq.analysis import analysis, fit_toolbox

class T1_Dataset:
    def __init__(self, location, T1_label=None, T1_wait_time=None, skip=[]):
        self.location = location
        self.skip = skip

        self.dataset = qc.load_data(self.location)
        if T1_label is not None:
            self.T1_label = T1_label
        elif 'up_proportion_3_0_0' in self.dataset.arrays.keys():
            self.T1_label = 'up_proportion_3_0_0'
        else:
            self.T1_label = 'up_proportion'

        self.T1_raw_data = self.dataset.arrays[self.T1_label]
        # Filter away the datarows that have not been completed (contain nans)
        self.T1_data = self.filter_T1_data(self.T1_raw_data)

        if T1_wait_time is None:
            self.sort_T1_data(self.dataset.T1_wait_time_set[0],
                              self.T1_data,
                              skip=self.skip)
        else:
            self.T1_wait_times = np.array([T1_wait_time])
            self.T1_data = np.reshape(self.T1_data,(self.max_idx,1))

        self.analyse_T1_data()

    def filter_T1_data(self, T1_raw_data):
        self.max_idx = np.sum(
            [~np.isnan(np.sum(datarow)) for datarow in T1_raw_data])
        print('Number of successful sweeps: {}'.format(self.max_idx))
        self.T1_data = T1_raw_data[:self.max_idx]
        return self.T1_data

    def sort_T1_data(self, T1_wait_times_unsorted, T1_data_unsorted, skip=[]):
        self.T1_wait_times = np.sort(T1_wait_times_unsorted)
        self.T1_data_idx = np.argsort(T1_wait_times_unsorted)

        self.T1_data = T1_data_unsorted[:, self.T1_data_idx]

        if skip:
            self.T1_wait_times = np.delete(self.T1_wait_times, skip, 0)
            self.T1_data = np.delete(self.T1_data, skip, 1)
            self.T1_data_idx = np.delete(self.T1_data_idx, skip, 0)

        return self.T1_wait_times, self.T1_data, self.T1_data_idx

    def analyse_T1_data(self, T1_data=None):
        if T1_data is None:
            T1_data = self.T1_data
        self.T1_data_mean = np.mean(T1_data, axis=0)
        self.T1_data_std = np.std(T1_data, axis=0)
        self.T1_data_std_mean = self.T1_data_std / np.sqrt(len(T1_data))

        return self.T1_data_mean, self.T1_data_std, self.T1_data_std_mean


class T1_Measurement:
    def __init__(self, location, B0, T1_label=None, skip=[]):
        self.B0 = B0
        self.datasets = []
        self.add_data(location=location,
                      skip=skip,
                      T1_label=T1_label)

    def add_data(self, location, T1_label=None, T1_wait_time=None, skip=[]):
        self.datasets.append(T1_Dataset(location=location,
                                        T1_label=T1_label,
                                        T1_wait_time=T1_wait_time,
                                        skip=skip))

        self.merge_datasets()
        self.fit_T1(weights='std')

    def merge_datasets(self, datasets=None):
        if datasets is None:
            datasets = self.datasets

        # Merge T1 wait times arrays and sort in ascending order
        T1_wait_times = np.concatenate(
            [dataset.T1_wait_times for dataset in datasets])
        sort_idx = np.argsort(T1_wait_times)

        self.T1_wait_times = T1_wait_times[sort_idx]
        self.T1_data_mean = np.concatenate(
            [dataset.T1_data_mean for dataset in datasets])[sort_idx]
        self.T1_data_std = np.concatenate(
            [dataset.T1_data_std for dataset in datasets])[sort_idx]
        self.T1_data_std_mean = np.concatenate(
            [dataset.T1_data_std_mean for dataset in datasets])[sort_idx]

        return (self.T1_wait_times, self.T1_data_mean,
                self.T1_data_std, self.T1_data_std_mean)

    def fit_T1(self, T1_wait_times=None, T1_data_mean=None,
               datasets=None, weights=None):
        if T1_wait_times is None:
            T1_wait_times = self.T1_wait_times
        if T1_data_mean is None:
            T1_data_mean = self.T1_data_mean
        if datasets is None:
            datasets = self.datasets
        if weights == 'std':
            std = np.std(self.T1_data_std_mean, axis=0) / len(self.T1_data_std_mean)
            weights = 1 / std ** 2
        else:
            weights = [1] * len(T1_wait_times)

        self.fit_model = fit_toolbox.ExponentialFit()
        self.fit_result = self.fit_model.perform_fit(T1_wait_times,
                                           T1_data_mean,
                                           weights=weights)
        self.T1 = self.fit_result.params['tau'].value
        self.T1_std = self.fit_result.params['tau'].stderr
        return self.fit_result

    def plot_T1(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        # Plot individual datapoints
        for dataset in self.datasets:
            for datarow in dataset.T1_data:
                ax.plot(dataset.T1_wait_times, datarow, 'ok', ms=3, alpha=0.2)

        # Plot mean datapoints with errorbars
        (_, caps, _) = ax.errorbar(self.T1_wait_times, self.T1_data_mean,
                                   yerr=self.T1_data_std_mean,
                                   marker='o', linestyle='', ms=10)

        # Choose correct widths for errorbars
        for cap in caps:
            cap.set_markeredgewidth(3)

        # Plot fit curve
        plt.plot(self.T1_wait_times, self.fit_result.best_fit, 'r-')

        ax.set_xscale("log")
        ax.set_xlim([0.9 * self.T1_wait_times[0], 1.1 * self.T1_wait_times[-1]])
        ax.set_ylim([0, 1])

        ax.set_xlabel('Wait time (ms)')
        ax.set_ylabel('Up population')

        return ax