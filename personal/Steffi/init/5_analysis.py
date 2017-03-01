from silq.analysis import analysis, fit_toolbox

class T1_Dataset:
    def __init__(self, location, T1_label=None,
                 T1_wait_time_label='T1_wait_time_set',
                 T1_wait_time=None, skip=[]):
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
        # Filter away thwe datarows that have not been completed (contain nans)
        self.T1_data = self.filter_T1_data(self.T1_raw_data)

        if T1_wait_time is None:
            T1_wait_times_unsorted = getattr(self.dataset,
                                             T1_wait_time_label)[0]
            self.sort_T1_data(T1_wait_times_unsorted,
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


class T1_Analysis:
    def __init__(self, location, skip_rows=None, T1_label=None,
                 T1_wait_time_label=None, analyse=True,     **kwargs):
        self.location = location
        self.dataset = qc.load_data(self.location)

        self.skip_rows = skip_rows
        self.T1_label = self.get_label(T1_label, 'up_proportion')
        self.T1_wait_time_label = self.get_label(T1_wait_time_label,
                                                 'T1_wait_time')

        if analyse:
            self.analyse_data()

    def get_label(self, label, default_label):
        """
        Finds the label of a data array in the dataset. If label is None,
        it searches for the closest match to default_label
        Args:
            label:
            default_label:

        Returns:

        """
        array_labels = self.dataset.arrays.keys()
        if label is not None:
            if label in array_labels:
                return label
            else:
                raise Exception('Label {} not found in dataset'.format(label))
        else:
            label = default_label
            labels = [l for l in array_labels if label in l]
            assert len(labels)==1,'Could not find unique label containing ' \
                                  '"{}", labels found: {}'.format(label, labels)
            return labels[0]

    def analyse_data(self, T1_data=None, T1_wait_times=None):
        # Filter and sort data
        self.T1_data = T1_data
        if T1_data is None:
            self.T1_data = self.dataset.arrays[self.T1_label]

        self.filter_data()

        self.T1_wait_times = T1_wait_times
        if T1_wait_times is None:
            self.T1_wait_times = getattr(self.dataset,  self.T1_wait_time_label)[0]

        self.sort_data()

        # Find mean, std, and std of mean
        self.T1_data_mean = np.mean(self.T1_data, axis=0)
        self.T1_data_std = np.std(self.T1_data, axis=0)
        self.T1_data_std_mean = self.T1_data_std / np.sqrt(len(self.T1_data))

        # Perform fitting
        self.fit_model = fit_toolbox.ExponentialFit(
            sweep_vals=self.T1_wait_times, data=self.T1_data_mean)
        self.setup_fit_parameters()
        self.fit_T1()

    def filter_data(self):
        """
        Select the T1 sweeps that do not have any NaN values, and removes
        rows in self.skip_rows. This modifies the value of self.T1_data

        Returns:
            Data where none of the rows have NaN values
        """
        self.max_idx = np.sum([~np.isnan(np.sum(datarow))
                               for datarow in self.T1_data])
        print('Number of successful sweeps: {}'.format(self.max_idx))
        self.T1_data = self.T1_data[:self.max_idx]

        if self.skip_rows is not None:
            self.T1_data = np.delete(self.T1_data, self.skip_rows, 1)

        return self.T1_data

    def sort_data(self):
        """
        Sorts T1 data, in case the wait times are shuffled.
        This modifies the value of self.T1_data and self.T1_wait_times
        Args:

        Returns:
            T1 wait times, T1_data, and T1_data_idx.
        """
        self.T1_data_idx = np.argsort(self.T1_wait_times)
        self.T1_wait_times = np.sort(self.T1_wait_times)

        self.T1_data = self.T1_data[:, self.T1_data_idx]

        return self.T1_wait_times, self.T1_data, self.T1_data_idx

    def setup_fit_parameters(self):
        self.parameters = self.fit_model.find_initial_parameters()
        self.parameters['tau'].set(min=0, max=20e3)
        self.parameters['amplitude'].set(min=0, max=3)
        self.parameters['offset'].set(min=0, max=3)
        return self.parameters

    def fit_T1(self, weights=None):
        if weights == 'std':
            weights = 1 / self.T1_data_std_mean ** 2
        else:
            weights = [1] * len(self.T1_wait_times)

        self.fit_result = self.fit_model.fit(
            data=self.T1_data_mean, t=self.T1_wait_times,
            weights=weights, params=self.parameters)
        self.T1 = self.fit_result.params['tau'].value
        self.T1_std = self.fit_result.params['tau'].stderr
        return self.fit_result

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
                                   marker = 'o', linestyle = '', ms = 10)

        # Choose correct widths for errorbars
        for cap in caps:
            cap.set_markeredgewidth(3)

        # Plot fit curve
        ax.plot(self.T1_wait_times, self.fit_result.best_fit, 'r-')

        ax.set_xscale("log")
        ax.set_xlim([0.9 * self.T1_wait_times[0], 1.1 * self.T1_wait_times[-1]])
        ax.set_ylim([0, 1])

        ax.text(0.5, 0.9, '$T_1={:.2f} \pm {:.2f} \mathrm{{s}}$'.format(
            self.T1 / 1e3, self.T1_std / 1e3),
                horizontalalignment='center', transform=ax.transAxes,
                fontsize=16)

        ax.set_xlabel('Wait time (ms)')
        ax.set_ylabel('Up population')

        return ax