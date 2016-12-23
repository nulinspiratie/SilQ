


class MeasurementSequence:
    def __init__(self, name=None, condition_set=None):
        self.measurements = []
        self.name = name
        self.condition_set = condition_set
        self.condition_sets = [] if condition_set is None else [condition_set]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.measurements[index]
        elif isinstance(index, str):
            measurements = [m for m in self.measurements if m.name == index]
            assert len(measurements) == 1, \
                "Found more than one measurement with name {}".format(index)
            return measurements[0]

    def __iter__(self):
        self.measurement = self.measurements[0]
        self.num_measurements = 0

    def __next__(self):
        if self.measurement is None:
            raise StopIteration

        self.num_measurements += 1
        self.measurement()
        result = self.measurement.check_condition_sets(*self.condition_sets)

        if result['action'] is None or result['action'][:4] == 'next':
            self.measurement = self.next_measurement
        else:
            self.measurement = None

        return result

    def __call__(self):
        self.results = [result for result in self]
        result = self.results[-1]
        if result['action'] is None:
            # TODO better wa to discern success from fail
            result['action'] = 'success' if result['is_satisfied'] else 'fail'
        elif result['action'][:4] == 'next':
            result['action'] = result['action'][5:]

        return result

    @property
    def next_measurement(self):
        """ Get measurement after self.measurement. In case there is no next
        measurement, returns None """
        if self.measurement is None:
            return None
        msmt_idx = self.measurements.index(self.measurement)
        if msmt_idx + 1 == len(self.measurements):
            # No next measurement
            return None
        else:
            return self.measurements[msmt_idx + 1]