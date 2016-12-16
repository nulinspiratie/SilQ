



class SelectFrequency_Parameter(MeasurementParameter):
    def __init__(self, threshold=0.5, discriminant='contrast', **kwargs):
        self.frequencies = None
        self.frequency = None

        names = [discriminant + spin_state for spin_state in self.spin_states]
        if 'mode' in kwargs:
            names.append('frequency_{}'.format(kwargs['mode']))
        else:
            names.append('frequency')

        super().__init__(name='select_frequency',
                         label='Select frequency',
                         snapshot_value=False,
                         names=names,
                         **kwargs)

        self.update_frequency = True
        self.threshold = threshold
        self.discriminant = discriminant

        self.measure_parameter = Adiabatic_Parameter(**kwargs)

        self._meta_attrs.extend(['frequencies', 'frequency', 'update_frequency',
                                 'threshold', 'discriminant'])

    @property
    def spin_states(self):
        spin_states_unsorted = self.frequencies.values()
        return sorted(spin_states_unsorted)

    @property
    def discriminant_idx(self):
        return self.measure_parameter.names.index(self.discriminant)

    def get(self):
        self.results = []
        # Perform measurement for all frequencies
        for spin_state in self.spin_states:
            # Set adiabatic frequency
            self.measure_parameter(self.frequencies[spin_state])
            fidelities = self.measure_parameter()

            # Only add dark counts and contrast
            self.results.append(fidelities[self.discriminant_idx])

            # Store raw traces if self.save_traces is True
            if self.save_traces:
                saved_traces = {
                    'acquisition_traces': self.data['acquisition_traces'][
                        'output']}
                if 'initialization_traces' in self.data:
                    saved_traces['initialization'] = \
                        self.data['initialization_traces']
                if 'post_initialization_traces' in self.data:
                    saved_traces['post_initialization_output'] = \
                        self.data['post_initialization_traces']['output']
                self.store_traces(saved_traces, subfolder='{}_{}'.format(
                    self.subfolder, spin_state))

        optimal_idx = np.argmax(self.results)
        optimal_spin_state = self.spin_states[optimal_idx]

        frequency = self.frequencies[optimal_spin_state]
        self.results += [frequency]

        # Print results
        if not self.silent:
            self.print_results()

        if self.update_frequency and max(self.results) > self.threshold:
            properties_config['frequency' + self.mode_str] = frequency
        elif not self.silent:
            logging.warning("Could not find frequency with high enough "
                            "contrast")

        return self.results



class AutoCalibration_Parameter(Parameter):
    def __init__(self, name, set_parameters, measure_parameter,
                 calibration_operations, key, conditions=None, **kwargs):
        """

        Args:
            name:
            set_parameters:
            measure_parameter:
            calibration_operations:
            key:
            conditions: Must be of one of the following forms
                {'mode': 'measure'}
                {'mode': '1D_scan', 'span', 'set_points', 'set_parameter',
                 'center_val'(optional)
            **kwargs:
        """
        super().__init__(name, **kwargs)

        self.key = key
        self.conditions = conditions
        self.calibration_operations = calibration_operations

        self.set_parameters = {p.name: p for p in set_parameters}
        self.measure_parameter = measure_parameter

        self.names = ['success', 'optimal_set_val', self.key]
        self.labels = self.names
        self._meta_attrs.extend(['measure_parameter_name', 'conditions',
                                 'calibration_operations', 'key',
                                 'set_vals_1D', 'measure_vals_1D'])

    @property
    def measure_parameter_name(self):
        return self.measure_parameter.name

    def satisfies_conditions(self, dataset, dims):
        # Start of with all set points satisfying conditions
        satisfied_final_arr = np.ones(dims)
        if self.conditions is None:
            return satisfied_final_arr
        for (attribute, target_val, relation) in self.conditions:
            test_vals = getattr(dataset, attribute).ndarray
            # Determine which elements satisfy condition
            satisfied_arr = general_tools.get_truth(test_vals, target_val,
                                                    relation)

            # Update satisfied elements with the ones satisfying current
            # condition
            satisfied_final_arr = np.logical_and(satisfied_final_arr,
                                                 satisfied_arr)
        return satisfied_final_arr

    def optimal_val(self, dataset, satisfied_set_vals=None):
        measurement_vals = getattr(dataset, self.key)

        set_vals_1D = np.ravel(self.set_vals)
        measurement_vals_1D = np.ravel(measurement_vals)

        if satisfied_set_vals is not None:
            # Filter 1D arrays by those satisfying conditions
            satisfied_set_vals_1D = np.ravel(satisfied_set_vals)
            satisfied_idx = np.nonzero(satisfied_set_vals)[0]

            set_vals_1D = np.take(set_vals_1D, satisfied_idx)
            measurement_vals_1D = np.take(measurement_vals_1D, satisfied_idx)

        max_idx = np.argmax(measurement_vals_1D)
        return set_vals_1D[max_idx], measurement_vals_1D[max_idx]

    def get(self):
        self.loop_parameters = []
        self.datasets = []

        if hasattr(self.measure_parameter, 'setup'):
            self.measure_parameter.setup()

        for k, calibration_operation in enumerate(self.calibration_operations):
            if calibration_operation['mode'] == 'measure':
                dims = (1)
                self.set_vals = [None]
                loop_parameter = Loop0D_Parameter(
                    name='calibration_0D',
                    measure_parameter=self.measure_parameter)

            elif calibration_operation['mode'] == '1D_scan':
                # Setup set vals
                set_parameter_name = calibration_operation['set_parameter']
                set_parameter = self.set_parameters[set_parameter_name]

                center_val = calibration_operation.get(
                    'center_val', set_parameter())
                span = calibration_operation['span']
                set_points = calibration_operation['set_points']
                self.set_vals = list(np.linspace(center_val - span / 2,
                                                 center_val + span / 2,
                                                 set_points))
                dims = (set_points)
                # Extract set_parameter
                loop_parameter = Loop1D_Parameter(
                    name='calibration_1D',
                    set_parameter=set_parameter,
                    measure_parameter=self.measure_parameter,
                    set_vals=self.set_vals)
            else:
                raise ValueError("Calibration mode not implemented")

            self.loop_parameters.append(loop_parameter)
            dataset = loop_parameter()
            self.datasets.append(dataset)

            satisfied_set_vals = self.satisfies_conditions(dataset, dims)
            if np.any(satisfied_set_vals):
                optimal_set_val, optimal_get_val = self.optimal_val(
                    dataset, satisfied_set_vals)
                cal_success = k
                break
        else:
            logging.warning('Could not find calibration point satisfying '
                            'conditions. Choosing best alternative')
            optimal_set_val, optimal_get_val = self.optimal_val(dataset)
            cal_success = -1

        if optimal_set_val is not None:
            set_parameter(optimal_set_val)
            # TODO implement for 2D
            return cal_success, optimal_set_val, optimal_get_val
        else:
            return cal_success, 1, optimal_get_val
