import qcodes as qc
import numpy as np
parameter_config = qc.config['user']['properties'].get('parameters', {})

def create_set_vals(num_parameters=None, step=None, step_percentage=None,
                    points=None, window=None, set_parameters=None, silent=True,
                    center_val=None, min_val=None, max_val=None, reverse=False):
    def calculate_step(min_step, max_step, percentage, round_vals=True):
        val = min_step * (max_step / min_step) ** (percentage / 100)
        if round_vals:
            # Round values to one decimal after first digit
            digits = int(np.ceil(np.log10(1 / val))) + 1
            val = round(val, digits)

        return val

    def determine_step(set_parameter):
        if step_percentage is not None:
            min_step, max_step = parameter_config[set_parameter.name]['steps']
            return calculate_step(min_step, max_step, step_percentage)
        else:
            return step

    def create_vals(set_parameter, points=None, window=None,
                    center_val=None, min_val=None, max_val=None):
        if window is None:
            if min_val is not None and max_val is not None:
                window = max_val - min_val
            else:
                step = determine_step(set_parameter)
                window = step * (points - 1)
        if points is None:
            step = determine_step(set_parameter)
            points = int(round(window / step))

        if min_val is None and max_val is None:
            if center_val is None:
                center_val = set_parameter()
            min_val = center_val - window / 2
            max_val = center_val + window / 2

        vals = list(np.linspace(min_val, max_val, points))
        if reverse:
            vals.reverse()

        if not silent:
            step = window / points
            print('{param}[{min_val}:{max_val}:{step}]'.format(
                param=set_parameter, min_val=min_val, max_val=max_val,
                step=step))

        return vals

    if set_parameters is None:
        # Get default parameters from station
        # the parameters to use depend on num_parameters
        station = qc.station.Station.default
        set_parameter_names = parameter_config[str(num_parameters)]
        set_parameters = [getattr(station, name) for name in
                          set_parameter_names]

    set_vals = []
    if isinstance(set_parameters, list):
        for k, set_parameter in enumerate(set_parameters):
            vals = create_vals(set_parameter, points=points, window=window,
                               center_val=center_val, min_val=min_val,
                               max_val=max_val)
            set_vals.append(set_parameter[vals])
    else:
        # Set_parameters is a single parameter
        vals = create_vals(set_parameters, points=points, window=window,
                           center_val=center_val, min_val=min_val,
                           max_val=max_val)
        set_vals = set_parameters[vals]
    return set_vals
