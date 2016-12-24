import qcodes as qc
import numpy as np

properties_config = qc.config['user'].get('properties', {})


def create_set_vals(num_parameters=None, steps=None, step_vals=None, points=9,
                    set_parameters=None, silent=False):
    def calculate_step(min_step, max_step, percentage, round_vals=True):
        val = min_step * (max_step / min_step) ** (percentage / 100)
        if round_vals:
            # Round values to one decimal after first digit
            digits = int(np.ceil(np.log10(1 / val))) + 1
            val = round(val, digits)

        return val

    def determine_step(set_parameter, k=None):
        if steps is not None:
            if hasattr(steps, "__iter__"):
                step_percentage = steps[k]
            else:
                step_percentage = steps
            min_step, max_step = \
            properties_config['set_parameters'][set_parameter.name]['steps']
            step = calculate_step(min_step, max_step, step_percentage)
        elif step_vals is not None:
            if hasattr(steps, "__iter__"):
                step = step_vals[k]
            else:
                step = step_vals
        return step

    def create_vals(set_parameter, k=None):
        if hasattr(points, "__iter__"):
            pts = points[k]
        else:
            pts = points

        step = determine_step(set_parameter, k)
        center_val = set_parameter()
        min_val = center_val - step * (pts - 1) / 2
        max_val = center_val + step * (pts - 1) / 2
        vals = list(np.linspace(min_val, max_val, pts))

        if not silent:
            print('{param}[{min_val}:{max_val}:{step}]'.format(
                param=set_parameter, min_val=min_val, max_val=max_val,
                step=step))

        return vals

    if set_parameters is None:
        station = qc.station.Station.default
        set_parameters_names = \
            properties_config['set_parameters'][str(num_parameters)]
        set_parameters = [getattr(station, name) for name in
                          set_parameters_names]
    else:
        num_parameters = len(set_parameters)

    set_vals = []
    if num_parameters == 0:
        pass
    elif num_parameters == 1:
        vals = create_vals(set_parameters[0])
        set_vals = [set_parameters[0][vals]]
    elif num_parameters == 2:
        for k, set_parameter in enumerate(set_parameters):
            vals = create_vals(set_parameter, k)
            set_vals.append(set_parameter[vals])
    return set_vals
