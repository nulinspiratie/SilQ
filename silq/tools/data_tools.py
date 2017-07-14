import os
from dateutil.parser import parse
import logging

import qcodes as qc
import silq
from qcodes.data.data_set import new_data

__all__ = ['create_data_set', 'store_data', 'get_data_folder']

logger = logging.getLogger(__name__)


def create_data_set(name, base_folder, subfolder=None, formatter=None):
    location_string = '{base_folder}/'
    if subfolder is not None:
        location_string += '{subfolder}/'
    location_string += '#{{counter}}_{{name}}_{{time}}'

    location = qc.data.location.FormatLocation(
        fmt=location_string.format(base_folder=base_folder, name=name,
                                   subfolder=subfolder))

    data_set = new_data(location=location,
                        name=name,
                        formatter=formatter)
    return data_set


def store_data(dataset, result):
    dataset.store(loop_indices=slice(0, result.shape[0], 1),
                  ids_values={'data_vals': result})


def get_data_folder(path_str='', subfolder=None, newest_date=None):
    if isinstance(newest_date, str):
        newest_date = parse(newest_date)


    path_str = path_str.replace('\\', '/')

    base_path = silq.config.properties.data_folder
    base_path = base_path.replace('\\', '/')

    logger.debug(f'Dirs: {os.listdir(base_path)}')

    if '/' not in path_str:
        logging.debug('No date provided, getting latest date folder')
        date_folders = reversed(os.listdir(base_path))
        for date_folder in date_folders:
            try:
                date = parse(date_folder)
                if newest_date is not None and date > newest_date:
                    continue
                else:
                    # date_folder has date format, exit loop
                    logging.debug(f'Date folder: {date_folder}')
                    break
            except ValueError:
                # date_folder does not have date format, continuing to next
                continue
        else:
            raise RuntimeError(f'No date folder found in {base_path}')
            # TODO include previous dates

        # No date provided, so any data folder must match path_str
        data_str = path_str
    elif ':' in path_str:
        raise NotImplementedError('Full paths not yet implemented')
    else:
        # relative path, likely of form {date}/{data_folder}
        date_folder, data_str = path_str.split('/')[-2:]

    if data_str[:1] == '#' and data_str[1:].isdigit():
        data_str = f'#{int(data_str[1:]):03}'

    date_path = os.path.join(base_path, date_folder)

    data_folders = reversed(os.listdir(date_path))
    data_folder = next(folder for folder in data_folders if data_str in folder)
    data_path = os.path.join(date_path, data_folder)

    if subfolder is not None:
        if subfolder[:1] == '#' and subfolder[1:].isdigit():
            subfolder = f'#{int(subfolder[1:]):03}'

        data_subfolders = reversed(os.listdir(data_path))
        data_subfolder = next(folder for folder in data_subfolders
                              if subfolder in folder)
        logger.debug(f'Subfolder: {data_subfolder}')
        data_folder = os.path.join(data_folder, data_subfolder)

    data_relative_path = os.path.join(date_folder, data_folder)
    data_relative_path.replace('\\', '/')

    return data_relative_path
