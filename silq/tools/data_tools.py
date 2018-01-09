import os
from dateutil.parser import parse
import logging

import qcodes as qc
from qcodes.data.format import Formatter
import silq
from qcodes.data.data_set import new_data

__all__ = ['create_data_set', 'get_data_folder']

logger = logging.getLogger(__name__)


def create_data_set(name: str,
                    base_folder: str,
                    subfolder: str = None,
                    formatter: Formatter = None):
    """Create empty ``DataSet`` within main data folder.

    Uses ``new_data``, and handles location formatting.

    Args:
        name: ``DataSet`` name, used as DataSet folder name.
        base_folder: Base folder for DataSet. Should be a pre-existing
            ``DataSet`` folder. If None, a new folder is created in main data
            folder.
        subfolder: Adds subfolder within base_folder for ``DataSet``.
            Should not be used without explicitly setting ``base_folder``.
        formatter: Formatter to use for data storage (e.g. GNUPlotFormat).

    Returns:
        New empty ``DataSet``.
    """
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


def get_data_folder(*path: str,
                    newest_date: str = None):
    """Get first data folder in main data folder satisfying conditions.

    Args:
        *path: Filter for data folder.
            First arg can be an absolute path, in which case data folder is
            searched in that path.
            If a list of strings, each element corresponds to a subfolder in
            main data folder, whose folder name must contain the given string.
            Arg can be dataset index, in which case it must start with #, and be
            followed by digits. leading zeroes are not necessary.
            If not provided, first data folder in main data path is used.
        newest_date: Latest date for dataset. If specified, the first dataset
            earlier than this date is searched.

    Returns:
        Relative path to found dataset

    Raises:
        IterationError: No dataset found.
    """

    # Ensure that all path items use '/'
    path = [p.replace('\\', '/') for p in path]

    # Determine if a specific date and/or data filter is required
    if not path:
        logger.debug('No path args provided, any data folder will suffice')
        date_folder = None
        path.append('')
    elif ':' in path[0]:
        return path[0]
    elif '/' in path[0]:
        date_folder, path[0] = path[0].split('/')
        logger.debug(f'Date {date_folder} and filter {path[0]} provided')
    else:
        date_folder = None
        logger.debug(f'Data filter {path[0]} provided, but no date')

    base_path = silq.config.properties.data_folder
    base_path = base_path.replace('\\', '/')
    if date_folder is None:
        # Find latest date folder
        date_folders = reversed(os.listdir(base_path))
        # Cycle through date folders
        for date_folder in date_folders:
            # Check if folder is actual date folder containing data
            try:
                date = parse(date_folder)
            except ValueError:
                # not actual date folder
                continue

            if newest_date is not None and date > newest_date:
                continue
            else:
                break
        else:
            raise RuntimeError(f'No matching date folder found in {base_path}')

    # Iteratively go through path items and add them to the data path
    relative_path = date_folder
    for subpath in path:
        if subpath[:1] == '#' and subpath[1:].isdigit():
            # prepend zeros if necessary
            subpath = f'#{int(subpath[1:]):03}'

        data_path = os.path.join(base_path, relative_path)
        data_folder = next(folder for folder in os.listdir(data_path)[::-1]
                           if subpath in folder)
        relative_path = os.path.join(relative_path, data_folder)

    return relative_path
