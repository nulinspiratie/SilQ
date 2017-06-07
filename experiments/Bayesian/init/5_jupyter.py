from dateutil.parser import parse
from IPython.core.magic import Magics, magics_class, line_magic
import logging
from silq import config
logger = logging.getLogger()


@magics_class
class CustomMagics(Magics):
    """Magics related to code management (loading, saving, editing, ...)."""

    def __init__(self, *args, **kwargs):
        self._knowntemps = set()
        super(CustomMagics, self).__init__(*args, **kwargs)

    @line_magic
    def EPR(self, line):
        EPR_parameter.setup(start=True)
        print(f'Succesfully started EPR pulse sequence')

    @line_magic
    def stop(self, line):
        layout.stop()
        print(f'Succesfully stopped Layout')

    @line_magic
    def data(self, line):
        line = line.replace('\\', '/')

        base_path = config.properties.data_folder
        base_path = base_path.replace('\\', '/')
        # base_folder = os.path.split(base_path)[1]

        logging.debug(f'Dirs: {os.listdir(base_path)}')

        if '/' not in line:
            logging.debug('No date provided, getting latest date folder')
            date_folders = reversed(os.listdir(base_path))
            for date_folder in date_folders:
                try:
                    parse(date_folder)
                    logging.debug(f'Date folder: {date_folder}')
                    # date_folder has date format, exit loop
                    break
                except:
                    # date_folder does not have date format, continuing to next
                    continue
            else:
                raise RuntimeError(f'No date folder found in {base_path}')
            # No date provided, so any data folder must match line
            data_str = line
        elif ':' in line:
            raise NotImplementedError('Full paths not yet implemented')
        else:
            date_folder, data_str = line.split('/')[-2:]
        date_path = os.path.join(base_path, date_folder)

        data_folders = reversed(os.listdir(date_path))
        data_folder = next(
            folder for folder in data_folders if data_str in folder)
        data_path = os.path.join(date_folder, data_folder)
        logging.debug(f'Data path: {data_path}')

        # Update cell
        contents = f"data = load_data(r'{data_path}')"
        self.shell.set_next_input(contents, replace=True)
        self.shell.run_cell(contents, store_history=False)


ip = get_ipython()
mm = ip.magics_manager
mm.register(CustomMagics)