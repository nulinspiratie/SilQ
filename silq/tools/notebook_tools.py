from IPython.display import display_javascript
from IPython.core.magic import Magics, magics_class, line_magic, line_cell_magic
import logging

from .data_tools import get_data_folder
import qcodes as qc


__all__ = ['create_cell']

logger = logging.getLogger()


def create_cell(text, location='below', execute=False, select=True):
    text = text.replace('\n', '\\n')
    text = text.replace("'", '"')

    code = ''
    # Create cell at specified location
    if location == 'below':
        code += """
        var current_cell = Jupyter.notebook.get_selected_cell();
        var current_index = Jupyter.notebook.find_cell_index(current_cell);
        var new_cell = Jupyter.notebook.insert_cell_below('code', current_index);
        """
    elif location == 'bottom':
        code += """
        var current_cell = Jupyter.notebook.get_selected_cell();
        var new_cell = Jupyter.notebook.insert_cell_at_bottom('code');
        """
    else:
        raise Exception('Location {} not understood'.format(location))

    # Add text to cell
    code += "new_cell.set_text('{}');".format(text);

    # Select cell
    if select:
        code += """
        current_cell.unselect();
        new_cell.select();"""

    # Execute cell
    if execute:
        code += "new_cell.execute();"

    # Run code
    display_javascript(code, raw=True)


# Used in %data magic to add additional code when loading dataset
data_handlers = {}


@magics_class
class SilQMagics(Magics):
    """Magics related to code management (loading, saving, editing, ...)."""

    def __init__(self, *args, **kwargs):
        self._knowntemps = set()
        super(SilQMagics, self).__init__(*args, **kwargs)

    @line_magic
    def stop(self, _):
        layout = qc.Instrument.find_instrument('layout')
        layout.stop()
        print(f'Succesfully stopped Layout')

    @line_cell_magic
    def data(self, line, cell=None):
        opts, args = self.parse_options(line, 'xprs:')
        logger.debug(f'opts: {opts}, args: {args}')

        # Include subfolder if provided by -s
        data_path = get_data_folder(*args.split(' '))

        # Create contents
        contents = f"data = load_data(r'{data_path}')"
        if 'p' in opts:
            contents += '\nprint(data)'
        if 'r' not in opts and cell is None:
            data_folder = data_path.rsplit('/')[-1]
            logger.debug(f'data folder is {data_folder}')
            for handler, handle_str in data_handlers.items():
                if handler in data_folder:
                    contents += f'\n{handle_str}'
                    break

        if cell is not None:
            contents += f'\n{cell}'

        # Update cell
        self.shell.set_next_input(contents, replace=True)
        if 'x' in opts:
            self.shell.run_cell(contents, store_history=False)
