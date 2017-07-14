from IPython.display import display_javascript

__all__ = ['create_cell']


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
