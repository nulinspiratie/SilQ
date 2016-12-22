from IPython.display import display_javascript


def cell_create_below_execute(text):
    """
    Creates a cell below the current one in a notebook, adds text and executes
    Args:
        text: Code to be run in the cell

    Returns:
        None
    """
    text = text.replace('\n', '\\n')
    display_javascript("""
    var current_cell = Jupyter.notebook.get_selected_cell();
    var current_index = Jupyter.notebook.find_cell_index(current_cell);
    var new_cell = Jupyter.notebook.insert_cell_below('code', current_index);
    new_cell.set_text('{}');
    current_cell.unselect();
    new_cell.execute();""".format(text), raw=True)

def cell_create_below_select(text):
    """
    Creates a cell below the current one in a notebook, adds text and executes
    Args:
        text: Code to be run in the cell

    Returns:
        None
    """
    text = text.replace('\n', '\\n')
    display_javascript("""
    var current_cell = Jupyter.notebook.get_selected_cell();
    var current_index = Jupyter.notebook.find_cell_index(current_cell);
    var new_cell = Jupyter.notebook.insert_cell_below('code', current_index);
    new_cell.set_text('{}');
    current_cell.unselect();
    new_cell.select();""".format(text), raw=True)
