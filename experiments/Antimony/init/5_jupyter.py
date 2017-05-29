@register_line_magic
def EPR(line):
    EPR_parameter.setup(start=True)
    print(f'Succesfully started EPR pulse sequence')

@register_line_magic
def stop(line):
    layout.stop()
    print(f'Succesfully stopped Layout')

@register_line_magic
def note(line):
    note_logger.info(line)

# @register_line_magic
# def data(line):
