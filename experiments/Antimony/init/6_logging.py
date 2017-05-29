###############
### Logging ###
###############

note_formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(
    os.path.join(config.properties.data_folder, r'notes\notes.log'))
file_handler = logging.handlers.TimedRotatingFileHandler(
    os.path.join(config.properties.data_folder, r'notes\notes.log'), when='midnight')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(note_formatter)

note_logger = logging.getLogger('notes')
note_logger.setLevel(logging.INFO)

note_logger.addHandler(file_handler)
note_logger.addFilter(logging.Filter(name='notes'))