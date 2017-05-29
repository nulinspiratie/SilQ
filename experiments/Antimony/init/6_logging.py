###############
### Logging ###
###############
from silq.tools.general_tools import ParallelTimedRotatingFileHandler
note_formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

# file_handler = logging.FileHandler(
#     os.path.join(config.properties.data_folder, r'notes\notes.log'))
# filename = os.path.join(config.properties.data_folder, r'notes\notes')
filename = r'F:\Antimony\data\notes\notes'
file_handler = ParallelTimedRotatingFileHandler(filename=filename,
                                                when='midnight')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(note_formatter)

note_logger = logging.getLogger('notes')
note_logger.setLevel(logging.INFO)

note_logger.addHandler(file_handler)
note_logger.addFilter(logging.Filter(name='notes'))