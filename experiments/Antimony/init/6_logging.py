###############
### Logging ###
###############
from silq.tools.general_tools import ParallelTimedRotatingFileHandler

###################
### Note logger ###
###################
note_formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

filename = r'F:\Antimony\data\notes\notes'
file_handler = ParallelTimedRotatingFileHandler(filename=filename,
                                                when='midnight')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(note_formatter)

note_logger = logging.getLogger('notes')
note_logger.setLevel(logging.INFO)

note_logger.addHandler(file_handler)
note_logger.addFilter(logging.Filter(name='notes'))


##############
### Logger ###
##############

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - '
                                  '%(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

filename = r'F:\Antimony\data\log\log'
file_handler = ParallelTimedRotatingFileHandler(filename=filename,
                                                when='midnight')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if logger.handlers:
    logger.handlers[0].setLevel(logging.INFO)
logger.addHandler(file_handler)

class logFilter:
    def filter(self, record):
        self.record = record
        if record.name == 'requests.packages.urllib3.connectionpool':
            # Filter these out, emitted by Slacker after every request
            return False
        else:
            return True

file_handler.addFilter(logFilter())
logger.addFilter(logFilter())