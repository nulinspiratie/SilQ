
class MockVisaHandle:
    '''
    mock the API needed for a visa handle that throws lots of errors:

    - any write command sets a single "state" variable to a float
      after the last : in the command
    - a negative number results in an error raised here
    - 0 results in a return code for visa timeout
    - any ask command returns the state
    - a state > 10 throws an error
    '''
    def __init__(self):
        self.state = 0

    def clear(self):
        self.state = 0

    def close(self):
        # make it an error to ask or write after close
        self.write = None
        self.ask = None

    def write(self, cmd):
        num = float(cmd.split(':')[-1])
        self.state = num

        if num < 0:
            raise ValueError('be more positive!')

        if num == 0:
            ret_code = visa.constants.VI_ERROR_TMO
        else:
            ret_code = 0

        return len(cmd), ret_code

    def ask(self, cmd):
        if self.state > 10:
            raise ValueError("I'm out of fingers")
        return self.state