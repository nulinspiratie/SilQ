from qcodes.instrument.visa import VisaInstrument



class MockVisaHandle:
    '''
    mock the API needed for a visa handle that throws lots of errors:
    '''
    def __init__(self):
        self.states = {}

    def clear(self):
        self.states = {}

    def close(self):
        # make it an error to ask or write after close
        self.write = None
        self.ask = None

    def write(self, cmd):
        prefix, num = cmd.split(' ')
        num = float(num)
        self.states[prefix] = num

        ret_code = 0
        return len(cmd), ret_code

    def ask(self, cmd):
        prefix = cmd[:-1]
        if prefix not in self.states:
            # print(f'{prefix} not found in {self.states.keys()}')
            num = 0
        else:
            num = self.states[prefix]
        return num



def set_address_mock(self, address):
    self._address = address
    self.visa_handle = MockVisaHandle()



VisaInstrument.set_address = set_address_mock