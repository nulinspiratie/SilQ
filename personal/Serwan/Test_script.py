
from functools import partial
import silq
import qcodes as qc


if __name__ == "__main__":
    from silq.tests.test_instruments import TestInnerInstrument, \
        TestOuterInstrument

    inner_instrument = TestInnerInstrument('inner')
    outer_instrument = TestOuterInstrument('outer',
                                           instruments=[inner_instrument],
                                           server_name='outer')

    outer_instrument.get_instruments()