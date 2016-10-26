def get_instrument_class(instrument):
    """
    Obtain the class name of the instrument.
    Args:
        instrument: Instrument from which to obtain the class

    Returns:
        instrument_class: Str representation of instrument class
    """
    instrument_class = instrument.__class__.__name__
    # Must use different approach if instrument is remote
    if instrument_class == 'RemoteInstrument':
        instrument_class = instrument._instrument_class.__name__
    return instrument_class