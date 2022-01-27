import numpy as np
import logging
from typing import Union, List, Dict, Any, Tuple, Sequence

from silq.instrument_interfaces import InstrumentInterface, Channel
from silq.meta_instruments.layout import SingleConnection
from silq.pulses import Pulse, DCPulse, DCRampPulse, TriggerPulse, SinePulse, \
    MultiSinePulse, SingleWaveformPulse, FrequencyRampPulse, \
    PulseImplementation, MarkerPulse

from qcodes.utils.helpers import arreqclose_in_list
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.parameter_node import parameter

logger = logging.getLogger(__name__)


class OPXInterface(InstrumentInterface):
    """ Interface for the Quantum Machines OPX.


        Args:
            instrument_name: name of instrument for which this is an interface
        """

    def __init__(self,
                 instrument_name: str,
                 **kwargs):
        super().__init__(name=instrument_name + '_interface', **kwargs)
        # OPX does not have a QCodes instrument.
        # self.instrument = self.find_instrument(name=instrument_name)

        self._input_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'input_{k}', id=k, output=False)
            for k in [1, 2]
        }
        self._output_channels = {
            f'ch{k}': Channel(instrument_name=self.instrument_name(),
                              name=f'output_{k}', id=k, output=True)
            for k in range(1, 10 + 1)
        }

        self._marker_channels = {
            f'marker_{k}': Channel(instrument_name=self.instrument_name(),
                                   name=f'marker_{k}', id=k, output_TTL=(0, 3.3))
            for k in range(1, 10 + 1)
        }

        self._channels = {
            **self._input_channels,
            **self._output_channels,
            **self._marker_channels,
            f'ext_trigger': Channel(instrument_name=self.instrument_name(),
                                    name=f'ext_trigger', input_trigger=True)
        }

        self.pulse_implementations = []

    def get_additional_pulses(self, connections) -> List[Pulse]:
        """Additional pulses needed by instrument after targeting of main pulses

        Args:
            connections: List of all connections in the layout

        Returns:
            List of additional pulses, empty by default.
        """
        return []

    def setup(self,
              samples: Union[int, None] = None,
              input_connections: list = [],
              output_connections: list = [],
              repeat: bool = True,
              **kwargs) -> Dict[str, Any]:
        """Set up instrument after layout has been targeted by pulse sequence.

        Needs to be implemented in subclass.

        Args:
            samples: Number of acquisition samples.
                If None, it will use the previously set value.
            input_connections: Input :class:`.Connection` list of
                instrument, needed by some interfaces to setup the instrument.
            output_connections: Output :class:`.Connection` list of
                instrument, needed by some interfaces to setup the instrument.
            repeat: Repeat the pulse sequence indefinitely. If False, calling
                :func:`Layout.start` will only run the pulse sequence once.
            **kwargs: Additional interface-specific kwarg.

        Returns:
            setup flags (see :attr:`.Layout.flags`)

        """

        assert self.is_primary(), \
            'OPX is currently only programmed to function as primary instrument'



    def start(self):
        """Start instrument

        Note:
            Acquisition instruments usually don't need to be started
        """
        raise NotImplementedError(
            'InstrumentInterface.start should be implemented in a subclass')

    def stop(self):
        """Stop instrument"""
        raise NotImplementedError(
            'InstrumentInterface.stop should be implemented in a subclass')
