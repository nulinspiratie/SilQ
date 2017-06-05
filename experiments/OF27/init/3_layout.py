from silq.meta_instruments.layout import Layout

### Layout and connectivity
layout = Layout(name='layout',
                instrument_interfaces=list(interfaces.values()))

layout.primary_instrument('pulseblaster')
layout.acquisition_instrument('ATS')

layout.load_connections()