from silq.meta_instruments.layout import Layout

### Layout and connectivity
layout = Layout(name='layout',
                instrument_interfaces=list(interfaces.values()),
                server_name='layout_server' if USE_MP else None)

layout.primary_instrument('pulseblaster')
layout.acquisition_instrument('ATS')

layout.load_connections()