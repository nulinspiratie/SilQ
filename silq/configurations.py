'''
This dictionary contains a list of initialization files.
An initialization file can be loaded using silq.initialize(init_name),
after which everything in the file will be loaded into the global namespace.
Each setting has a name, which can be given to silq.initialize.
The filename corresponds to the init file to be run, from bsae folder SilQ.
Additionally, a list of mac addresses can be added to each setting.
Adding a mac address to a setting will result in that settings file to be
instantiated upon calling intialize without specifying init_name
'''
_configurations = {
    'Serwan': {'folder': r"personal\Serwan",
               'macs': [113780617482],
               'modes': {
               }},
    'EWJN': {'folder': r"experiments\EWJN",
             'macs': [140870679927045],
             'modes': {
                 'analysis': {'select': ['imports', 'functions', 'data',
                                         'analysis', 'plotting']}
             }},
    'OF27': {'folder': r"experiments\OF27",
             'macs': [],
             'modes': {
                 'analysis': {'select': ['imports', 'functions', 'data',
                                         'analysis', 'plotting']}
             }},
    'Steffi': {'folder': r"personal\Steffi",
               'macs': [140870679927045],
               'modes': {
                 'analysis': {'select': ['imports', 'functions', 'data',
                                         'analysis', 'parameters', 'plotting']}
             }},
    'Antimony': {'folder': r"experiments\Antimony",
                 'macs': [30902517727200],
                 'modes': {
                     'analysis': {'select': ['imports', 'functions', 'data']}
                 }},
    'Bayesian': {'folder': r"experiments\Bayesian",
                 'macs': [],
                 'modes': {}
                 }
}