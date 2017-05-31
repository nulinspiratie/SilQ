import threading
t = threading.Thread(target=Slack, name='slack',
                     kwargs={'auto_start': True,
                             'df_ds': DF_DS,
                            'tgac': TGAC,
                            'run': run_code})
t.start()