import threading
t = threading.Thread(target=Slack, name='slack',
                     kwargs={'df_ds': DF_DS,
                            'tgac': TGAC,
                            'run': run_code})
t.start()