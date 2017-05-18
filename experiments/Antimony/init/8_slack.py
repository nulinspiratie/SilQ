import threading
threading.Thread(target=Slack, name='slack')


slack.commands['df_ds'] = DF_DS
slack.commands['tgac'] = TGAC
slack.commands['run'] = run_code
slack.commands['halt'] = qc.halt_bg