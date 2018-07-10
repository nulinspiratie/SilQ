========================
Clean up notebook saving
========================

When saving a notebook, a lot of extra code is generated resulting from
interactive matplotlib plots. Every interactive figure saves ~600 lines of
unnecessary code. The following snippet strips off any matplotlib output
(excluding figures). Furthermore, Additionally, outputs consisting of
matplotlib or MatPlot objects are also stripped (these may arise if the last
function in a cell returns the figure object).

Place the following snippet in the jupyter notebook config file (~/
.jupyter/jupyter_notebook_config.py). The best to place it is as a
replacement of the pre-existing line `c.ContentsManager.pre_save_hook`

.. code-block::
    def scrub_output_pre_save(model, **kwargs):
        """scrub output before saving notebooks"""
        # only run on notebooks
        print('scrubbing output before saving')
        if model['type'] != 'notebook':
            return
        # only run on nbformat v4
        if model['content']['nbformat'] != 4:
            return

        for cell in model['content']['cells']:
            if cell['cell_type'] != 'code':
                continue

            # Remove
            filtered_outputs = []
            for output in cell['outputs']:
                javascript_output = output.get('data', {}).get('application/javascript', '')
                plain_output = output.get('data', {}).get('text/plain', '')
                if 'Put everything inside the global mpl namespace' in javascript_output:
                    continue
                elif plain_output.startswith('<qcodes.plots.qcmatplotlib.MatPlot'):
                    continue
                elif plain_output.startswith('[<matplotlib.'):
                    continue
                else:
                    filtered_outputs.append(output)
            cell['outputs'] = filtered_outputs
            cell['execution_count'] = None

    c.ContentsManager.pre_save_hook = scrub_output_pre_save