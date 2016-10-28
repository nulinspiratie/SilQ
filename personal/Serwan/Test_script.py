
if __name__ == "__main__":
    import silq

    silq.initialize("EWJN")

    gate1 = TGAC
    gate2 = DF_DS

    gate1_vals = list(np.linspace(0.24, 0.26, 4))
    gate2_vals = list(np.linspace(0.45, 0.5, 15))

    DC_parameter.setup()
    data = qc.Loop(gate1[gate1_vals]
                   ).loop(gate2[gate2_vals]
                          ).each(TG, DC_parameter
                                 ).then(qc.Task(layout.stop)
                                        ).run(
        name='DC_{}_vs_{}_scan'.format(gate1.name, gate2.name),
        progress_interval=True)

    plot = qc.MatPlot()
    plot.add(data.DC_voltage)