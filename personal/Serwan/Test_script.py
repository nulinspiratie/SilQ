

if __name__ == "__main__":
    USE_MP = False
    import silq

    silq.initialize("EWJN")


    # Calculate T1 durations
    T1_wait_times = list(np.logspace(1, 3, num=10, base=10))

    # Shuffle times
    # np.random.shuffle(T1_wait_times)
    print('Shuffled T1 wait times: {}'.format(T1_wait_times))

    # Single T1 sweep

    T1_parameter.label = 'T1_wait_time'
    data = qc.Loop(
        T1_parameter[T1_wait_times]).each(
        T1_parameter).run(name='T1_single_sweep', background=False)
    sleep(1000)