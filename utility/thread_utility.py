def start_thread(thread_list: list) -> None:
    """ 
    Start threads
    :param thread_list: list of threads
    """
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
