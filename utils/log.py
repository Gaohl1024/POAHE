import datetime
import os


def logger(log_root, log_name, text, is_print=True):
    os.makedirs(log_root, exist_ok=True)
    log_dir = log_root + log_name
    current_time = datetime.datetime.now().strftime("[%Y/%m/%d %H:%M:%S] ")
    text = '\n' + current_time + text + '\n'
    if is_print:
        print(text)
    with open(log_dir, 'a') as file:
        file.write(text)
