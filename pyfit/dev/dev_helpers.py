import os

def init_directories(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)