import os
import datetime

def exist_path(*args):
    path = os.path.join(*args)

    if(os.path.exists(path)):
        pass
    else:
        os.mkdir(path)
    return path

def date_path():
    now = datetime.datetime.now()
    return f"{now.year}-{now.month:02d}-{now.day:02d}"

def now_file_name(file_ext:str="")->str:
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    return f"{formatted_time}.{file_ext}"

if __name__ == "__main__":
    # print(date_path())
    pass