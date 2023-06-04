import time

def time_now(asc=False):
    right_now = time.time()

    if asc:
        output = time.asctime(time.localtime(right_now))
    else:
        output = right_now

    return output


def track_time(time_start, second=True):
    time_elapsed = time_now(asc=False) - time_start
    hrs = str(int(time_elapsed//3600)).zfill(2)
    mnt = str(int(time_elapsed%3600//60)).zfill(2)
    scn = str(int(time_elapsed%3600%60)).zfill(2)
    
    if second:
        output = f"{hrs}:{mnt}:{scn}"
    else:
        output = f"{hrs}:{mnt}"

    return output
