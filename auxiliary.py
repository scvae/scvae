from math import floor

# Time

def formatDuration(seconds):
    if seconds < 0.001:
        return "<1 ms"
    elif seconds < 1:
        return "{:.0f} ms".format(1000 * seconds)
    elif seconds < 60:
        return "{:.3g} s".format(seconds)
    elif seconds < 60 * 60:
        minutes = floor(seconds / 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        return "{:.0f}m {:.0f}s".format(minutes, seconds)
    else:
        hours = floor(seconds / 60 / 60)
        minutes = floor((seconds / 60) % 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            hours += 1
        return "{:.0f}h {:.0f}m {:.0f}s".format(hours, minutes, seconds)

# Strings

def normaliseString(s):
    return s.lower().replace(" ", "_")

# IO

def directory(base_directory, data_set_name, preprocessing_methods = None):
    
    data_set_name = normaliseString(data_set_name)
    directory = os.path.join(base_directory, data_set_name)
    
    if preprocessing_methods:
        preprocessing_methods = map(normaliseString, preprocessing_methods)
        preprocessing_methods = "-".join(preprocessing_methods)
        directory = os.path.join(directory, preprocessing_methods)
    
    return directory

def download(URL, path):
    urllib.request.urlretrieve(URL, path, download_report_hook)

def download_report_hook(block_num, block_size, total_size):
    bytes_read = block_num * block_size
    if total_size > 0:
        percent = bytes_read / total_size * 100
        sys.stderr.write("\r{:3.0f}%.".format(percent))
        if bytes_read >= total_size:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("\r{:d} bytes.".format(bytes_read))
