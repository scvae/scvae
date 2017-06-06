import os
import sys
import time

import re

from math import floor

import urllib.request

# Time

def formatTime(t):
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t))

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
    
    s = s.lower()
    
    replacements = {
        "_": [" ", "-"],
        "": ["(", ")", "$"]
    }
    
    for replacement, characters in replacements.items():
        pattern = r"[" + "".join(characters) + "]"
        s = re.sub(pattern, replacement, s)
    
    return s

def properString(s, translation):
    
    s = normaliseString(s)
    
    for proper_string, normalised_strings in translation.items():
        if s in normalised_strings:
            return proper_string

# IO

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

# Shell output

RESETFORMAT = "\033[0m"
BOLD = "\033[1m"

def bold(string):
    """Convert to bold type."""
    return BOLD + string + RESETFORMAT

def underline(string, character="="):
    """Convert string to header marks"""
    return character * len(string)

def title(string):
    """Display a bold title."""
    print("{}\n{}\n".format(bold(string), underline(string, "≡")))

def subtitle(string):
    """Display a bold subtitle."""
    print("{}\n{}\n".format(bold(string), underline(string, "=")))

def heading(string):
    """Display a bold heading."""
    print("{}\n{}\n".format(bold(string), underline(string, "-")))
