import json
import os


class Facts(object):
    def __init__(self):
        self.FILE = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'facts.json'))
        self.read = self.open_file()
        self.facts = self.dotdict(self.read)

    def open_file(self):
        with open(self.FILE) as f:
            return json.load(f)

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        #__setattr__ = dict.__setitem__
        #__delattr__ = dict.__delitem__

if __name__ == "__main__":
    f= Facts()
    print(f.FILE)
    print(f.facts.pcl_corner_pts['ptx'])
    print(f.facts.crop_idx)
