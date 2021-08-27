from .facts import facts

class Facts(object):
    def __init__(self):
        self.dict = facts
        self.facts = self.dotdict(self.dict)

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get

