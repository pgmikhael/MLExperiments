from collections import defaultdict

class Ddict(defaultdict, dict):
    def __init__(self):
        defaultdict.__init__(self, Ddict)

    def __repr__(self):
        return dict.__repr__(self)
