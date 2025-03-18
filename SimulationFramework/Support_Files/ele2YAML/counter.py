class Counter(dict):
    def __init__(self, sub={}):
        super().__init__()
        self.sub = sub

    def counter(self, type):
        type = self.sub[type] if type in self.sub else type
        if type not in self:
            return 1
        return self[type] + 1

    def value(self, type):
        type = self.sub[type] if type in self.sub else type
        if type not in self:
            return 1
        return self[type]

    def add(self, type, n=1):
        type = self.sub[type] if type in self.sub else type
        if type not in self:
            self[type] = n
        else:
            self[type] += n
        return self[type]
