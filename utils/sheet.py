class sheet:

    def __init__(self, column, entropy, probability):
        super().__init__()
        self.column = column
        self.sons = dict()
        self.entropy = entropy
        self.probability = probability

    def add_son(self, son, condition):
        self.sons[condition] = son

    def creat_add_son(self, entropy, probability, condition):
        son = sheet(entropy, probability)
        self.add_son(son, condition)

    def get_son(self, condition):
        return self.sons.get(condition)

    def remove_son(self, condition):
        return self.sons.pop(condition)