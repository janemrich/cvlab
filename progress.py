import sys

class Bar():

    def __init__(self, title, finish=100, width=50):
        self.title = title
        self.progress = 0
        self.finish = finish
        self.width = width
        self.print()

    def print(self):
        print("{}: [{}{}] {} / {}".format(
            self.title,
            "#" * int(self.progress/self.finish * self.width),
            "-" * int((self.finish - self.progress)/self.finish * self.width),
            self.progress,
            self.finish
        ), end='\r')

    def inc_progress(self, x):
        self.progress += x
        self.print()

    def set_progress(self, x):
        self.progress = x
        self.print()