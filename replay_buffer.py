import numpy as np
import random

class replay_buffer():
    def __init__(self):
        self.main_bugger = []
        self.mini_bugger = []

    def discount(self, gamma):
        for i,rb in enumerate(self.mini_bugger):
            for j,drb in enumerate(self.mini_bugger[i:]):
                rb[3] += drb[3]*gamma**(j) if not j==0 else 0
                if drb[4]:
                    break
        self.main_bugger += self.mini_bugger
        self.mini_bugger = []

    def v0_reward(self):
        for i,rb in enumerate(self.mini_bugger):
            r0 = rb[3]
            for j,drb in enumerate(self.mini_bugger[(i):]):
                rb[3] += drb[3] if not j==0 else 0
                drb[3] += r0 if j==1 else 0
                if drb[4]:
                    break
        return disrb

    def main_buffer(self):
        return self.main_bugger

    def len_buffer(self):
        return len(self.main_bugger)

    def append_buffer(self, buff):
        self.mini_bugger.append(buff)

    def shuffle(self):
        random.shuffle(self.main_bugger)

    def reset_mini(self):
        self.mini_bugger = []
