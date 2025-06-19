import random
import numpy as np
from picure.py as *

class spgd(object):

    def __init__(self,T,cool,sigma,target,limit):
        self.start_T = T

    def T_cool(self):
        self.T *= self.cool

    