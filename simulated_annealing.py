import random
import numpy as np

class SA(object):

    def __init__(self,T,cool,sigma,target,limit):
        self.start_T = T
        self.cool = cool
        self.start_target = target
        self.goal = 10
        self.start_sigma = sigma
        self.limit = limit

        self.reset()
        self.target = [round(2*self.limit*random.random() - self.limit , 2) for _ in range(len(self.target))]

    def T_cool(self):
        self.T *= self.cool

    def action(self, i=1, rand=True):
        if rand:
          if self.sigma >= 0.5:
            a = self.sigma
          else:
            a = 0.5
          rate = round(a,3)
          action = self.find_numbers_with_norm(rate,len(self.target))
          action = [round(a,3) for a in action]
          self.sigma *= 0.995
        else:
          if i == 0:
              action = [round(1000*random.random() - 500 , 1) for _ in range(len(self.target))]
          rate = 250*(1-(self.LN_2d.laplacian_blur(1)/self.blur_max))
          action = self.find_numbers_with_norm(rate,len(self.target))

        check = [a + b for a,b in zip(self.target, action)]
        # print('check', check)
        for j in range(len(check)):
          if (abs(check[j]))>=self.limit:
            check[j] = round(2*self.limit*random.random() - self.limit , 3)

        action = [a-b for a,b in zip(check, self.target)]
        action = [round(a,3) for a in action]
        # print('rate',rate,'action',action)
        self.target = [round(a + b,3) for a,b in zip(self.target, action)]
        before = [a - b for a,b in zip(self.target, action)]
        return action, before

    def step(self, current, next):
        if next > current:
            return False, next

        elif np.random.rand() < np.exp((next - current)/self.T):
            return False, next

        else:
            return True, current


    def reset(self):
        self.target = self.start_target
        self.T = self.start_T
        self.sigma = self.start_sigma

    def find_numbers_with_norm(self, norm, num):
        numbers = [round(2*self.limit*random.random() - self.limit , 3) for _ in range(len(self.target))]
        numbers = np.array(numbers)
        current_norm = np.linalg.norm(numbers)
        numbers = numbers * (norm / current_norm)
        return numbers.tolist()