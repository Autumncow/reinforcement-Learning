'''
    @Author:Yaping Deng
    @Function:Given tracks randomly(Black rectangle)，Agent(red rectangle) will find shortest path
     to achieve destination(yellow oval).
'''

import numpy as np
import pandas as pd

class QLearning():
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.epsilon = 0.9
        self.Action = [0,1,2,3]
        self.q_table = pd.DataFrame(columns=self.Action,dtype=np.float)

    def learn(self,s,a,r,s_prime):
        #update q-value in the q_table
        if s_prime not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.Action),
                    index=self.q_table.columns,
                    name=s_prime,
                )
            )
        q_pre = self.q_table.loc[s, a]
        if s_prime == 'terminal':
            q_tar = r
        else:

            q_tar = r + self.gamma * self.q_table.loc[s_prime, :].max()
        self.q_table.loc[s, a] += self.alpha * (q_tar - q_pre)

    def chooseAction(self,state):
        # span the q_table
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.Action),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        # ε-greedy policy to choose action
        if(np.random.uniform() < self.epsilon):
            actions_best = self.q_table.loc[state,:]
            s_a = np.random.choice(actions_best[actions_best == np.max(actions_best)].index) #Choose a action from actions
        else:
            s_a = np.random.choice((self.Action))
        return s_a







