from QL import *
from Maze import *

def run():
    for i in range(100):
        state = env.reset()
        while True:
            env.render()
            s_a = RL.chooseAction(str(state))  # choose action randomly
            print(s_a)
            s_prime, r, is_done = env.step(s_a)  # preform action
            RL.learn(str(state), s_a, r, str(s_prime))
            state = s_prime
            if is_done:
                break
    print("Over!")
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = QLearning()
    env.after(10, run)
    env.mainloop()

