import numpy as np

num_state=7
# {"0": "C1", "1":"C2", "2":"C3", "3":"Pass", "4":"Pub", "5":"FB", "6":"Sleep"
i_to_n={}
i_to_n["0"]="C1"   #Create index and initialize the reflection from index to state
i_to_n["1"]="C2"
i_to_n["2"]="C3"
i_to_n["3"]="PASS"
i_to_n["4"]="Pub"
i_to_n["5"]="FB"
i_to_n["6"]="Sleep"

n_to_i={}   #initialize the reflection from state to index
for i,name in zip(i_to_n.keys(),i_to_n.values()):
    n_to_i[name]=int(i)

Pss=[   #The matrix of probability
    [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0 ],
    [ 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2 ],
    [ 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0 ],
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
    [ 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0 ],
    [ 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0 ],
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
]

Pss=np.array(Pss)

#rewards and count
rewards=[-2, -2, -2, 10, 1, -1, 0]
gamma=0.5

# define funtion for compute return.
def Compute_reward(state_index=0,chain=None,gamma=0.5)->float:
    '''
    @Author:Autumncow
    @arg:
        state_index is the location of state would be calculate.
        chain:the chain of state
        gamma:count factor
    '''
    reward,power,gamma=0.0,0,gamma
    for i in range(state_index,len(chain)):
        reward+=np.power(gamma,power)*rewards[n_to_i[chain[i]]]  #calculate Gt
        power+=1
    return reward

chains=[  #initialize the chains
    ["C1", "C2", "C3", "PASS", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "PASS", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB"],
    ["FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

print(Compute_reward(0,chains[3],gamma=0.5))
