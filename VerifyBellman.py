from utils import * #import util

S=['Playing','class1','class2','class3','rest']
A=['play','study','leavepaly','pubing','leavestudy']
R={}
P={}
gamma=1.0

#initialize
set_prob(P,S[0],A[0],S[0])
set_prob(P,S[0],A[2],S[1])
set_prob(P,S[1],A[0],S[0])
set_prob(P,S[1],A[1],S[2])
set_prob(P,S[2],A[1],S[3])
set_prob(P,S[2],A[4],S[4])
set_prob(P,S[3],A[1],S[4])
set_prob(P,S[3],A[3],S[1],p=0.2)
set_prob(P,S[3],A[3],S[2],p=0.4)
set_prob(P,S[3],A[3],S[3],p=0.4)

set_reward(R,S[0],A[0],-1)
set_reward(R,S[0],A[2],0)
set_reward(R,S[1],A[0],-1)
set_reward(R,S[1],A[1],-2)
set_reward(R,S[2],A[1],-2)
set_reward(R,S[2],A[4],0)
set_reward(R,S[3],A[1],10)
set_reward(R,S[3],A[3],+1)

MDP=(S,P,A,R,gamma)
print('this probability matrix:')
display_dict(P)
print('This is R matrix:')
display_dict(R)

Pi={}  #initialize policy pi
set_pi(Pi, S[0], A[0], 0.5)
set_pi(Pi, S[0], A[2], 0.5)
set_pi(Pi, S[1], A[0], 0.5)
set_pi(Pi, S[1], A[1], 0.5)
set_pi(Pi, S[2], A[1], 0.5)
set_pi(Pi, S[2], A[4], 0.5)
set_pi(Pi, S[3], A[1], 0.5)
set_pi(Pi, S[3], A[3], 0.5)

print('This is policy matrix:')
display_dict(Pi)

V={}
print('This is value of state:')
display_dict(V)

#compute action-value 
def compute_q(MDP,V,s,a): #calculate the q-value
    S,P,A,R,gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P,s,a,s_prime)*get_value(V,s_prime)
        q_sa = get_reward(R,s,a)+gamma*q_sa
    return q_sa

#compute state-value
def compute_v(MDP,V,Pi,s):
    S,P,A,R,gamma = MDP
    v_s=0
    for a in A:
        v_s += get_pi(Pi,s,a)*compute_q(MDP,V,s,a)
    return v_s
#update state-value
def update_v(MDP,V,Pi):
    S,_,_,_,_ = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[str_key(s)] = compute_v(MDP,V_prime,Pi,s)
    return V_prime

def policy_evaluation(MDP,V,Pi,n):
    for i in range(n):
        V=update_v(MDP,V,Pi)
    return V

V=policy_evaluation(MDP,V,Pi,1)
display_dict(V)




