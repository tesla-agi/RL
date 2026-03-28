import torch
import random
k=3
true_val=torch.normal(mean=torch.zeros(k),std=torch.ones(k))
tau=0.7
Q=torch.zeros(k)
N=torch.zeros(k)
rewards=[]

for i in range(100):
    exp_val=[torch.exp(q/tau) for q in Q]
    sum_exp=sum(exp_val)
    prob=[x/sum_exp for x in exp_val]

    r=random.random()
    cummulative=0
    for i,p in enumerate(prob):
        cummulative+=p
        if r<cummulative:
            action=i
            break

    reward=torch.normal(mean=true_val[action],std=1).item()
    rewards.append(reward)
    N[action]+=1
    Q[action]+=(reward-Q[action])/N[action]

print("True means:      ", true_val)
print("Estimated Q:     ", Q)
print("Pull counts N:   ", N)
#print("Average reward:  ", round(sum(rewards)/len(rewards),3))