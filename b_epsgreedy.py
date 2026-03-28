import torch
import random

k=5
true_val=torch.normal(mean=torch.zeros(k),std=torch.ones(k))
Q=torch.zeros(k)
N=torch.zeros(k)
eps=0.1
rewards=[]

for i in range(1000):
    if random.random()<eps:
        a=random.randint(0,k-1)
    else:
        a=torch.argmax(Q).item()

    reward=torch.normal(mean=true_val[a],std=1).item()
    rewards.append(reward)
    N[a]+=1
    Q[a]+=(reward-Q[a])/N[a]

print("True values:", true_val)
print("Estimated Q:", Q)
print("Average reward:", sum(rewards) / len(rewards))


true_val=torch.normal(mean=torch.zeros(k),std=torch.ones(k))
Q=torch.zeros(k)
N=torch.zeros(k)
Rewards=[]

for i in range(1000):
    a=torch.argmax(Q).item()
    reward=torch.normal(mean=true_val[a],std=1).item()
    rewards.append(reward)
    N[a]+=1
    Q[a]+=(reward-Q[a])/N[a]

print("True values:", true_val)
print("Estimated Q:", Q)
print("Average reward:", sum(rewards) / len(rewards))
print(Rewards)