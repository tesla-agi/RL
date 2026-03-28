import torch
import math

k=5
true_val=torch.normal(mean=torch.zeros(k),std=torch.ones(k))
Q=torch.zeros(k)
N=torch.zeros(k)
c=0.5
rewards=[]

for t in range(1,1000):
    ucb_val=[]
    for a in range(k):
        if N[action]==0:
            ucb=float('inf')
        else:
            hoffding_ineq=c*torch.sqrt(math.log(t)/N[a])
            ucb=Q[a]+hoffding_ineq
        ucb_val.append(ucb)
    action=torch.argmax(torch.tensor(ucb_val)).item()
    reward=torch.normal(mean=true_val[action],std=1).item()
    rewards.append(reward)
    N[action]+=1
    Q[action]+=(reward-Q[action])/N[action]

print("True values:", true_val)
print("Estimated Q:", Q)
print("Pull counts N:", N)
print("Average reward:", sum(rewards) / len(rewards))