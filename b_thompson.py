import torch
import matplotlib.pyplot as plt

k=3
true_val=torch.normal(mean=torch.zeros(k),std=torch.ones(k))
Q=torch.zeros(k)
N=torch.zeros(k)
rewards=[]
steps=1000


for t in range(1,steps+1):
    sampled_val=[]
    for a in range(k):
        if N[a]==0:
            sampled=torch.normal(mean=torch.tensor(0.0),std=torch.tensor(1.0))
        else:
            sampled=torch.normal(mean=torch.tensor(float(Q[a])),std=torch.tensor(float(1/torch.sqrt(N[a]))))

        sampled_val.append(sampled)

    action=torch.argmax(torch.tensor(sampled_val)).item()
    reward=torch.normal(mean=true_val[action],std=torch.tensor(1.0)).item()
    rewards.append(reward)
    N[action]+=1
    Q[action]+=(reward-Q[action])/N[action]

plt.figure(figsize=(10,5))
plt.plot(torch.cumsum(torch.tensor(rewards), dim=0), label="Cumulative Reward")
plt.title("Thompson Sampling for 2-Armed Bandit")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.show()

print(f"True values: {true_val}")
print(f"Estimated Q values after {steps} steps: {Q}")
print(f"Number of times each arm pulled: {N}") 
