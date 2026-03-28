import torch
import matplotlib.pyplot as plt
'''
k=5
true_val=torch.normal(mean=torch.zeros(k),std=torch.ones(k))

Q=torch.zeros(k)
N=torch.zeros(k)

rewards=[]
for i in range(1000):
    action=torch.argmax(Q).item()
    reward=torch.normal(mean=true_val[action],std=1).item()
    rewards.append(reward)
    N[action]+=1
    Q[action]+=(reward-Q[action])/N[action]

print("True values:", true_val)
print("Estimated Q:", Q)
print("Average reward:", sum(rewards) / len(rewards))


'''

class Bandit_env:
    def __init__(self,k=5,reward_std=1,random_int_mean=True):
        self.k=k
        self.reward_std=reward_std
        self.rewards=[]
        self.N=torch.zeros(k)
        self.Q=torch.zeros(k)

        if random_int_mean:
            self.true_val=torch.randint(0,10,(k,),dtype=torch.float)
        else:
            self.true_val=torch.normal(mean=torch.zeros(k),std=torch.ones(k))
            
    def pull(self,action):
        reward=torch.normal(mean=self.true_val[action],std=torch.tensor(self.reward_std)).item()
        return reward
        
    def select_action(self):
        action=torch.argmax(self.Q).item()
        return int(action)
        
    def update_Q(self,action,reward):
        self.N[action]+=1
        self.Q[action]+=(reward-self.Q[action])/self.N[action]
        self.rewards.append(reward)
        return {
            "action": action,
            "reward": round(reward, 4),
            "updated_Q": round(self.Q[action].item(), 4),
        "N": int(self.N[action].item())
        }

env = Bandit_env(k=5)

for step in range(10):
    action = env.select_action()
    reward = env.pull(action)
    result = env.update_Q(action, reward)
    print(f"Step {step+1}: {result}")
            


