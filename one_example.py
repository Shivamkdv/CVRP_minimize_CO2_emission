
# ### Run Using actor_crtic environment

# # Importing Libraries

import os
import torch
import matplotlib
import numpy as  np
from Tasks import vrp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Models.actor import DRL4TSP
from Models.critc import StateCritic
from Tasks.vrp import VehicleRoutingDataset
from vrp_ortools import solve_ortools
# from VRP_vsn import solve_vrp_c02
matplotlib.use('TkAgg')
""" %matplotlib inline
 """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
print('Detected device {}'.format(device))

#For error removing
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False

#Hyperparameter
LOAD_DICT={10:20,20:30,50:40,60:40, 100:50}
MAX_DEMAND=9
STATIC_SIZE=2 #(x,y)
DYNAMIC_SIZE=2 #(load,demand)
seed=42

hidden_size=128
num_layers=1
dropout=0.1

# ### Generating problems

num_nodes=10
num_samples=1
max_load=LOAD_DICT[num_nodes]
seed=43
test_data=VehicleRoutingDataset(num_samples,
                                num_nodes,
                                max_load,
                                MAX_DEMAND,
                                seed)


batch_size=16
test_loader=DataLoader(test_data,batch_size,False,num_workers=0)

# ### Loading Model

actor=DRL4TSP(STATIC_SIZE,
              DYNAMIC_SIZE,
              hidden_size,
              test_data.update_dynamic,
              test_data.update_mask,
              num_layers,
              dropout
              ).to(device)

path=r'vrp\10\actor (1).pt'
# actor.load_state_dict(torch.load(os.path.join(os.getcwd(),path),device))

for batch_idx, batch in enumerate(test_loader):
   """  print(f'Batch {batch_idx} Static :{batch[0].shape}')
    print(f'Batch {batch_idx} Dynamic :{batch[1].shape}')
    print(f'Batch {batch_idx} Xo: {batch[2].shape}') """


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    actor.eval()
    rewards=[]
    
    for batch_idx,batch in enumerate(data_loader):
        
        static, dynamic, x0 = batch
        
        static=static.to(device)
        dynamic=dynamic.to(device)
        x0=x0.to(device)

        capacity=dynamic[:, 0, 0]  #added to accomodate 3rd dimension

        capacity=capacity.view(-1, 1, 1) #added to accomodate 3rd dimension

        x0 = torch.cat((x0,capacity), dim=1) #added to accomodate 3rd dimension
        
        with torch.no_grad():
            # tour_indices,_ = actor.forward(static,dynamic,x0)  #original
            tour_indices, _, cap = actor.forward(static, dynamic, x0) #changed, cap added

        reward = reward_fn(static, tour_indices, cap).mean().item()   #changed, cap added
        # reward=reward_fn(static,tour_indices).mean().item()   #original
        rewards.append(reward)
        
        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices)
        
    actor.train()
    return np.mean(rewards),tour_indices

def render_(static, tour_indices):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        
        idx = tour_indices[i]
        
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)
        
        idx = idx.expand(static.size(1), -1)
        #print(f'static : {static[i]}')
        #print(f'idx = {idx}')
        
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()
        #print(f'Data = {data}')
        
        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        #print(f'idx = {idx}')
        where = np.where(idx == 0)[0]
        #print(where)
        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.grid()
    plt.title('Routes Generated using RL')
    plt.tight_layout()
    plt.savefig('Routes Generated using RL', bbox_inches='tight', dpi=200)
    plt.show()
    

def or_render(locations,route):
    #plt.figure(figsize=(7,6),dpi=100)
    route=[i for i in route if len(i)>2]
    for temp,i in enumerate(route):
        x=[locations[0][index] for index in i]
        y=[locations[1][index] for index in i]
        plt.plot(x,y,label='{}'.format(temp))
    plt.scatter(locations[0][1:],locations[1][1:],c='red',label='Customer')
    plt.scatter(locations[0][0],locations[1][0],marker='*',c='black',label='Depot')
    for i,t in enumerate(zip(locations[0],locations[1])):
        plt.annotate(i,t)
    plt.legend(loc=0)
    plt.title("Route Generated using OR Tool")
    plt.grid()
    plt.savefig('Route Generated using OR Tool', bbox_inches='tight')
    plt.show()
    return 



static, demand, xo=test_data[0]
or_distance, or_routes = solve_ortools(static,demand[1],max_load,multiplier=1e4)
# print(f'static: {static}')
# print(f'demand: {demand[1]}')
# print(f'max_load: {max_load}')
# print(f'capacity: {demand[0][0]}')
# vns_distance, vns_co2_emission = solve_vrp_c02(static, demand[1], demand[0][0], F=0.772, v=0.00982, max_shake=3, num_iterations=100)
# print(f'The Total distance travelled by OR Tool method is :{vns_distance}m and CO2 emission is : {vns_co2_emission}')
print(f'The Total distance travelled by OR Tool method is :{or_distance}m ')

or_render(static,or_routes)

rl_distance, rl_routes = validate(test_loader, actor, vrp.reward,render_)
print(f'The Total distance travelled by RL method is :{rl_distance}m')
