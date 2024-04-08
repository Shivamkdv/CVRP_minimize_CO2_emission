import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

from Models.actor import DRL4TSP
from Tasks import vrp

from Tasks.vrp import VehicleRoutingDataset
from Models.critc import StateCritic

# Ignore all warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Detected device {}'.format(device))


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""
    print('Validate: ')

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None
        # x0 = x0.to(device) if len(x0) > 0 else None
        # print('Dynamic: ',dynamic)
        # capacity=dynamic[0][0][0]
        # capacity=capacity.view(1,1,1)
        # x0 = torch.cat((x0,capacity ), dim=1)
        # print(f'x0 shape: {x0.shape}')
        # print(f'Capapcity: {capacity}')
        # print(f'capacity shape: {capacity.shape}')
        capacity=dynamic[:, 0, 0]
        capacity=capacity.view(-1, 1, 1)
        #print(f'capacity 2: {capacity}')
        #print(f'x0 shape: {x0.shape}, capacity shape: {capacity.shape}, x1 shape: {x1.shape}')
        x0 = torch.cat((x0,capacity), dim=1)


        with torch.no_grad():
            tour_indices, _, cap_1 = actor.forward(static, dynamic, x0) #changed, cap_1 added

        reward = reward_fn(static, tour_indices, cap_1).mean().item()   #changed, cap_1 added
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    # print(f'reward: {rewards}')
    return np.mean(rewards)

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    # save_dir = os.path.join('C:\\Users\\Admin\\Documents\\VRP_CO2_emission\\VRP_CO2\\VRP\\RL-VRP-PtrNtwrk\\vrp', '%d' % num_nodes, now)

    save_dir = os.path.join('vrp', '%d' % num_nodes, now)
    print(f'save_dir: {save_dir}')

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=1)
    #print(f'train_loader shape: {train_loader.shape}')
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)



    best_params = None
    best_reward = np.inf

    for epoch in range(1):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            
            static, dynamic, x0 = batch

            static = static.to(device)
            # print(f'Static: {static}')
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None
            # print(f'x0: {x0}')
            # print(f'x0 shape: {x0.shape}')
            # print('Dynamic: ',dynamic)
            #print(f'dynamic: {dynamic[1][0][0]}')
            # capacity=dynamic[0][0][0]
            # print(f'capacity: {capacity}')
            # capacity=capacity.view(1,1,1)
            # print(f'capacity 2: {capacity}')
            # x0 = torch.cat((x0,capacity ), dim=1)
            # x0[:, :, 0] += dynamic[:, :, 0]
            # print(f'capacity: {dynamic[:, 0, 0]}')
            capacity=dynamic[:, 0, 0]
            capacity=capacity.view(-1, 1, 1)
            #print(f'capacity 2: {capacity}')
            #print(f'x0 shape: {x0.shape}, capacity shape: {capacity.shape}, x1 shape: {x1.shape}')
            x0 = torch.cat((x0,capacity), dim=1)


            # print(f'x0 shape: {x0.shape}')
            # print(f'Capapcity: {capacity}')
            # print(f'capacity shape: {capacity.shape}')
            # print(f'x0: {x0}')


            # Full forward pass through the dataset
            tour_indices, tour_logp, cap = actor(static, dynamic, x0)   #changed

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices, cap)  #changed

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))


def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    #print('Starting VRP training')

    # Determines the maximum amount of load for a vehicle based on num nodes
    #LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    LOAD_DICT = {5:10,10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 5
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed)

    #print('Train data: {}'.format(train_data))
    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1)
    #print(f"Static size {STATIC_SIZE}\n Dynamic Size{DYNAMIC_SIZE} \n hidden size{args.hidden_size}\n train data update dynamic {train_data.update_dynamic} \n train data update mask {train_data.update_mask}")
    #print(f"Num layers {args.num_layers} \n droput {args.dropout}")
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)
    #print('Actor: {} '.format(actor))

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    #print('Critic: {}'.format(critic))


    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    print(bool(args.checkpoint))
    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print('Average CO2 emission(in kg): ', out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1456, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=3, type=int)
    parser.add_argument('--valid-size', default=2, type=int)

    args = parser.parse_args("")
    #print(args)
    #print('NOTE: SETTTING CHECKPOINT: ')
    # args.checkpoint = os.path.join('vrp'+ os.path.sep)
    #print(args.checkpoint)

    
    
    train_vrp(args)