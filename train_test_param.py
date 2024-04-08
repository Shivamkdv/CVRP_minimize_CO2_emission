{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0c7de2d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting matplotlib\n",
      "  Downloading matplotlib-3.7.2-cp39-cp39-win_amd64.whl (7.5 MB)\n",
      "     ---------------------------------------- 7.5/7.5 MB 9.2 MB/s eta 0:00:00\n",
      "Collecting cycler>=0.10\n",
      "  Using cached cycler-0.11.0-py3-none-any.whl (6.4 kB)\n",
      "Collecting importlib-resources>=3.2.0\n",
      "  Downloading importlib_resources-6.0.0-py3-none-any.whl (31 kB)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\admin\\anaconda3\\envs\\actor_critic\\lib\\site-packages (from matplotlib) (22.0)\n",
      "Requirement already satisfied: numpy>=1.20 in c:\\users\\admin\\anaconda3\\envs\\actor_critic\\lib\\site-packages (from matplotlib) (1.25.1)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in c:\\users\\admin\\anaconda3\\envs\\actor_critic\\lib\\site-packages (from matplotlib) (2.8.2)\n",
      "Collecting contourpy>=1.0.1\n",
      "  Downloading contourpy-1.1.0-cp39-cp39-win_amd64.whl (429 kB)\n",
      "     -------------------------------------- 429.4/429.4 kB 8.9 MB/s eta 0:00:00\n",
      "Collecting pillow>=6.2.0\n",
      "  Downloading Pillow-10.0.0-cp39-cp39-win_amd64.whl (2.5 MB)\n",
      "     ---------------------------------------- 2.5/2.5 MB 10.0 MB/s eta 0:00:00\n",
      "Collecting kiwisolver>=1.0.1\n",
      "  Using cached kiwisolver-1.4.4-cp39-cp39-win_amd64.whl (55 kB)\n",
      "Collecting pyparsing<3.1,>=2.3.1\n",
      "  Using cached pyparsing-3.0.9-py3-none-any.whl (98 kB)\n",
      "Collecting fonttools>=4.22.0\n",
      "  Downloading fonttools-4.41.0-cp39-cp39-win_amd64.whl (2.0 MB)\n",
      "     ---------------------------------------- 2.0/2.0 MB 7.0 MB/s eta 0:00:00\n",
      "Collecting zipp>=3.1.0\n",
      "  Downloading zipp-3.16.2-py3-none-any.whl (7.2 kB)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\admin\\anaconda3\\envs\\actor_critic\\lib\\site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)\n",
      "Installing collected packages: zipp, pyparsing, pillow, kiwisolver, fonttools, cycler, contourpy, importlib-resources, matplotlib\n",
      "Successfully installed contourpy-1.1.0 cycler-0.11.0 fonttools-4.41.0 importlib-resources-6.0.0 kiwisolver-1.4.4 matplotlib-3.7.2 pillow-10.0.0 pyparsing-3.0.9 zipp-3.16.2\n"
     ]
    }
   ],
   "source": [
    "!pip install matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d48831c5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Detected device cpu\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import time\n",
    "import argparse\n",
    "import datetime\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "from Models.actor import DRL4TSP\n",
    "from Tasks import vrp\n",
    "from Tasks.vrp import VehicleRoutingDataset\n",
    "from Models.critc import StateCritic\n",
    "\n",
    "torch.backends.cudnn.benchmark = True\n",
    "torch.backends.cudnn.enabled=False\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print('Detected device {}'.format(device))\n",
    "\n",
    "\n",
    "def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',\n",
    "             num_plot=5):\n",
    "    \"\"\"Used to monitor progress on a validation set & optionally plot solution.\"\"\"\n",
    "\n",
    "    actor.eval()\n",
    "\n",
    "    if not os.path.exists(save_dir):\n",
    "        os.makedirs(save_dir)\n",
    "\n",
    "    rewards = []\n",
    "    for batch_idx, batch in enumerate(data_loader):\n",
    "\n",
    "        static, dynamic, x0 = batch\n",
    "\n",
    "        static = static.to(device)\n",
    "        dynamic = dynamic.to(device)\n",
    "        x0 = x0.to(device) if len(x0) > 0 else None\n",
    "\n",
    "        with torch.no_grad():\n",
    "            tour_indices, _ = actor.forward(static, dynamic, x0)\n",
    "\n",
    "        reward = reward_fn(static, tour_indices).mean().item()\n",
    "        rewards.append(reward)\n",
    "\n",
    "        if render_fn is not None and batch_idx < num_plot:\n",
    "            name = 'batch%d_%2.4f.png'%(batch_idx, reward)\n",
    "            path = os.path.join(save_dir, name)\n",
    "            render_fn(static, tour_indices, path)\n",
    "\n",
    "    actor.train()\n",
    "    return np.mean(rewards)\n",
    "\n",
    "def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,\n",
    "          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,\n",
    "          **kwargs):\n",
    "    \"\"\"Constructs the main actor & critic networks, and performs all training.\"\"\"\n",
    "\n",
    "    now = '%s' % datetime.datetime.now().time()\n",
    "    now = now.replace(':', '_')\n",
    "    save_dir = os.path.join(task, '%d' % num_nodes, now)\n",
    "\n",
    "    print('Starting training')\n",
    "\n",
    "    checkpoint_dir = os.path.join(save_dir, 'checkpoints')\n",
    "    if not os.path.exists(checkpoint_dir):\n",
    "        os.makedirs(checkpoint_dir)\n",
    "\n",
    "    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)\n",
    "    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)\n",
    "\n",
    "    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)\n",
    "    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)\n",
    "\n",
    "\n",
    "\n",
    "    best_params = None\n",
    "    best_reward = np.inf\n",
    "\n",
    "    for epoch in range(20):\n",
    "\n",
    "        actor.train()\n",
    "        critic.train()\n",
    "\n",
    "        times, losses, rewards, critic_rewards = [], [], [], []\n",
    "\n",
    "        epoch_start = time.time()\n",
    "        start = epoch_start\n",
    "\n",
    "        for batch_idx, batch in enumerate(train_loader):\n",
    "\n",
    "            static, dynamic, x0 = batch\n",
    "\n",
    "            static = static.to(device)\n",
    "            dynamic = dynamic.to(device)\n",
    "            x0 = x0.to(device) if len(x0) > 0 else None\n",
    "\n",
    "            # Full forward pass through the dataset\n",
    "            tour_indices, tour_logp = actor(static, dynamic, x0)\n",
    "\n",
    "            # Sum the log probabilities for each city in the tour\n",
    "            reward = reward_fn(static, tour_indices)\n",
    "\n",
    "            # Query the critic for an estimate of the reward\n",
    "            critic_est = critic(static, dynamic).view(-1)\n",
    "\n",
    "            advantage = (reward - critic_est)\n",
    "            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))\n",
    "            critic_loss = torch.mean(advantage ** 2)\n",
    "\n",
    "            actor_optim.zero_grad()\n",
    "            actor_loss.backward()\n",
    "            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)\n",
    "            actor_optim.step()\n",
    "\n",
    "            critic_optim.zero_grad()\n",
    "            critic_loss.backward()\n",
    "            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)\n",
    "            critic_optim.step()\n",
    "\n",
    "            critic_rewards.append(torch.mean(critic_est.detach()).item())\n",
    "            rewards.append(torch.mean(reward.detach()).item())\n",
    "            losses.append(torch.mean(actor_loss.detach()).item())\n",
    "\n",
    "            if (batch_idx + 1) % 100 == 0:\n",
    "                end = time.time()\n",
    "                times.append(end - start)\n",
    "                start = end\n",
    "\n",
    "                mean_loss = np.mean(losses[-100:])\n",
    "                mean_reward = np.mean(rewards[-100:])\n",
    "\n",
    "                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %\n",
    "                      (batch_idx, len(train_loader), mean_reward, mean_loss,\n",
    "                       times[-1]))\n",
    "\n",
    "        mean_loss = np.mean(losses)\n",
    "        mean_reward = np.mean(rewards)\n",
    "\n",
    "        # Save the weights\n",
    "        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)\n",
    "        if not os.path.exists(epoch_dir):\n",
    "            os.makedirs(epoch_dir)\n",
    "\n",
    "        save_path = os.path.join(epoch_dir, 'actor.pt')\n",
    "        torch.save(actor.state_dict(), save_path)\n",
    "\n",
    "        save_path = os.path.join(epoch_dir, 'critic.pt')\n",
    "        torch.save(critic.state_dict(), save_path)\n",
    "\n",
    "        # Save rendering of validation set tours\n",
    "        valid_dir = os.path.join(save_dir, '%s' % epoch)\n",
    "\n",
    "        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,\n",
    "                              valid_dir, num_plot=5)\n",
    "\n",
    "        # Save best model parameters\n",
    "        if mean_valid < best_reward:\n",
    "\n",
    "            best_reward = mean_valid\n",
    "\n",
    "            save_path = os.path.join(save_dir, 'actor.pt')\n",
    "            torch.save(actor.state_dict(), save_path)\n",
    "\n",
    "            save_path = os.path.join(save_dir, 'critic.pt')\n",
    "            torch.save(critic.state_dict(), save_path)\n",
    "\n",
    "        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\\\n",
    "              '(%2.4fs / 100 batches)\\n' % \\\n",
    "              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,\n",
    "              np.mean(times)))\n",
    "\n",
    "\n",
    "def train_vrp(args):\n",
    "\n",
    "    # Goals from paper:\n",
    "    # VRP10, Capacity 20:  4.84  (Greedy)\n",
    "    # VRP20, Capacity 30:  6.59  (Greedy)\n",
    "    # VRP50, Capacity 40:  11.39 (Greedy)\n",
    "    # VRP100, Capacity 50: 17.23  (Greedy)\n",
    "\n",
    "    print('Starting VRP training')\n",
    "\n",
    "    # Determines the maximum amount of load for a vehicle based on num nodes\n",
    "    #LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}\n",
    "    LOAD_DICT = {5:10,10: 20, 20: 30, 50: 40, 100: 50}\n",
    "    MAX_DEMAND = 9\n",
    "    STATIC_SIZE = 2 # (x, y)\n",
    "    DYNAMIC_SIZE = 2 # (load, demand)\n",
    "\n",
    "    max_load = LOAD_DICT[args.num_nodes]\n",
    "\n",
    "    train_data = VehicleRoutingDataset(args.train_size,\n",
    "                                       args.num_nodes,\n",
    "                                       max_load,\n",
    "                                       MAX_DEMAND,\n",
    "                                       args.seed)\n",
    "\n",
    "    print('Train data: {}'.format(train_data))\n",
    "    valid_data = VehicleRoutingDataset(args.valid_size,\n",
    "                                       args.num_nodes,\n",
    "                                       max_load,\n",
    "                                       MAX_DEMAND,\n",
    "                                       args.seed + 1)\n",
    "\n",
    "    actor = DRL4TSP(STATIC_SIZE,\n",
    "                    DYNAMIC_SIZE,\n",
    "                    args.hidden_size,\n",
    "                    train_data.update_dynamic,\n",
    "                    train_data.update_mask,\n",
    "                    args.num_layers,\n",
    "                    args.dropout).to(device)\n",
    "    print('Actor: {} '.format(actor))\n",
    "\n",
    "    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)\n",
    "\n",
    "    print('Critic: {}'.format(critic))\n",
    "\n",
    "    kwargs = vars(args)\n",
    "    kwargs['train_data'] = train_data\n",
    "    kwargs['valid_data'] = valid_data\n",
    "    kwargs['reward_fn'] = vrp.reward\n",
    "    kwargs['render_fn'] = vrp.render\n",
    "\n",
    "    if args.checkpoint:\n",
    "        path = os.path.join(args.checkpoint, 'actor.pt')\n",
    "        actor.load_state_dict(torch.load(path, device))\n",
    "\n",
    "        path = os.path.join(args.checkpoint, 'critic.pt')\n",
    "        critic.load_state_dict(torch.load(path, device))\n",
    "\n",
    "    if not args.test:\n",
    "        train(actor, critic, **kwargs)\n",
    "\n",
    "    test_data = VehicleRoutingDataset(args.valid_size,\n",
    "                                      args.num_nodes,\n",
    "                                      max_load,\n",
    "                                      MAX_DEMAND,\n",
    "                                      args.seed + 2)\n",
    "\n",
    "    test_dir = 'test'\n",
    "    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)\n",
    "    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)\n",
    "\n",
    "    print('Average tour length: ', out)\n",
    "\n",
    "\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "49f8432a",
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == '__main__':\n",
    "\n",
    "    parser = argparse.ArgumentParser(description='Combinatorial Optimization')\n",
    "    parser.add_argument('--seed', default=25, type=int)\n",
    "    parser.add_argument('--checkpoint', default=None)\n",
    "    parser.add_argument('--test', action='store_true', default=False)\n",
    "    parser.add_argument('--task', default='vrp')\n",
    "    parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)\n",
    "    parser.add_argument('--actor_lr', default=5e-3, type=float)\n",
    "    parser.add_argument('--critic_lr', default=5e-3, type=float)\n",
    "    parser.add_argument('--max_grad_norm', default=2., type=float)\n",
    "    parser.add_argument('--batch_size', default=256, type=int)\n",
    "    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)\n",
    "    parser.add_argument('--dropout', default=0.1, type=float)\n",
    "    parser.add_argument('--layers', dest='num_layers', default=1, type=int)\n",
    "    parser.add_argument('--train-size',default=100000, type=int)\n",
    "    parser.add_argument('--valid-size', default=100, type=int)\n",
    "\n",
    "    args = parser.parse_args()\n",
    "\n",
    "    #print('NOTE: SETTTING CHECKPOINT: ')\n",
    "    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)\n",
    "    #print(args.checkpoint)\n",
    "\n",
    "    \n",
    "    \n",
    "    train_vrp(args)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}