from html import parser
from envs_dqn_agent_atari import AtariDQNAgent
import argparse
import os

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    # config = {
	# 	"gpu": True,
	# 	"training_steps": 1e8,
	# 	"gamma": 0.99,
	# 	"batch_size": 32,
	# 	"eps_min": 0.1,
	# 	"warmup_steps": 20000,
	# 	"eps_decay": 1000000,
	# 	"eval_epsilon": 0.01,
	# 	"replay_buffer_capacity": 100000,
	# 	"logdir": 'log/DQN/Enduro/',
	# 	"update_freq": 4,
	# 	"update_target_freq": 10000,
	# 	"learning_rate": 0.0000625,
    #     "eval_interval": 100,
    #     "eval_episode": 5,
	# 	"env_id": 'ALE/Enduro-v5',
	# }
	parser = argparse.ArgumentParser(description="訓練 Atari DQN")
	parser.add_argument('--network', '-n', type=str, default='DQN', help='選擇 DQN 或 Dueling 網路結構')
	parser.add_argument('--game', '-g', type=str, default='Pacman', choices=['Pacman', 'Enduro'], help='遊戲環境名稱')
	parser.add_argument('--inference', '-i', action='store_true', help='是否進行推論')
	args = parser.parse_args()
    
	config = {
		"gpu": True,
        "parallel_gpu": False,
		"training_steps": 1e8,
		"gamma": 0.99,
        "num_envs": 4,
		"batch_size": 512,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		"update_freq": 32,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 40,
        "eval_episode": 5,
        "frame_stack": 10,
		# "env_id": 'ALE/Enduro-v5',
		# "env_id": 'ALE/MsPacman-v5',
        "video": True,
        "network": args.network,
		"logdir": 'log',
        "game": args.game,
	}
	agent = AtariDQNAgent(config)
	if args.inference:
		agent.load_and_evaluate(os.path.join("saved_model", config["game"] + '_checkpoint_' + config["network"] + '_model.pth'))
	else:
		agent.train()