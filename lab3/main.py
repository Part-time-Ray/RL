from ppo_agent_atari import AtariPPOAgent
import argparse
import random
import tqdm

if __name__ == '__main__':

	# config = {
	# 	"gpu": True,
	# 	"training_steps": 1e8,
	# 	"update_sample_count": 10000,
	# 	"discount_factor_gamma": 0.99,
	# 	"discount_factor_lambda": 0.95,
	# 	"clip_epsilon": 0.2,
	# 	"max_gradient_norm": 0.5,
	# 	"batch_size": 128,
	# 	"logdir": 'log/Enduro_release/',
	# 	"update_ppo_epoch": 3,
	# 	"learning_rate": 2.5e-4,
	# 	"value_coefficient": 0.5,
	# 	"entropy_coefficient": 0.01,
	# 	"horizon": 128,
	# 	"env_id": 'ALE/Enduro-v5',
	# 	"eval_interval": 100,
	# 	"eval_episode": 3,
	# 	"video": True,
	# }
	parser = argparse.ArgumentParser()
	parser.add_argument('--inference', '-i', action='store_true', help='inference mode')
	parser.add_argument('--load_model', '-l', type=str, default='', help='load model path for inference')
	parser.add_argument('--video', '-v', action='store_true', help='record video during inference')
	parser.add_argument('--demo', '-d', action='store_true', help='run demo episodes to find best seed')
	args = parser.parse_args()
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 30000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.1,
		"max_gradient_norm": 0.5,
		"batch_size": 256,
		"logdir": 'log/Enduro_release/',
		"update_ppo_epoch": 3,
		"learning_rate": 1e-5,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		"eval_interval": 100,
		"eval_episode": 3,
		"video": args.video,
		# "load_model": "/mnt/nfs/work/ray/RL/lab3/log/Enduro_release/model_22660280_730.pth",
		"load_model": args.load_model,
	}
	agent = AtariPPOAgent(config)
	if args.inference:
		agent.inference(args.load_model)
	elif args.demo:
		mx_reward = 0
		mx_seed = 0
		tq = tqdm.tqdm(range(1000))
		for _ in tq:
			seed = random.randint(0, 10000)
			reward = agent.demo(seed=seed)
			if reward > mx_reward:
				mx_reward = reward
				mx_seed = seed
			tq.set_description(f"Max Reward: {mx_reward} (Seed: {mx_seed})")		
		print(f"Max Reward: {mx_reward} with seed {mx_seed}")
		agent.demo()
	else:
		agent.train()



