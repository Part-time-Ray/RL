from td3_agent_CarRacing import CarRacingTD3Agent
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video', '-v', action='store_true', help='whether to record evaluation video')
	parser.add_argument('--seed', '-s', type=int, nargs='*', default=[], help='random seed for evaluation')
	parser.add_argument('--single', action='store_true', help='whether to use single critic')
	parser.add_argument('--disable_smoothing', action='store_true', help='disable target policy smoothing')
	parser.add_argument('--delay_update', action='store_true', help='enable delayed policy update')
	parser.add_argument('--big_action_noise', action='store_true', help='enable big action noise during training')
	parser.add_argument('--self_reward_function', action='store_true', help='enable self reward function')
	parser.add_argument('--self_reward_function2', action='store_true', help='enable self reward function 2')
	parser.add_argument('--ns', '-ns', type=str, default='', help='name suffix for logdir')
	parser.add_argument('--load_model', '-l', type=str, default='', help='path to load model')
	args = parser.parse_args()
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		# "lra": 1e-5,
		# "lrc": 1e-5,
		"replay_buffer_capacity": 5000,
		"logdir": 'log/CarRacing/td3_test/',
		"update_freq": 10 if args.delay_update else 2,
		"eval_interval": 15,
		"eval_episode": 3,
		"video": args.video,
		"seed": args.seed,
		"single_critic": args.single,
		"disable_smoothing": args.disable_smoothing,
		"big_action_noise": args.big_action_noise,
		"self_reward_function": args.self_reward_function,
		"self_reward_function2": args.self_reward_function2,
		"ns": args.ns,
		"load_model": args.load_model,
	}
	agent = CarRacingTD3Agent(config)
	agent.train()


