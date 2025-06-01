import argparse


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', default="metaworld")
    parser.add_argument('--max_episode_steps', type=int, default=200)
    parser.add_argument('--num_eval_episodes', type=int, default=1)

    parser.add_argument('--eval_tasks', default=[1, 4, 14, 15])

    # for data collection
    parser.add_argument('--num_tasks', type=int, default=22)
    parser.add_argument('--num_train_tasks', type=int, default=18)
    # parser.add_argument('--context_horizon', type=int, default=4)

    # for decision transformer
    parser.add_argument('--dt_batch_size', default=32, type=int)
    parser.add_argument('--dt_horizon', type=int, default=50)
    parser.add_argument('--dt_embed_dim', type=int, default=128)
    parser.add_argument('--dt_n_layer', type=int, default=3)
    parser.add_argument('--dt_n_head', type=int, default=1)
    parser.add_argument('--dt_activation_function', type=str, default='relu')
    parser.add_argument('--dt_dropout', type=float, default=0.1)
    parser.add_argument('--dt_lr', type=float, default=1e-4)
    parser.add_argument('--dt_weight_decay', type=float, default=1e-4)
    # parser.add_argument('--dt_warmup_steps', type=int, default=1000)
    parser.add_argument('--dt_return_scale', type=float, default=800.)
    parser.add_argument('--dt_target_return', type=float, default=1500.)
    # parser.add_argument('--dt_num_iters', type=int, default=1e4)
    # parser.add_argument('--dt_eval_iters', type=int, default=100)

    # for meta-decision transformer
    parser.add_argument('--transformer_batch_size', default=128, type=int)
    parser.add_argument('--transformer_warmup_steps', type=int, default=10000)
    parser.add_argument('--transformer_num_iters', type=int, default=100000)
    parser.add_argument('--transformer_eval_iters', type=int, default=500)

    # for prompt-decision transformer
    parser.add_argument('--prompt_len', default=4, type=int)
    parser.add_argument('--num_demos', default=5, type=int)

    args = parser.parse_args(rest_args)
    return args
