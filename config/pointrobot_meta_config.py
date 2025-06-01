import torch

from params_proto.proto import ParamsProto

class Config(ParamsProto):
    # misc
    seed = 100
    dataset = 'PointRobot-v0'
    bucket = f'saves_diffuser/{dataset}/{seed}'
    num_tasks = 50
    eval_tasks = [45, 46, 47, 48, 49]

    ##########################################################################
    ###                                  dataset                           ###
    ##########################################################################
    loader = 'datasets.SequenceDataset'
    normalizer = 'GaussianNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 1.
    horizon = 10
    max_path_length = 20
    termination_penalty = None
    returns_scale = 10.0        # Determined using rewards from the dataset
    test_ret = - 5 / returns_scale
    ### hyperparameters for the prompt design
    prompts_type = None         # 'oracle'
    num_demos = 5 
    prompt_traj_len = 4 
    prompt_oracle_dim = 2
    prompt_embed_dim = 16


    ##########################################################################
    ###                      Model and Diffusion                           ###
    ##########################################################################
    ### diffusion
    diffusion = 'models.GaussianDiffusion'
    predict_epsilon = True
    condition_dropout = 0.25
    condition_guidance_w = 1.2
    n_diffusion_steps = 20
    ### choose the model from [ 'TemporalUNet', 'DiT', 'MaskedDiT' ]
    model = 'models.TemporalUNet'
    hidden_dim = 128
    dim_mults = (1, 4, 8)
    depth = 4
    num_heads = 8
    mlp_ratio = 4.0


    ##########################################################################
    ###                                 Training                           ###
    ##########################################################################
    n_steps_per_epoch = 500
    loss_type = 'l2'
    n_train_steps = 200 * n_steps_per_epoch
    log_freq = n_steps_per_epoch
    save_freq = n_steps_per_epoch
    batch_size = 128
    learning_rate = 5e-5
    gradient_accumulate_every = 2
    ema_decay = 0.995
    action_weight = 10.
    loss_weights = None
    loss_discount = 1.

