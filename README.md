# Text-to-Decision Agent: Offline Meta-Reinforcement Learning from Natural Language Supervision

## **Overview**
Official codebase for T2DA: **T**ext-to-**D**ecision **A**gent: Offline Meta-Reinforcement Learning from Natural Language Supervision. Paper link: [[ArXiv]](https://arxiv.org/abs/2504.15046).

## **Installation**
Experiments require MuJoCo and Metaworld. Follow the instructions in the [[MuJoCo]](https://github.com/openai/mujoco-py)[[Metaworld]](https://github.com/Farama-Foundation/Metaworld) to install.
Create a virtual environment using conda, and see `requirments.txt` file for more information about how to install the dependencies.
```shell
conda create -n t2da python=3.8.20 -y
conda activate t2da
pip install -r requirements.txt
```

## **Data Collection**
Note that we set ```done = False``` in all environments.

### Train SAC
We use SAC to train agents on different environments and collect datasets.  
Train agents on different tasks in PointRobot-v0:
```shell
python train_data_collection.py --env point-robot --save_freq 40 --task_id_start 0 --task_id_end 5
```
in which ```task_id_start``` and ```task_id_end``` mean that training tasks of [task_id_start, task_id_end).

### Generate Datasets
We use all checkpoints of traning process to generate datasets.
```shell
python get_datasets_mix.py --env point-robot --task_id_start 0 --task_id_end 5 --capacity 80
```
in which ```capacity``` means the number of transitions collected by one checkpoint.
For example, for the PointRobot-v0 environment, we collect 80 transitions (4 episodes) per checkpoint.

## **Train Trajectory Encoder**
Train the Trajectory Encoder on different tasks in PointRobot-v0:
```shell
python train_traj_encoder.py --env point-robot --context_horizon 20
```
We use an entire trajectory as the context, so the parameter ```context_horizon``` should be consistent with max_episode_steps of the environment.
The trained checkpoint will be saved in `saves_world_model/point-robot/`.

## **Align Text Encoder with Trajectory Encoder**
Fine-tune the text encoder to align the produced text embeddings with dynamics-aware decision embeddings:
```shell
python train_align.py --text_encoder clip --env point-robot --context_horizon 20
```
By modifying the parameter ```text_encoder```, you can switch to using T5 or BERT as text_encoder.
The trained checkpoint will be saved in `saves_align/point-robot/`.

## **Downstream Task Training**
### Text-to-Decision Diffuser
Train the Text-to-Decision Diffuser on different tasks in PointRobot-v0:
```shell
python train_t2d_diffuser.py --env point-robot --prompts_type aligned_clip
```

Evaluate the Text-to-Decision Diffuser on different tasks in PointRobot-v0:
```shell
python evaluate_parallel.py --env point-robot --prompts_type aligned_clip
```

### Text-to-Decision Transformer
Train the Text-to-Decision Transformer on different tasks in PointRobot-v0:
```shell
python train_t2d_transformer.py --env point-robot --prompts_type aligned_clip
```
