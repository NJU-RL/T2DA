import shutil

from pathlib import Path
from torch.nn import Module, Linear, init, Embedding, LayerNorm
from typing import Tuple, Dict, Any, List, Optional


def copy_files(to_path: str, folders: List = [], files: List = [], parent: bool = True) -> None:
    '''
    Copy files to "to_path".

    :param to_path: the destination of copied files.
    '''
    path = Path(to_path) / 'Codes' if parent else Path(to_path)
    path.mkdir(parents=True, exist_ok=True)

    # copy files
    for folder in folders:
        destiantion = path / folder
        if destiantion.exists():
            shutil.rmtree(destiantion)
        shutil.copytree(folder, destiantion, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))

    for file in files:
        shutil.copy(file, path)
        
        
def soft_update(target: Module, source: Module, tau: float) -> None:
    '''
    Soft update.

    :param target: target network.
    :param source: network.
    :param tau: update ratio.
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1 - tau))
        
        
def hard_update(target: Module, source: Module) -> None:
    '''
    Soft update.

    :param target: target network.
    :param source: network.
    '''
    target.load_state_dict(source.state_dict())


def weight_init_(m, gain: float = 1.) -> None:
    '''
    Initialize weight of module m.

    :param m: model.
    '''
    if isinstance(m, Linear):
        init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, Embedding):
        init.normal_(m.weight, mean=0, std=0.01)
    elif isinstance(m, LayerNorm):
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()


def freeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
        

def unfreeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = True