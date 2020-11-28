from math import sqrt

import numpy as np
import torch


class AttackMethod:
    def __init__(self) -> None:
        pass

    def attack(self, v_i, **kwargs):
        pass


class RandomAttack(AttackMethod):
    """
    Random value attack.
    """

    def __init__(self) -> None:
        pass

    def attack(self, v_i, **kwargs):
        return torch.randn_like(v_i) * kwargs["sigma"] + torch.ones_like(v_i) * 0.01


class MaxAttack(AttackMethod):
    """
    Return the opposite of origin value.
    """

    def __init__(self) -> None:
        pass

    def attack(self, v_i, **kwargs):
        return -v_i


class OneCoordinateAttack(AttackMethod):
    def __init__(self) -> None:
        pass

    def attack(self, v_i, **kwargs):
        size = v_i.shape[0]
        gamma = sqrt(size) * 0.1  # default norm is 2, sigma: 0.1 briefly
        zeros = torch.zeros_like(v_i)
        zeros[kwargs["attack_dim"], 0] = 1  # one hot
        return v_i + gamma * zeros  # use v_i instead of the mean of n-f gradients


class SwitcherAttack(AttackMethod):
    """
    50% random attack and 50% max attack.
    """

    def __init__(self) -> None:
        self.random_attack = RandomAttack()
        self.max_attack = MaxAttack()
        pass

    def attack(self, v_i, **kwargs):
        if np.random.rand() < 0.5:
            print("Random Attack")
            return self.random_attack.attack(v_i, sigma=kwargs["sigma"])
        else:
            print("Max Attack")
            return self.max_attack.attack(v_i)


attack_methods = {
    "Random": RandomAttack,
    "Max": MaxAttack,
    "OneCoordinate": OneCoordinateAttack,
    "Switcher": SwitcherAttack,
}
