import pystk
import math
import torch

from SuperTuxKart.utils import PyTux


class ActionSteer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(3 , 2, bias = False)

    def forward(self, x):
        f = self.network(x)
        return f

model = ActionSteer().eval()


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    x, y = aim_point
    ang = (math.atan(x/-y)/(math.pi/2))
    steer_val = math.fabs(ang)

    """Your Code """
    if(current_vel < 20):
        action.acceleration = 1
    if(0.50 <= steer_val <= 1):
        action.drift = True
    if (0.75 <= steer_val <= 1):
        action.brake = True
    if (2 <= steer_val <= 0.10 and current_vel < 7):
        action.nitro = True
    action.steer = ang
    return action


if __name__ == '__main__':
    from argparse import ArgumentParser
    import copy
    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', default='zengarden')
    parser.add_argument('-v', '--verbose', action='store_true', default = True)
    args = parser.parse_args()
    test_controller(args)
