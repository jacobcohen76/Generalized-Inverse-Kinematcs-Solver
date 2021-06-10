import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import functools
import operator
import random

from typing import Any, Callable, List, Tuple
from nptyping import NDArray

class Link:
    """
    Callable constructor for Denavit-Hartenberg Matrices of some D-H parameters.

    Attributes:
        offset (float): link offset (d in mathematical notation)
        length (float): link length (a in mathematical notation)
        alpha  (float): link twist
        bounds (Tuple[float, float]): boundaries of joint angle, use None for no boundary ie. (None, None)
    """
    def __init__(self,
            offset: float, length: float, alpha: float,
            bounds: Tuple[float, float] = (None, None)
        ) -> None:
        self.offset = offset
        self.length = length
        self.alpha = alpha
        self.bounds = bounds

    def __call__(self, theta: float) -> NDArray[(4, 4), float]:
        """
        Constructs the Denavit-Hartenberg Matrix for this Link given the joint
        angle theta.

        Args:
            theta (float): the joint angle to use
        
        Returns:
            A 4x4 numpy float array representing the Denavit-Hartenberg Matrix
            for this Link using the given joint angle theta.
        """
        ca, ct = np.cos(self.alpha), np.cos(theta)
        sa, st = np.sin(self.alpha), np.sin(theta)
        return np.array([[  ct, -st * ca,  st * sa, self.length * ct],
                         [  st,  ct * ca, -ct * sa, self.length * st],
                         [ 0.0,       sa,       ca,      self.offset],
                         [ 0.0,      0.0,      0.0,              1.0]], dtype=float)

def forward(chain: List[Link], config: List[float]) -> NDArray[(4, 4), float]:
    """
    Performs forward kinematics on a Link chain.

    Args:
        chain  (List[Link]): a list of Links
        config (List[float]): the parameters to be forwarded to the link
    
    Returns:
        The resulting Denavit-Hartenberg Matrix acquired from applying all of
        the transformations for the given configuration.
    """
    assert(len(chain) == len(config))
    return functools.reduce(operator.matmul, (link(angle) for link, angle in zip(chain, config)))

def inverse(chain: List[Link], target: NDArray[(3,), float], config: NDArray[(Any,), float] = None) -> NDArray[(Any,), float]:
    """
    Performs inverse kinematics on a chain.

    Args:
        chain (List[Link]): a list of Links
        target (NDArray[(3,), float]): the end effector position we are trying to reach
        config (List[float]): the initial configuration to use
    
    Returns:
        The found configuration to minimize the distance of applying forward
        kinematics to the target.
    """
    if config == None:
        config = np.zeros((len(chain),), dtype=float)
    assert(len(chain) == len(config))
    def loss(config: NDArray[(Any,), float]) -> float:
        forward_dh_mat = forward(chain, config)
        pos = forward_dh_mat[0:3, 3]
        return np.linalg.norm(target - pos)
    results = scipy.optimize.minimize(loss, config, method='L-BFGS-B', bounds=[link.bounds for link in chain])
    return results.x

def show_arm(chain: List[Link], config: List[float], ax, pt_color: str, line_color: str) -> None:
    """
    Performs forward kiinematics on a Link chain and visually displays the
    transformations to the matplot axes.
    """
    mat = np.identity(n=4, dtype=float)
    curr_pos = mat[0:3, 3]
    ax.plot(*curr_pos, pt_color)
    for link, angle in zip(chain, config):
        mat = mat @ link(angle)
        prev_pos = curr_pos
        curr_pos = mat[0:3, 3]
        ax.plot(*curr_pos, pt_color)
        ax.plot(*zip(prev_pos, curr_pos), line_color)

def run_test(chain: List[Link], target: NDArray[(3,), float]) -> None:
    ax = plt.axes(projection='3d')
    target_config = inverse(chain, target)
    show_arm(chain, target_config, ax, 'bo', 'blue')
    ax.plot(*target, 'ro')
    ax.set_xlim(-2.0, +2.0)
    ax.set_ylim(-2.0, +2.0)
    ax.set_zlim( 0.0, +4.0)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=45, elev=10)
    plt.show()

def main(args: argparse.Namespace) -> int:
    random_target = lambda: np.array([random.uniform(-2.0, +2.0), random.uniform(-2.0, +2.0), random.uniform(-1.0, +4.0)], dtype=float)
    chain = [
        #    offset  length         twist      min-angle     max-angle
        Link(  +1.0,      0, +np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +0.2,      0, -np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +1.0,      0, +np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  -0.2,      0, -np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +1.0,      0, +np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +0.2,      0, -np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +1.0,      0, +np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  -0.2,      0, -np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +1.0,      0, +np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +0.2,      0, -np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
        Link(  +1.0,      0, +np.pi / 2.0, (-np.pi / 2.0, +np.pi / 2.0)),
    ]
    for test in range(args.num_tests):
        run_test(chain, random_target())
    return 0

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_tests', default=1, type=int)
    args = arg_parser.parse_args()
    exit(main(args))
