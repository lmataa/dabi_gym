import argparse
from pathlib import Path, PosixPath
import logging
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, epsilons, file, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    if epsilons:
        ax.plot(x, epsilons, color="C0")
        ax.set_ylabel("Epsilon", color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = [np.mean(scores[max(0, t-20):(t+1)]) for t in range(N)]

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)
    plt.savefig(file)

def set_verbosity(verbose):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

class DefaultParser(argparse.ArgumentParser):
    """
    Parser for functional tests.
    - TODO: Fit this into a functional_utils.py file
    - TODO: Add more arguments
    """
    def __init__(self, *args, defaults, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_arguments(defaults)
    
    def set_verbosity(self, verbose):
        set_verbosity(verbose)

    def handle(self, args):
        self.set_verbosity(args.verbose)
        if type(args.path) is PosixPath: 
            path = args.path
        elif type(args.path) is str:
            path = Path(args.path)
        else:
            raise ValueError(f"Path must be str or PosixPath")
        path.mkdir(parents=True, exist_ok=True)

    def add_arguments(self, defaults):
        self.add_argument(
            "-path",
            nargs="?",
            default=defaults["path"],
            help="Path to create the file system.")

        self.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=False,
            help="Verbose mode.")

        self.add_argument(
            "-exp_name",
            nargs="?",
            default=defaults["exp_name"],
            help="Name of the experiment.")

        self.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=defaults["epochs"],
            help="Number of epochs.")

        self.add_argument(
            "-b",
            "--batch_size",
            type=int,
            default=defaults["batch_size"],
            help="Batch size.")

        self.add_argument(
            "-g",
            "--gamma",
            type=float,
            default=defaults["gamma"],
            help="Discount factor.")
        
        self.add_argument(
            "-lr",
            "--learning_rate",
            type=float,
            default=defaults["learning_rate"],
            help="Learning rate.")

        self.add_argument(
            "-eps",
            "--epsilon",
            type=float,
            default=defaults["epsilon"],
            help="Maximum exploration rate.")
        
        self.add_argument(
            "-d",
            "--decay",
            type=float,
            default=defaults["epsilon_decay"],
            help="Rate of decay of exploration rate.")
        
        self.add_argument(
            "-m",
            "--epsilon_min",
            type=float,
            default=defaults["epsilon_min"],
            help="Minimum exploration rate.")
        
        self.add_argument(
            "-mem",
            "--memory_size",
            type=int,
            default=defaults["memory_size"],
            help="Total memory size.")

class QLearningParser(argparse.ArgumentParser):
    """
    Parser for Q-Learning.
    """
    def __init__(self, *args, defaults, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_arguments(defaults)
    
    def set_verbosity(self, verbose):
        set_verbosity(verbose)

    def handle(self, args):
        self.set_verbosity(args.verbose)
        if type(args.path) is PosixPath: 
            path = args.path
        elif type(args.path) is str:
            path = Path(args.path)
        else:
            raise ValueError(f"Path must be str or PosixPath")
        path.mkdir(parents=True, exist_ok=True)

    def add_arguments(self, defaults):
        self.add_argument(
            "-path",
            nargs="?",
            default=defaults["path"],
            help="Path to create the file system.")

        self.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=False,
            help="Verbose mode.")
        
        self.add_argument(
            "-t",
            "--test",
            action="store_true",
            default=False,
            help="Just test the model.")

        self.add_argument(
            "-exp_name",
            nargs="?",
            default=defaults["exp_name"],
            help="Name of the experiment.")

        self.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=defaults["epochs"],
            help="Number of epochs.")

        self.add_argument(
            "-a",
            "--alpha",
            type=float,
            default=defaults["alpha"],
            help="Learning rate")

        self.add_argument(
            "-g",
            "--gamma",
            type=float,
            default=defaults["gamma"],
            help="Discount factor.")
        

        self.add_argument(
            "-eps",
            "--epsilon",
            type=float,
            default=defaults["epsilon"],
            help="Maximum exploration rate.")
        
        self.add_argument(
            "-d",
            "--decay",
            type=float,
            default=defaults["epsilon_decay"],
            help="Rate of decay of exploration rate.")
        
        self.add_argument(
            "-m",
            "--epsilon_min",
            type=float,
            default=defaults["epsilon_min"],
            help="Minimum exploration rate.")
        
        self.add_argument(
            "-pat",
            "--patience",
            type=int,
            default=defaults["patience"],
            help="Patience with the environment.")

        self.add_argument(
            "-b",
            "--bins",
            type=int,
            default=defaults["bins"],
            help="Number of bins for the discretization.")
