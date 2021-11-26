import argparse
from gym import envs


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Filter for OpenAI Gym environments')
    parser.add_argument(
            "-f", "-filter",
            nargs="?",
            default="Pacman",
            help="Filter for the OpenAI environment list to be matched.")
    args = parser.parse_args()
    #filt =  str(args.f)
    for e in envs.registry.all():
        if args.f in str(e):
            print(e)

if __name__ == "__main__":
    main()
