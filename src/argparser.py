import argparse

def parse_args():
    parser = argparse.ArgumentParser(
                        description="common ground control"
    )

    parser.add_argument('-r', '--run_mode', help="run mode (train/run)", type=str)
    parser.add_argument('-e','--environment', help="environment to run in (test/production", type=str)

    args = parser.parse_args()

    return args.run_mode, args.environment

if __name__ == "__main__":
    pass


