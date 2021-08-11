import argparse


def parse_args():

    parser = argparse.ArgumentParser(
                        description="common ground control")

    parser.add_argument('-r',
                        '--run_mode',
                        help="run mode (hello=hello cgc/process=img processsing/dataset=create dataset/learn=run ml)",
                        type=str
                        )
    parser.add_argument('-e',
                        '--environment',
                        help="environment to run in (test/production)",
                        default='production',
                        type=str
                        )
    parser.add_argument('-l',
                        '--level',
                        help="level of toolpath (all/1-0/1-1/1-2/2-0/2-1/2-2)",
                        default='1-0',
                        type=str
                        )
    parser.add_argument('-t',
                        '--img_type',
                        help="image type for ML (c2c/c2h/h2h/h2c/b2g/g2b)",
                        default='h2h',
                        type=str
                        )

    args = parser.parse_args()

    return args.run_mode, args.environment, args.level, args.img_type


if __name__ == "__main__":
    pass
