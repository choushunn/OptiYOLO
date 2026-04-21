from teacher_training import run_training
from teacher_training.cli import config_from_args


def main() -> None:
    cfg = config_from_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
