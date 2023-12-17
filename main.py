import argparse

from utils import Hparam
from trainer import Trainer
from data_utils import AudioLabelLoader


def main():
    args = Hparam("./config.yaml")
    train(args=args)

def train(args):
    # data loader
    train_dataset = AudioLabelLoader(args=args, set_name='train').get_data()
    valid_dataset = AudioLabelLoader(args=args, set_name='valid').get_data()

    trainer = Trainer(
        args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )
    
    if args.common.do_train:
        trainer.train()


if __name__=='__main__':
    main()