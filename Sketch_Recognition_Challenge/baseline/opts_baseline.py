import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings

    ## model info
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet50')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--pretrain',
        type=str,
        default='/home/like/Projects/pretrain/resnet50-19c8e357.pth')

    ## parameter for optimizer
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.002)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=200)
    parser.add_argument(
        '--step_size',
        type=int,
        default=50)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.25)
    ## parameter for dataset
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)
    parser.add_argument(
        '--trainset_path',
        type=str,
        default='/Data/TUBerLin/png/')
    parser.add_argument(
        '--testset_path',
        type=str,
        default='/Data/TUBerLin/png/')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=20)
    ## loss infor
    parser.add_argument(
        '--loss_func',
        type=str,
        default='crossentropy'#arcface
    )

    parser.add_argument(
        '--test_type',
        type=str,
        default='single' #single,multi
        )
    parser.add_argument(
        '--gpus',
        type=str,
        default='0')
    parser.add_argument(
        '--debug', 
        action='store_true')
    args = parser.parse_args()

    return args
import argparse



