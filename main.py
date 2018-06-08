import argparse
import os
import time
import shutil
import pdb
import math
import sys
sys.path.append("..")

from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications import inception_v3, vgg19, resnet50

from models import TSN
from opts import parser
import datasets_video
from data_generator import MFFGenerator

from utils import *


best_prec1 = 0
modules = {'VGG19': vgg19, 'ResNet50': resnet50, 'InceptionV3': inception_v3}


def main():
    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()

    categories, train_list, val_list, data_root_path, prefix = datasets_video.return_dataset(args.root_path, args.dataset, args.modality)
    
    num_class = len(categories)

    architecture = architecture_name_parser(args.architecture)

    store_name = '_'.join(['MFF', args.dataset, args.modality, architecture,
                                'segment%d'% args.num_segments, '%df1c'% args.num_motion])
    print('storing name: ' + store_name)

    
    print("Using " + architecture + " architecture")

    tsn = TSN(num_class, args.num_segments, args.modality,
                architecture=architecture,
                consensus_type=args.consensus_type,
                dropout=args.dropout, num_motion=args.num_motion,
                img_feature_dim=args.img_feature_dim,
                partial_bn=args.partialbn,
                dataset=args.dataset,
                group_norm=args.group_norm)


    model = tsn.total_model
    model.summary()

    #TODO: group normalize for non RGBDiff or RGBFlow

    transform_fn = modules[architecture].preprocess_input

    # define loss function (criterion) and optimizer
    if args.loss_type == 'cce':
        criterion = 'categorical_crossentropy'
    else:
        raise ValueError("Unknown loss type")

    # Create optimizer
    optimizer = optimizers.SGD(lr=args.lr, momentum=args.momentum, 
                                decay=args.weight_decay)

    if 'adam' in args.optimizer.lower():
        optimizer = optimizers.Adam(lr=args.lr, 
                                decay=args.weight_decay,
                                clipnorm=args.clip_gradient)
    if 'rms' in args.optimizer.lower():
        optimizer = optimizers.RMSprop(lr=args.lr, decay=args.weight_decay)

    # #######
    model.compile(optimizer, loss=criterion, metrics=['accuracy'])

    if args.experiment_name:
        log_dir = './logs/'+args.experiment_name
    else:
        log_dir = './logs'

    tb = TensorBoard(log_dir=log_dir, batch_size=args.batch_size)
    checkpoint = ModelCheckpoint("model/"+store_name, save_best_only=True)

    train(model, args, optimizer, num_class, data_root_path, tsn.image_dim, 
        train_list, val_list, [tb, checkpoint], transform_fn)
    # TODO: partialBN?
    # TODO: incorporate args.clipgradient during training


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]


    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


def train(model, args, optimizer, num_classes, data_root_path, img_dim, train_list, val_list, callbacks=[], transform_fn=None):
    train_generator = MFFGenerator(
        num_classes, train_list, data_root_path, img_dim,
        training=True, batch_size=args.batch_size, modality=args.modality, transform_fn=transform_fn
        )

    val_generator = MFFGenerator(
        num_classes, val_list, data_root_path, img_dim,
        training=False, batch_size=args.batch_size, modality=args.modality, transform_fn=transform_fn
        )

    train_steps = math.ceil(len(train_generator.video_list) / args.batch_size)
    val_steps = math.ceil(len(val_generator.video_list) / args.batch_size)
    
    model.fit_generator(
        train_generator.generator(),
        steps_per_epoch = train_steps,
        epochs = args.epochs,
        validation_data = val_generator.generator(),
        validation_steps = val_steps,
        callbacks=callbacks
    )


def evaluate():
    pass



if __name__ == '__main__':
    main()
