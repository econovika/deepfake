from extract.train import TrainEvalModel
from argparse import ArgumentParser


parser = ArgumentParser(description='PyTorch YoloV3-Face implementation')
# YoloV3-Face model settings:
parser.add_argument('--model', type=str, default=None, metavar='M',
                    help='path to saved model to evaluate (default: None: define YoloV3-Face model and do training)')
parser.add_argument('--num-classes', type=int, default=2, metavar='M',
                    help='number of classes to classify images (default: 2: object exists on image or not)')
parser.add_argument('--valid-anchors-wh', type=str, default='/home/ksenia/PycharmProjects/deepfake/plugins/extract/default_options/anchors', metavar='M',
                    help='path to file with anchors definition, '
                         'see default file for example of formatting (default: None: compute optimal anchors '
                         'from given training dataset)')
# Training settings:
parser.add_argument('--train-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--eval-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for evaluation (default: 1)')
parser.add_argument('--num-epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--max-patience', type=int, default=5, metavar='N',
                    help='number of epochs to train successively with no loss improving before '
                         'lr reduction (default: 5)')
parser.add_argument('--early-stopping', type=int, default=10, metavar='N',
                    help='number of epochs to train successively with no loss improving (default: 10)')
# Optimizer settings:
parser.add_argument('--lr', type=float, default=0.01, metavar='O',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='O',
                    help='SGD momentum (default: 0.8)')
# Dataset / Dataloader / Model settings
parser.add_argument('--train-input-path', type=str, metavar='D',
                    help='path to input training dataset in format: path to landmarks file; path to images directory')
parser.add_argument('--eval-input-path', type=str, metavar='D',
                    help='path to input evaluation dataset in format: '
                         'path to landmarks file (optional); path to images directory')
parser.add_argument('--model-save-directory', type=str, default=None, metavar='D',
                    help='path to model save directory (default: ./models)')
parser.add_argument('--seed', type=int, default=1, metavar='D',
                    help='random seed (default: 1)')
# Other settings:
parser.add_argument('--log-interval', type=int, default=10, metavar='S',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-workers', type=int, default=1, metavar='S',
                    help='how many training workers to use (default: 1)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')


if __name__ == '__main__':
    args = parser.parse_args()
    input_data_paths = {
        'train': args.train_input_path.split(';'),
        'eval': args.eval_input_path.split(';')
    }
    train_eval = TrainEvalModel(
        model=args.model,
        input_data_paths=input_data_paths,
        train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        lr=args.lr, momentum=args.momentum, num_epochs=args.num_epochs,
        log_interval=args.log_interval, num_workers=args.num_workers, max_patience=args.max_patience,
        model_save_dir=args.model_save_directory, early_stopping=args.early_stopping,
        cuda=args.cuda,
        seed=args.seed,
        valid_anchors_wh=args.valid_anchors_wh, num_classes=args.num_classes
    )
    if input_data_paths['train']:
        preds, GIoU_list = train_eval(train=True)
    else:
        preds, GIoU_list = train_eval()
