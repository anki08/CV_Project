from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Planner().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train15'))

    """
    Your code here
    Hint: Use the log function below to debug and visualize your model
    """

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = torch.nn.MSELoss()

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    train_data = load_data('../drive_data', batch_size=60, transform=transform)
    global_step_1 = 0
    epoch = args.num_epoch

    for epoch in range(epoch):
        model.train()
        for img, peak in train_data:
            """calculate loss, add optimizer"""
            global_step_1 += 1

        model.eval()

        print("epoch", epoch)
        if (epoch == 20):
            optimizer.param_groups[0]['lr'] = 0.009
        if (epoch == 30):
            optimizer.param_groups[0]['lr'] = 0.005
        if (epoch == 40):
            optimizer.param_groups[0]['lr'] = 0.003
        if (epoch == 55):
            optimizer.param_groups[0]['lr'] = 0.001

        save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.011487908395619178)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), ToTensor()])')

    args = parser.parse_args()
    train(args)
