from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
# from ignite.engine import create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Loss, Accuracy
# from ignite.contrib.handlers import FastaiLRFinder, ProgressBar

def train(args):
    from os import path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Planner().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train15'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = torch.nn.MSELoss()

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    train_data = load_data('/Users/asinha4/UTAustin/cs342-file/homework5/drive_data', batch_size=60, transform=transform)

    # trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    # ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})
    #
    # lr_finder = FastaiLRFinder()
    # to_save = {'model': model, 'optimizer': optimizer}
    # with lr_finder.attach(trainer, to_save, diverge_th=1.5) as trainer_with_lr_finder:
    #     trainer_with_lr_finder.run(train_data)
    #
    # trainer.run(train_data, max_epochs=10)
    #
    # evaluator = create_supervised_evaluator(model, metrics={"loss": Loss(criterion)},
    #                                         device=device)
    # evaluator.run(train_data)
    #
    # print(evaluator.state.metrics)
    #
    #
    #
    # print("suggestion", lr_finder.lr_suggestion())
    # lr_finder.plot()


    #**************************************************************************************
    global_step_1 = 0
    epoch = args.num_epoch


    for epoch in range(epoch):
        model.train()
        for img, peak in train_data:
            img, peak = img.to(device), peak.to(device)
            # print(img.shape, peak.shape)
            result = model(img)
            # print("loss", result.shape, peak.shape)
            loss_img_calc = criterion(result, peak)
            # log(train_logger, img, peak, result, global_step_1)

            train_logger.add_scalar('loss_image', loss_img_calc, global_step_1)

            optimizer.zero_grad()
            loss_img_calc.backward()
            optimizer.step()
            global_step_1 += 1

            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step_1)
            log(train_logger, img, peak, result, global_step_1)

        train_logger.add_scalar('epoch', epoch, global_step_1)

        model.eval()

        # for img, peak in valid_data:
        #     img, peak = img.to(device), peak.to(device)
        #     logit = model(img)
        #     log(valid_logger, img, peak, logit, global_step_1)
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
