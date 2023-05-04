import time

import torch

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

torch.cuda.empty_cache()
import warnings

import util.util as util

warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cudnn.benchmark = True


def train_model(epoch, total_iters, dataset, validation_set, model, opt):
    """
    Forward pass to train the model for one epoch

    Args:
        epoch (int): index of the current epoch
        total_iters (int): total number of training iterations so far
        dataset (dataloader): training dataloader to load the training data
        validation_set (dataloader): validation dataloader to load the validation data
        model (nn.Module): the model to train
        opt (Options): training options

    Returns:
        total_iters: updated total number of training iterations so far
    """
    model.train()
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
    iter_data_start_time = time.time()  # timer for data loading per iteration

    for i, data in enumerate(dataset):  # inner loop within one epoch
        t_data = time.time() - iter_data_start_time

        # update the count
        # In SKIT, there are two batch_size, one is batch_size for visual output (N, set to 1), and the other one is for tactile output (NT, set to 16). i.e. we sample 16 tactile patches for each visual image.
        S_key = (
            "S" if "S" in data.keys() else "S_images"
        )  # use 'S' for singleskit and skit dataset (entire sketch as input), use 'S_images' for patchskit dataset (paired patches as input)
        batch_size = data[S_key].size(0)
        total_iters += batch_size
        epoch_iter += batch_size
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()

        # set input data
        set_input_start_time = time.time()
        if epoch == opt.epoch_count and i == 0:
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            model.parallelize()
        model.set_input(data, phase="train", verbose=False)  # unpack data from dataset and apply preprocessing
        t_input = (time.time() - set_input_start_time) / batch_size

        # forward pass
        opt_param_start_time = time.time()
        model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        t_opt = (time.time() - opt_param_start_time) / batch_size

        # log results
        if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_opt, t_data, t_input)
            visualizer.plot_current_losses(
                epoch, float(epoch_iter) / dataset_size, losses, use_visdom=False, step=total_iters
            )

            # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, step=total_iters)

        # save latest model
        if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
            model.save_networks(save_suffix)

        iter_data_start_time = time.time()

    # if the validation set is not emtpy, run separate validation
    if len(validation_set) > 0:
        model.eval()
        for i, data in enumerate(validation_set):
            # update the count
            S_key = "S" if "S" in data.keys() else "S_images"
            batch_size = data[S_key].size(0)
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            # set input data
            set_input_start_time = time.time()
            model.set_input(data, phase="val", verbose=False)  # unpack data from dataset and apply
            model.test()  # run inference
            visuals = model.get_current_visuals()  # update the visual results
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

    return total_iters


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    model = create_model(opt)  # create a model given opt.model and other options
    print("The number of training images = %d" % dataset_size)

    if hasattr(opt, "dataset") and opt.dataset == "patchskit":
        # if the model uses patchskit dataset, we need to generate a separate validation dataset
        opt_val = TrainOptions().parse()
        opt_val.separate_val_set = True
        validation_set = create_dataset(opt_val)
    else:
        validation_set = []

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    opt.visualizer = visualizer

    # print training setting
    print(
        "total epoch opt.epoch_count {} opt.n_epochs {} opt.n_epochs_decay {}".format(
            opt.epoch_count, opt.n_epochs, opt.n_epochs_decay
        )
    )
    print("check display opt params opt.print_freq {} opt.display_freq {}".format(opt.print_freq, opt.display_freq))
    print("check test setting", opt.val_for_each_epoch)

    total_iters = (
        opt.epoch_count - 1
    ) * dataset_size  # the total number of training iterations, accommodate for resuming training
    training_start_time = time.time()
    eval_metrics_best = None

    for epoch in range(
        opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        if epoch % opt.update_html_epch_freq == 0:
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every update_html_epch_freq number of epochs

        print("Start epoch %d" % (epoch))
        dataset.set_epoch(epoch)

        if opt.train_for_each_epoch:
            total_iters = train_model(epoch, total_iters, dataset, validation_set, model, opt)
            print("total_iters", total_iters)

        # Log the evaluation metric after each epoch
        eval_metrics = model.get_current_metrics()
        visualizer.print_current_metrics(epoch, eval_metrics)
        visualizer.plot_current_metrics(eval_metrics, use_visdom=False, step=total_iters)
        # Save the quantitative metrics as a dictionary file.
        visualizer.save_current_metrics(eval_metrics, epoch=epoch)

        # Compare the current metric with the current best metric, save the model if the metric improves
        if eval_metrics_best is None:  # first epoch
            eval_metrics_best = {}
            for k, v in eval_metrics.items():
                if "train" in k:
                    continue
                eval_metrics_best[k] = v
            eval_metrics_best = eval_metrics
            print("Save the 1st epoch as best model")
            model.save_networks("best")
        else:
            # eval_metrics is an ordered dict
            counter = 0
            total_counter = 0  # exclude training metrics
            # Iterate over the metric names, and save the best model if the metric improves (in terms of at least half of the metric names)
            for k, v in eval_metrics.items():
                if "train" in k:
                    # print(f"skip training metrics {k}")
                    continue
                total_counter += 1
                if any(x in k for x in ["LPIPS", "AE", "MSE", "SIFID"]):
                    if v < eval_metrics_best[k]:
                        counter += 1
                else:  # PSNR, SSIM
                    assert any(x in k for x in ["PSNR", "SSIM"]), "The metric name should be PSNR or SSIM"
                    if v > eval_metrics_best[k]:
                        counter += 1

            if counter >= total_counter // 2:
                eval_metrics_best = eval_metrics
                print("Save the current best model at epoch " + str(epoch))
                model.save_networks("best")

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            model.save_networks("latest")
            model.save_networks(epoch)

        epoch_time = time.time() - epoch_start_time
        print("End of epoch %d / %d \t Time Taken: %d sec" % (epoch, opt.n_epochs + opt.n_epochs_decay, epoch_time))
        # Record the training time for each epoch.
        visualizer.plot_epoch_time(epoch, epoch_time)

        if opt.train_for_each_epoch:
            model.update_learning_rate()  # update learning rates at the end of every epoch.

        # For Pix2PixHD
        ### instead of only training the local enhancer, train the entire network after certain iterations
        if hasattr(opt, "niter_fix_global"):
            if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
                model.module.update_fixed_params()

        torch.cuda.empty_cache()  # empty cache after each epoch

print("End of training. Takes {}".format(time.time() - training_start_time))
