import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import myhtml
import util.util as util
from util.visualizer import Visualizer
import pickle
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
import pandas as pd


"""
General-purpose test script for visual-tactile synthesis. 

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.
It first creates model and dataset given the option and then runs inference for --num_test images and save results to an HTML file
See options/base_options.py and options/test_options.py for more test options.
"""


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test option
    # hard-code some base option parameters for test
    opt.num_threads = 0   # set to 0 to disable multi-process data loading. otherwise, may incur errors
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file and/or wandb.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('The number of test images = %d' % dataset_size)
    
    # Initialize a model
    model = create_model(opt)      # create a model given opt.model and other options

    # Initialize a visualizer for testing script
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    if opt.use_wandb:
        wandb_run = wandb.init(project='SKIT', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='SKIT')
    
    # Configure style encoding
    if opt.model == 'skitG':
        save_style_image_name = True
        style_image_name = opt.test_style_material
    else:
        save_style_image_name = False
        style_image_name = None

    # Run inference
    for i, data in enumerate(dataset):
        if i == 0:
            torch.cuda.empty_cache()
            model.setup(opt) # regular setup: load and print networks; create schedulers
            if len(opt.gpu_ids) > 0: model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test: # only apply our model to opt.num_test images.
            break
        model.set_input(data, phase='test', verbose=False) # unpack data from data loader

        # forward pass
        model.test(timing=True)           

        # update visual results
        visuals = model.get_current_visuals()  # get image results
        visualizer.display_current_results(visuals, epoch=opt.epoch, save_result=True)

        # update metrics
        eval_metrics = model.get_current_metrics() # OrderedDict
        visualizer.print_current_metrics(opt.epoch, eval_metrics)
        if opt.model != 'skitG':
            visualizer.plot_current_metrics(eval_metrics,  use_visdom=False)

        # save the quantitative metrics as a dictionary file
        save_metrics=False if 'edit' in opt.dataroot else True
        save_metric_index = True if opt.model == 'skitG' else False
        web_dir = visualizer.save_current_metrics(eval_metrics, return_web_dir=True, save_metrics=save_metrics, save_metric_index=save_metric_index, i=i)
        # save sample images
        webpage = myhtml.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        img_path = model.get_image_paths() # get image paths
        save_images(webpage, visuals, img_path, width=opt.display_winsize, use_wandb=opt.use_wandb, save_raw_gxgy=True, save_raw_arr_vis=opt.save_raw_arr_vis, save_style_image_name=save_style_image_name, style_image_name=style_image_name)

    if opt.model == 'skitG':
        # Since the metrics are computed for each material, we need to compute the mean of the metrics.
        # Load the metrics dictionary file for each material and compute the mean as a summary
        metric_list = []
        for i in range(len(dataset)):
            dict_path = os.path.join(web_dir, 'eval_metrics_{}.pkl'.format(i))
            metric_i = pickle.load(open(dict_path, 'rb'))
            metric_list.append(metric_i)

        # compute mean
        df = pd.DataFrame(metric_list)
        mean_metrics = dict(df.mean())
        print(mean_metrics)
        dict_path = os.path.join(web_dir, 'eval_metrics.pkl')
        print('dump test results to %s'%dict_path)
        with open(dict_path, 'wb') as f:
            pickle.dump(mean_metrics, f)
        visualizer.plot_current_metrics(mean_metrics,  use_visdom=False)

    webpage.save()  # save the HTML

    print('End of testing!')
