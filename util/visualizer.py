import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import pickle
import pandas as pd
import json
import skimage.io as io

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False, save_raw_gxgy=False, padded_size=1800, save_raw_arr_vis=False, save_style_image_name=False, style_image_name=None):
    """
    Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    if save_style_image_name:
        assert style_image_name is not None, "style_image_name is None"
        name += f"_style_{style_image_name}"

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}

    # save numpy array for gx and gy
    if save_raw_gxgy:
        # save both generated fake_gx, fake_gy and generated and real gx, gy patches, to verify 3D reconstruction
        keyword_dict = {}
        for label, im_data in visuals.items():
            if 'gx' in label or 'gy' in label:
                arr = util.tensor2arr(im_data)
                keyword_dict[label] = arr

        os.makedirs(os.path.join(image_dir, 'fake_gxgy_raw'), exist_ok=True)
        np.savez(os.path.join(image_dir, 'fake_gxgy_raw', 'fake_gxgy_raw.npz'), **keyword_dict)


    # save png images
    for label, im_data in visuals.items():
        if 'patch_coords' in label: # shape (N,2) (center_x, center_y) patch_size = 32
            # save the coords as json file
            print("find patch_coords, save as pickle file, label is", label)
            json_path = os.path.join(image_dir, label+'.json')

            coords_list = im_data.astype(int)
            # coords_data = {"coords":[{'x': x, 'y': y} for x, y in coords_list]}
            xs = coords_list[:,0].tolist()
            ys = (1536-coords_list[:,1]).tolist() # the website image is anchored at bottom left corner, so need to flip y
            coords_data = {"coords": {'x': xs, 'y': ys, "len": len(xs)}}
            # print(coords_data)
            with open(json_path, 'w') as f:
                json.dump(coords_data, f)
            continue
        
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)


        # print(f"In visulizer, self.use_html {self.use_html}, self.saved {self.saved}, save_raw_arr_vis {save_raw_arr_vis}, save_result {save_result}")
        if save_raw_arr_vis:
            if 'bb' in label or 'coord' in label:
                # print(f"skip visual label {label}")
                continue
            # 2023-03-05: save raw arr as npy and exr files for gx and gy
            if 'gx' in label or 'gy' in label:
                image_raw_arr = util.tensor2arr(im_data, imtype=np.float32) # datatype float, range (-1,1), shape (C, H, W)
                img_npy_path = save_path.replace('.png', '.npy')
                img_exr_path = save_path.replace('.png', '.exr')
                if image_raw_arr.shape[0] == 1:
                    image_raw_arr = image_raw_arr[0]
                else:
                    if image_raw_arr.shape[0] != 3 and image_raw_arr.shape[2] == 3:
                        # already transposed
                    #    print(f"label {label} already transposed, pass to save image")
                        pass
                    else:
                        assert image_raw_arr.shape[0] == 3, f"label {label} raw arr has abnormal shapeshape {image_raw_arr.shape}"
                        image_raw_arr = image_raw_arr.transpose(1,2,0)
                # print(f"save image {label} to png and exr, shape {image_raw_arr.shape} range {np.min(image_raw_arr)} {np.max(image_raw_arr)} dtype {image_raw_arr.dtype}")
                np.save(img_npy_path, image_raw_arr)
                io.imsave(img_exr_path, image_raw_arr, check_contrast=False) # set check_contrast to False to avoid warning
                # print(f"check saved image {img_path} and raw arr {img_npy_path} and {img_exr_path}")
                # raise ValueError("stop here to check")

    webpage.add_images(ims, txts, links, width=width)

    if use_wandb:
        columns = [key for key, _ in visuals.items()]
        columns.insert(0,'epoch')
        result_table = wandb.Table(columns=columns)
        table_row = ['test']
        ims_dict = {}
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            wandb_image = wandb.Image(image_numpy)
            table_row.append(wandb_image)
            ims_dict[label] = wandb_image

        wandb.log(ims_dict)
        result_table.add_data(*table_row)
        wandb.log({"Result": result_table})


def save_images_org_size(webpage, visuals, image_path, aspect_ratio=None):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    raise ValueError('Ruihan defined this func, need to check compatibility with wandb visualization')
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        print('label',label, 'size', im.shape)
        image_name = '%s/%s_org.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)




class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        if opt.display_id is None:
            self.display_id = np.random.randint(100000) * 10  # just a random display id
        else:
            self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0
        self.ncols = opt.display_ncols

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.plot_data = {}
            self.ncols = opt.display_ncols
            if "tensorboard_base_url" not in os.environ:
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            else:
                self.vis = visdom.Visdom(port=2004,
                                         base_url=os.environ['tensorboard_base_url'] + '/visdom')
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_wandb:
            self.wandb_run = wandb.init(project='SKIT', name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='SKIT')

        # if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            # TODO: check the train and test option settings again to ensure consistency
            # if opt.phase == 'test':
                # create a webpage for viewing the results

        # create dir for saving raw arrays even if we don't use html
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, step=None):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        # print("In display_current_results")

        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0

                # print("Iterate over visuals.items()")
                for label, image in visuals.items():
                    print("label",label)
                    print("image type {} shape {}".format(type(visuals.items()), image.shape))
                    # TODO: hard coder here. real_T can have len > 1
                    if len(image) > 1:
                        image = image[0]
                    # TODO: hard code here.
                    if len(image) == 0 : 
                        # empty list for real_gx, real_gy, fake_gx, fake_gy
                        continue
                    try:
                        image_numpy = util.tensor2im(image)
                        label_html_row += '<td>%s</td>' % label
                        images.append(image_numpy.transpose([2, 0, 1]))
                    except:
                        print(f"fail to procecess image {label}, ")
                    
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, ncols, 2, self.display_id + 1,
                                    None, dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(
                            image_numpy.transpose([2, 0, 1]),
                            self.display_id + idx,
                            None,
                            dict(title=label)
                        )
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()


        if self.use_wandb:
            # print('Using wandb')
            columns = [key for key, _ in visuals.items()]
            # print('wandb columns')
            # print(columns)
            columns.insert(0,'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            # print('wandb log ims_dict')
            # print(ims_dict)
            if step is None:
                self.wandb_run.log(ims_dict)
            else:
                self.wandb_run.log(ims_dict, step=step)

            if epoch != self.current_epoch: # update the table for each epoch
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                # print('wandb log result table')
                # print(result_table)
                if step is None:
                    self.wandb_run.log({"Result": result_table})
                else:
                    self.wandb_run.log({"Result": result_table}, step=step)

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)


            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses, use_visdom=True, step=None):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if len(losses) == 0:
            return

        if self.display_id is None or self.display_id > 0:
            if use_visdom:
                plot_name = '_'.join(list(losses.keys()))

                if plot_name not in self.plot_data:
                    self.plot_data[plot_name] = {'X': [], 'Y': [], 'legend': list(losses.keys())}

                plot_data = self.plot_data[plot_name]
                plot_id = list(self.plot_data.keys()).index(plot_name)

                plot_data['X'].append(epoch + counter_ratio)
                plot_data['Y'].append([losses[k] for k in plot_data['legend']])

                X = np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1)
                Y = np.array(plot_data['Y'])
                try:
                    self.vis.line(
                        X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                        Y=np.array(plot_data['Y']),
                        opts={
                            'title': self.name,
                            'legend': plot_data['legend'],
                            'xlabel': 'epoch',
                            'ylabel': 'loss'},
                        win=self.display_id - plot_id)
                except VisdomExceptionBase:
                    self.create_visdom_connections()
        
        # print(losses)
        if self.use_wandb:
            if step is  None: 
                # use default step index, which increases each time `wandb.log` is called
                self.wandb_run.log(losses)
            else:
                # pass a step index
                self.wandb_run.log(losses, step=step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, t_input):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, computational time: %.3f, data loading time: %.3f, setting input time: %.3f ) \n' % (epoch, iters, t_comp, t_data, t_input)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


    def print_current_metrics(self, epoch, eval_metrics):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            eval_metrics (OrderedDict) -- evaluation metrics stored in the format of (name, float) pairs
        """
        message = '(epoch: {}, evaluation_metrics ) \n'.format(epoch)
        for k, v in eval_metrics.items():
            message += '%s: %.3f ' % (k, v)

        # print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


    def plot_current_metrics(self, eval_metrics,  use_visdom=False, step=None):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            eval_metrics (OrderedDict) -- evaluation metrics stored in the format of (name, float) pairs
        """
        if len(eval_metrics) == 0:
            return
        
        if self.display_id is None or self.display_id > 0:
            if use_visdom:
                raise NotImplementedError("Plot for metrics hasn't been implemented for visdom display")
        
        # print("wandb log metrics")
        # print(eval_metrics)
        if self.use_wandb:
            if step is None:
                self.wandb_run.log(eval_metrics)
            else:
                self.wandb_run.log(eval_metrics, step=step)
    
    def save_current_metrics(self, eval_metrics, return_web_dir=False, save_metrics=True, epoch=None, verbose=False, save_metric_index=False, i=0):
        if epoch is None: epoch = self.opt.epoch
        web_dir = os.path.join(self.opt.results_dir, self.opt.name, '{}_{}'.format(self.opt.phase, epoch))  # define the website directory
        if not os.path.exists(web_dir): os.makedirs(web_dir)
        if save_metrics:
            if save_metric_index: 
                dict_path = os.path.join(web_dir, 'eval_metrics_{}.pkl'.format(i))
            else:
                dict_path = os.path.join(web_dir, 'eval_metrics.pkl')
            print('dump test results to %s'%dict_path)
            with open(dict_path, 'wb') as f:
                pickle.dump(eval_metrics, f)
            if verbose: print("save eval metrics to %s"%dict_path)
        if return_web_dir:
            return web_dir
    

    def plot_epoch_time(self, epoch, epoch_time):
        """display the time taken for each epoch on wandb, if available

        Parameters:
            epoch (int)           -- current epoch
            epoch_time            --float
        """
        if self.use_wandb:
            self.wandb_run.log({"time taken": epoch_time, "epoch": epoch}) # can plot wandb chart with epoch as x-axis label
            