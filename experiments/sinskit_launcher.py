from lib2to3.pgen2.literals import evalString
from .tmux_launcher import Options, TmuxLauncher


##### Method 1. Define options for each experiment, and set training and testing commonds together #####
"""
Adopted from pretrained_launcher.py
"""

class Launcher(TmuxLauncher):

    def common_options(self):
        """
        Set options for each individual experiment, one command for one experiment. 
        Then functions `commands` and `test_commands` run the training and testing, respectively.
        """
        return [
            # Commmand 0, for end-to-end baseline
            Options(
                dataroot="./datasets/singleskit_BlackShorts/",
                name="BlackShorts_0512baseline_e2e",
                train_S2I=True,
                train_I2=True,
            ),

            # Commmand 1, for end-to-end baseline, feeding sketch to I2T model
            Options(
                dataroot="./datasets/singleskit_BlackShorts/",
                name="BlackShorts_0512baseline_e2e_S2T",
                train_S2I=True,
                train_I2=True,
                feed_S2T=True,
            ),
            

            # Commmand 2, for S2I baseline
            Options(
                dataroot="./datasets/singleskit_BlackShorts/",
                name="BlackShorts_0512baseline_S2I",
                train_S2I=True,
                train_I2=False,
            ),

            # Commmand 3, for I2T baseline
            Options(
                dataroot="./datasets/singleskit_BlackShorts/",
                name="BlackShorts_0512baseline_I2T",
                train_S2I=False,
                train_I2=True,
            ),

        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        # TODO: check whether the syntax `set(eval)` works here.
        return ["python test.py " + str(opt.set(eval)) for opt in self.common_options()]




##### Method 2. separate common options and options for individual command #####
    # # List of training commands
    # def commands(self):
    #     opt = Options()

    #     # common options for all training sessions defined in this launcher
    #     opt.set(dataroot="~/datasets/cityscapes/",  # specify --dataroot option here
    #             model="contrastive_cycle_gan",
    #             pool_size=0,
    #             no_dropout="",
    #             init_type="xavier",
    #             batch_size=1,
    #             display_freq=400,
    #             evaluation_metrics="fid,cityscapes",
    #             evaluation_freq=10000,
    #             direction="BtoA",
    #             use_recommended_options="",
    #             nce_idt_freq=0.1,
    #             )

    #     # Specify individual options here
    #     commands = [

    #         # first command.
    #         # This command can be run using python -m experiments placeholder run 0
    #         # It will output python train.py [OPTIONS], where OPTIONS are everything defined in the variable opt
    #         "python train.py " + str(opt.clone().set(
    #             name="cityscapes_placeholder_noidt",  # name of experiments
    #             nce_idt=False,
    #         )),

    #         # second command.
    #         # This command can be run using python -m experiments placeholder run 1
    #         # It removes the option --nce_idt_freq 0.1 that was defined by our common options
    #         "python train.py " + str(opt.clone().set(
    #             name="cityscapes_placeholder_singlelayer",
    #             nce_layers="16",
    #         ).remove("nce_idt_freq")),


    #         # third command that performs multigpu training
    #         # This command can be run using python -m experiments placeholder run 2
    #         "python train.py " + str(opt.clone().set(
    #             name="cityscapes_placeholder_multigpu",
    #             nce_layers="16",
    #             batch_size=4,
    #             gpu_ids="0,1",
    #         )),

    #     ]

    #     return commands

    # # This is the command used for testing.
    # # They can be run using python -m experiments placeholder run_test $i
    # def test_commands(self):
    #     opt = Options()
    #     opt.set(dataroot="~/datasets/cityscapes_unaligned/cityscapes",
    #             model="contrastive_cycle_gan",
    #             no_dropout="",
    #             init_type="xavier",
    #             batch_size=1,
    #             direction="BtoA",
    #             epoch=40,
    #             phase='train',
    #             evaluation_metrics="fid",
    #             )

    #     commands = [
    #         "python test.py " + str(opt.clone().set(
    #             name="cityscapes_nce",
    #             nce_layers="0,8,16",
    #             direction="BtoA",
    #         )),
    #     ]

    #     return commands
