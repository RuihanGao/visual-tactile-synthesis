import os
from lib2to3.pgen2.literals import evalString
from optparse import Option

from .tmux_launcher import Options, TmuxLauncher

"""
This launcher runs the pix2pix experiments for all materials. Used as comparison to the SingleG_AllMaterials_baseline_ours_gpu_idser.py
"""


class Launcher(TmuxLauncher):
    def common_options(self):
        """
        Set options for each individual experiment, one command for one experiment.
        Then functions `commands` and `test_commands` run the training and testing, respectively.
        """

        option_list = []
        dataset = "data14"
        materials = [
            "BlackJeans",
            "BluePants",
            "BlueSports",
            "BrownVest",
            "ColorPants",
            "ColorSweater",
            "DenimShirt",
            "FlowerJeans",
            "FlowerShorts",
            "GrayPants",
            "GreenShirt",
            "GreenSkirt",
            "GreenSweater",
            "GreenTee",
            "NavyHoodie",
            "PinkShorts",
            "PurplePants",
            "RedShirt",
            "WhiteTshirt",
            "WhiteVest",
        ]

        for material in materials:
            option_list.append(
                Options(
                    name=f"{material}_spade_baseline",
                    # use_wandb="",
                    model="spade",
                    dataset="patchskit",
                    dataroot=f"./datasets/singleskit_{material}_padded_1800_x1/",
                )
            )

        return option_list

    def commands(self):
        return ["python train.py " + str(common_opt) for common_opt in self.common_options()]

    def test_commands(self):
        option_list = []
        for opt in self.common_options():
            train_material = opt.kvs["name"].split("_")[0]
            # print(f"check whether best checkpoint exists for {train_material} model")
            checkpoint_dir = os.path.join("checkpoints", opt.kvs["name"])
            if any("best" in s for s in os.listdir(checkpoint_dir)):
                print(f"best checkpoint exists for {train_material} model")
                epoch = "best"
            else:
                assert any(
                    "100" in s for s in os.listdir(checkpoint_dir)
                ), f"no suitable checkpoint exists for {train_material} model"
                epoch = 100  # use 400 epoch. some are trained to 400, some are trained to 500. take the minimum
            opt.set(epoch=epoch)
            opt.set(gpu_ids=-1)  # spade is too large to test on 1 gpu
            option_list.append(
                "python test.py "
                + str(
                    opt.set(
                        eval="",
                        preprocess="none",
                        data_len=1,
                        num_touch_patch_for_logging=100,
                        batch_size_G2=100,
                        save_raw_arr_vis=True,
                    )
                )
            )

        return option_list
