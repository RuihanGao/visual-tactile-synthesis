import glob
import os

from .tmux_launcher import Options, TmuxLauncher

"""
Test our method on DALLE generated sketches.
"""

class Launcher(TmuxLauncher):
    def common_options(self):
        """
        Set options for each individual experiment, one command for one experiment.
        Then functions `commands` and `test_commands` run the training and testing, respectively.
        """

        option_list = []

        # Run the current best model on new materials
        train_materials = [
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

        for material in train_materials:
            # iterate through multiple DALLE sketch modified from the same material sketch
            # for DALLE_material_dataset in glob.glob(f"./datasets/{sketch_material}_DALLE*"):
            for DALLE_material_dataset in glob.glob(f"./datasets/DALLE_1800_*"):
                print(f"Test {material} model with {DALLE_material_dataset} sketches")
                option_list.append(
                    Options(
                        name=f"{material}_sinskitG_baseline_ours",
                        # use_wandb="",
                        model="sinskitG",
                        dataroot=DALLE_material_dataset,
                    )
                )

        return option_list

    def commands(self):
        return ["python train.py " + str(common_opt) for common_opt in self.common_options()]

    def test_commands(self):
        # return ["python test.py " + str(opt.set(eval="", preprocess='none', data_len=1, num_touch_patch_for_logging=100, batch_size_G2=100)) for opt in self.common_options()]
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
                    "400" in s for s in os.listdir(checkpoint_dir)
                ), f"no suitable checkpoint exists for {train_material} model"
                epoch = 400  # use 400 epoch. some are trained to 400, some are trained to 500. take the minimum
            opt.set(epoch=epoch)
            # opt.set(gpu_ids=-1)
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
