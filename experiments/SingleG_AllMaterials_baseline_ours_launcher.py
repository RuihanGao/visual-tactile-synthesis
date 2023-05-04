import os

from .tmux_launcher import Options, TmuxLauncher

"""
Sample commands: 
training: python -m experiments SingleG_AllMaterials_baseline_ours launch 0
testing: python -m experiments SingleG_AllMaterials_baseline_ours launch_test 0
Replace the index 0 by any other number or 'all' to run multiple experiments at once.
Use `launch` to run multiple training scripts at once and `launch_test` to run multiple testing scripts
"""


class Launcher(TmuxLauncher):
    def common_options(self):
        """
        Set options for each individual experiment, one command for one experiment.
        Then functions `commands` and `test_commands` run the training and testing, respectively.
        """

        option_list = []

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
                    name=f"{material}_sinskitG_baseline_ours",
                    use_wandb="",
                    model="sinskitG",
                    dataroot=f"./datasets/singleskit_{material}_padded_1800_x1/",
                    netD="multiscale",
                    netD2="multiscale",
                    # gpu_ids=-1, # use CPU
                    # data_len=3, # set to a small number for debugging
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
                    "400" in s for s in os.listdir(checkpoint_dir)
                ), f"no suitable checkpoint exists for {train_material} model"
                epoch = 400  # use 400 epoch checkpoint if best checkpoint does not exist
            opt.set(epoch=epoch)
            # opt.set(gpu_ids=-1) # use CPU
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
