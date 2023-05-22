import argparse
import os
import sys
import pickle
import re
import pandas as pd
from tabulate import tabulate

# append parent directory to system path
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
print(f"check parent directory: {parent_directory}")
sys.path.append(parent_directory)
from utils import create_log_dir_by_date, upload_df_to_GoogleSheet


def retrieve_final_epoch(subdir, latest_epoch=400):
    """
    Retrieve the final epoch for a given subdir. If the subdir contains a 'best' checkpoint, return 'best'. Otherwise, return the latest epoch.

    Args:
        subdir (str): directory of the model's test results.
        latest_epoch (int, optional): the maximum number of epochs the model has been trained. Defaults to 400.

    Returns:
        epoch (str): the final epoch for a given subdir.
    """

    if any("best" in s for s in os.listdir(subdir)):
        epoch = "best"
    else:
        assert any(
            str(latest_epoch) in s for s in os.listdir(subdir)
        ), f"no suitable checkpoint exists for subdir {subdir}"
        epoch = str(latest_epoch)
    return epoch


def compile_metrics_for_exp(
    all_subdirs,
    phase,
    model_base_names=[],
    sheetName=None,
    latest_epoch=400,
    average_over_materials=True,
    num_materials=10,
    column_names=[
        "Method",
        "m_I_PSNR",
        "m_I_SSIM",
        "m_I_LPIPS",
        "m_I_SIFID",
        "m_T_LPIPS",
        "m_T_SIFID",
        "m_T_AE",
        "m_T_MSE",
    ],
    num_decimal_avg=3,
    num_decimal_single=4,
    results_dir="results",
    upload_GoogleSheet=True,
    json_keyfile_name="skit_GoogleCloud.json",
    spreadsheetName="CSV-to-Google-Sheet",
    verbose=False,
):
    """
    Compile evaluation metrics for an experiment. For single-object model, we search for different object based on the model_base_names, and compute the average metric for the method.

    Args:
        all_subdirs (list): list of all subdirs in the results folder. The list of directories are used to filter to retrieve related metrics.
        phase (str): 'test' or 'train'. It indicates the phase of the metrics.
        model_base_names (list, optional): list of string that model names should contain. Defaults to []. We retrieve the metrics for each model name in the list and average the metrics to get a summarized statistics.
        sheetName (str, optional): sheet name for local csv file and online Google Spreadsheet. Defaults to None.
        latest_epoch (int, optional): latest epoch for retrieving the metrics. Defaults to 400.

        average_over_materials (bool, optional): option to compute average over object models of the same method. Defaults to True.
        num_materials (int, optional): number of object models to average. Defaults to 10.
        column_names (list, optional): column names of the metric dataframe. Defaults to ['Method','m_I_PSNR', 'm_I_SSIM', 'm_I_LPIPS','m_I_SIFID', 'm_T_LPIPS', 'm_T_SIFID', 'm_T_AE', 'm_T_MSE']
        num_decimal_avg (int, optional): number of decimal places. Defaults to 3.
        num_decimal_single (int, optional): number of decimal places for a single object model, set to a higher number than num_decimal_avg for higher precision. Defaults to 4.
        results_dir (str, optional): directory to save the csv results. Defaults to "results".

        upload_GoogleSheet (bool, optional): option to upload to Google sheet. Defaults to True.
        json_keyfile_name (str, optional): json key file name. Defaults to 'skit_GoogleCloud.json'. Replace it by your credential file name here.
        spreadsheetName (str, optional): name for the Goole SpreadSheet. Defaults to 'CSV-to-Google-Sheet'.
        verbose (bool, optional): option to print logs. Defaults to False.

    Returns:
        df (pd.DataFrame): dataframe of the metrics of each model. If sheetName is not None, also upload the dataframe to Google sheet as csv file.
    """

    # The results are sorted based on model names in `results` folder. Model names represent both the method name and the object model.
    # Based on the method we want to test, we iterate through all_subdirs in `results` folder to retrieve the eval metrics for each object model and average the metrics to get a summarized statistics.
    subdirs = []
    for model_base_name in model_base_names:
        # iterate through the list of model_base_names and find the model names that contain at least one string from the list of model_base_names
        model_base_name_subdirs = [
            f for f in all_subdirs if re.match(model_base_name + "$", f.split("/")[-1])
        ]  # add a $ sign to match the exact string at the end of the string
        if verbose:
            print(f"looking for subdirs for {model_base_name}: {model_base_name_subdirs}")
        subdirs.extend(model_base_name_subdirs)

    # Create an empty dataframe to gather metrics of different object models.
    df = pd.DataFrame(columns=column_names)

    # Iterate through each model and retrieve the metrics
    for subdir in subdirs:
        epoch = retrieve_final_epoch(subdir, latest_epoch=latest_epoch)
        test_epoch = f"{phase}_{epoch}"
        test_epoch_dir = os.path.join(subdir, test_epoch)

        assert os.path.exists(test_epoch_dir), f"Cannot find {test_epoch_dir}"
        # Load the metrics
        dict_path = os.path.join(subdir, test_epoch, "eval_metrics.pkl")
        assert os.path.exists(dict_path), f"Cannot find eval_metrics.pkl in {test_epoch_dir}"
        with open(dict_path, "rb") as f:
            eval_dict = pickle.load(f)

        # Parse the metrics
        method = subdir.split("/")[-1]
        eval_dict["Method"] = method
        for col in column_names:
            if col not in eval_dict.keys():
                eval_dict[col] = "NaN"
            if col != "Method":
                eval_dict[col] = round(float(eval_dict[col]), num_decimal_single)
        eval_dict_filterd = {k: v for k, v in eval_dict.items() if k in column_names}
        df2 = pd.DataFrame(eval_dict_filterd, index=[0], columns=column_names)
        # concatenate the current object model's metrics to the dataframe
        df = pd.concat([df, df2], ignore_index=True)

    # Average the metrics over different object models
    if average_over_materials:
        assert (
            len(df) == num_materials
        ), f"len(df) should be the same as number of materials {num_materials}, but is {len(df)} \n In df, there are methods \n {df['Method'].to_string(index=False)}"

        df_mean = df[column_names[1:]].mean().to_frame().T
        df_mean.loc[df_mean.index[0], "Method"] = "_".join(
            model_base_names[0].split("_")[1:]
        )  # assueme the first string is material, the rest represent the method.
        # append the average metrics to the dataframe
        df = pd.concat([df, df_mean], ignore_index=True)

    # Convert datatypes to float and round the metrics
    metric_list = column_names[1:]  # exclude "Method column"
    df[metric_list] = df[metric_list].apply(pd.to_numeric)
    df = df.round(num_decimal_avg)

    # Assign new worksheet name
    sheetName = "Results" if sheetName is None else sheetName

    # Create a result dir based on the date and save the dataframe as a local csv file
    results_date_dir = create_log_dir_by_date(log_dir=results_dir)
    os.makedirs(results_date_dir, exist_ok=True)
    csv_path = os.path.join(results_date_dir, f"results_{sheetName}.csv")
    df.to_csv(csv_path)
    print(f"Save the metrics dataframe to {csv_path}")

    # Upload the metric to Google Sheet
    if upload_GoogleSheet:
        upload_df_to_GoogleSheet(
            df,
            sheetName=sheetName,
            json_keyfile_name=json_keyfile_name,
            spreadsheetName=spreadsheetName,
            verbose=verbose,
        )
        print(f"Successfully uploaded the dataframe to Google sheet {spreadsheetName} -> {sheetName}")

    # Return the dataframe
    return df


def main(args):
    """
    Parse the argument and retrieve the metrics for the experiment.
    The metrics will be printed out, saved in local csv file in args.results_dir, and optionally uploaded to Google sheet.

    Args:
        args (argparse.Namespace): argument parser

    """
    # List of object models. There are 20 object in our dataset. For single-object model, we average the metrics over 20 objects.
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

    model_base_names = [f"{material}{args.method_postfix}" for material in materials]
    all_subdirs = [f.path for f in os.scandir(args.results_dir) if f.is_dir()]
    df = compile_metrics_for_exp(
        all_subdirs=all_subdirs,
        phase=args.phase,
        model_base_names=model_base_names,
        sheetName=args.sheetName,
        average_over_materials=True,
        num_materials=len(materials),
        column_names=[
            "Method",
            "m_I_LPIPS",
            "m_I_SIFID",
            "m_T_LPIPS",
            "m_T_SIFID",
        ],
        results_dir=args.results_dir,
        upload_GoogleSheet=args.upload_GoogleSheet,
    )

    if args.print_df:
        # tabulate the dataframe for printing
        pdtabulate = lambda df: tabulate(df, headers="keys")
        print(pdtabulate(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="directory to save the results")
    parser.add_argument("--phase", type=str, default="test", help="phase of the metrics")
    parser.add_argument(
        "-n",
        "--sheetName",
        type=str,
        default="ours",
        help="sheet name for local csv file and online Google Spreadsheet",
    )
    parser.add_argument(
        "-m", "--method_postfix", type=str, default="_sinskitG_baseline_ours", help="postfix of the method name"
    )
    parser.add_argument("--upload_GoogleSheet", type=bool, default=False, help="option to upload to Google sheet")
    parser.add_argument(
        "--print_df", type=bool, default=True, help="option to print the metrics in tabulated dataframe"
    )
    args = parser.parse_args()
    main(args)
