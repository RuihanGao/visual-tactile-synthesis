# Google sheet API
import csv
import os
import sys
from datetime import datetime

import cv2
import gspread
import matplotlib.pyplot as plt
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials


def create_log_dir_by_date(parent_dir=".", log_dir="logs"):
    """
    Create a log directory based on date.

    Args:
        parent_dir (str, optional): parent directory. Defaults to '.'.
        log_dir (str, optional): log directory. Defaults to 'logs'.

    Returns:
        logs_date_dir (str): log directory based on date.
    """
    today = datetime.today().strftime("%Y-%m-%d")
    logs_date_dir = os.path.join(parent_dir, log_dir, today)
    if os.path.isdir(logs_date_dir) == False:
        os.makedirs(logs_date_dir, exist_ok=True)
    return logs_date_dir


def upload_df_to_GoogleSheet(
    df, sheetName=None, json_keyfile_name="skit_GoogleCloud.json", spreadsheetName="CSV-to-Google-Sheet", verbose=False
):
    """
    Upload the dataframe to Google sheet online .

    Args:
        df (pd.DataFrame): dataframe of the metrics of each model. If sheetName is not None, also upload the dataframe to Google sheet as csv file.
        sheetName (str, optional): sheet name for Google sheet. Defaults to None.
        json_keyfile_name (str, optional): json key file name. Defaults to 'skit_GoogleCloud.json'. Replace it by your credential file name here.
        spreadsheetName (str, optional): name for the Goole SpreadSheet. Defaults to 'CSV-to-Google-Sheet'.
        verbose (bool, optional): option to pring logs. Defaults to False.
    """

    # Upload the dataframe to Google sheet
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]

    # Authorize the client
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_name, scope)
    client = gspread.authorize(credentials)

    # Open a spreadsheet
    spreadsheet = client.open(spreadsheetName)

    # List all worksheets
    worksheet_list = spreadsheet.worksheets()
    if verbose:
        print("all worksheets in the spreadsheet: ", worksheet_list)

    # Create a sheet if it doesn't exist
    if sheetName not in [worksheet.title for worksheet in worksheet_list]:
        print(f"Create a new worksheet {sheetName}")
        worksheet = spreadsheet.add_worksheet(title=sheetName, rows="100", cols="20")

    # Update the value
    spreadsheet.values_update(
        sheetName, params={"valueInputOption": "USER_ENTERED"}, body={"values": list(csv.reader(open(csv_path)))}
    )

    print(f"Successfully uploaded the dataframe to Google sheet {spreadsheetName} -> {sheetName}")


def print_dict(d):
    for k, v in d.items():
        print(k, v)


def print_dict_l2(d):
    for subdic_key in d:
        print("sud_dict key", subdic_key)
        for k, v in d[subdic_key].items():
            print(k, v)


def normalize_im(im):
    arr = np.asarray(im)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr


def equalize_this(image_src, with_plot=False, gray_scale=False, convert2gray=True, clipLimit=2.0, tileGridLength=8):
    if len(image_src.shape) == 3:
        gray_scale = False

    if not gray_scale:
        if convert2gray:
            if np.max(image_src) <= 1:
                image_src = image_src * 255
            image_src = cv2.cvtColor(image_src.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            # image_eq = cv2.equalizeHist(image_src)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridLength, tileGridLength))
            image_eq = clahe.apply(image_src)
            cmap_val = "gray"
        else:
            r_image, g_image, b_image = cv2.split(image_src)
            print("r_image", type(r_image))
            print(r_image.shape, r_image.dtype)

            r_image_eq = cv2.equalizeHist(r_image.astype(np.uint8))
            g_image_eq = cv2.equalizeHist(g_image.astype(np.uint8))
            b_image_eq = cv2.equalizeHist(b_image.astype(np.uint8))

            image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
            cmap_val = None
    else:
        image_eq = cv2.equalizeHist(image_src)
        cmap_val = "gray"

    if with_plot:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text("Original")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_eq, cmap=cmap_val)
        return image_eq
    return image_eq
