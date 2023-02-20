import csv
import os
import re
from collections import OrderedDict
from datetime import datetime
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

DL_PATH = "C:/Users/" + os.environ["USERNAME"] + "/Downloads/"
IMG_DATASET_PATH = "C:/Users/" + os.environ["USERNAME"] + "/IntelliJProjects/brailleBlockTracking/dataset/img"
PYTHON_PROJECT_PATH = "C:/Users/" + os.environ["USERNAME"] + "/PycharmProjects/tactilePavingDetectionResultGenerator"

OUTPUT_DIR = "output"

IC_SEQ_NUMBER = {1, 2, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20}
OB_SEQ_NUMBER = {6, 15, 17, 18, 19, 20}
DE_SEQ_NUMBER = {3, 4, 6, 8, 9, 10, 11, 14, 18, 19, 20}
SH_SEQ_NUMBER = {1, 2, 3, 5, 6, 10, 11, 12, 16, 17, 18}
CV_SEQ_NUMBER = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20}
FM_SEQ_NUMBER = {1, 4, 5, 6, 7, 8, 9, 12}

# chart setting
WRITE_PDF = True
USE_COLOR_CYCLE = False
VALUE_SORT = False

LEGEND_LOC = 'upper right'  # 'upper right', 'lower left'

COLOR_CYCLES = ["pink", "red", "blue", "green", "orange", "purple", "black"]
DASHED_LINE_KEY = ""


def main():
    print("1. Create Average Success Ratio Chart")
    print("2. Create Attribute Chart")
    print("3. Create Multi Chart")
    print("4. Create Average Success Ratio & Attribute Chart")
    print("5. Create Many Multi Chart")
    match input("Input > "):
        case "1":
            create_average_success_ratio_chart(filedialog.askdirectory(initialdir=PYTHON_PROJECT_PATH))
        case "2":
            create_attribute_chart(filedialog.askdirectory(initialdir=PYTHON_PROJECT_PATH))
        case "3":
            create_multi_chart_from_success_ratio(filedialog.askdirectory(initialdir=PYTHON_PROJECT_PATH))
        case "4":
            directory_path = filedialog.askdirectory(initialdir=PYTHON_PROJECT_PATH)
            create_average_success_ratio_chart(directory_path)
            create_attribute_chart(directory_path)
        case "5":
            directory_path = filedialog.askdirectory(initialdir=PYTHON_PROJECT_PATH)
            for f in os.listdir(directory_path):
                create_multi_chart_from_success_ratio(os.path.join(directory_path, f), date=f)
        case _:
            print("Invalid input")


def create_multi_chart_from_success_ratio(directory_path, date=None):
    # Get all the ".csv" files in the "input" directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

    # Sort the list
    csv_files.sort()

    # Get Success Ratio for csv files
    success_ratio_dict = {}
    for csv_file in csv_files:
        success_ratio_dict[csv_file.replace(".csv", "")] = read_success_ratio_array(
            os.path.join(directory_path, csv_file))

    # Create the chart
    create_chart(success_ratio_dict=success_ratio_dict, date=date)


def create_attribute_chart(directory_path):
    # Get all the ".csv" files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

    # Get sequence from file name
    sequence_dict = {}
    sequence_folders = os.listdir(IMG_DATASET_PATH)

    for sequence_folder in sequence_folders:
        sequence_images = os.listdir(IMG_DATASET_PATH + "/" + sequence_folder)
        for sequence_image in sequence_images:
            sequence_dict[sequence_image] = sequence_folder

    # Get Iou for csv files by Attribute
    ic_iou_list = []
    ob_iou_list = []
    de_iou_list = []
    sh_iou_list = []
    cv_iou_list = []
    fm_iou_list = []

    for csv_file in csv_files:
        with open(directory_path + "/" + csv_file, "r") as f:
            # Skip the first line
            legend = [s.strip() for s in f.readline().split(",")]
            try:
                iou_column = legend.index("iou")
            except ValueError:
                try:
                    iou_column = legend.index("success_ratio")
                except ValueError:
                    iou_column = legend.index("hsvIou")

            for line in f.readlines():
                try:
                    iou = float(line.split(",")[iou_column])
                except ValueError:
                    iou_column = 1
                    iou = float(line.split(",")[iou_column])

                img_name = line.split(",")[0]
                if ".jpg" not in img_name:
                    img_name += ".jpg"

                sequence = sequence_dict[img_name]

                if int(sequence) in IC_SEQ_NUMBER:
                    ic_iou_list.append(iou)
                if int(sequence) in OB_SEQ_NUMBER:
                    ob_iou_list.append(iou)
                if int(sequence) in DE_SEQ_NUMBER:
                    de_iou_list.append(iou)
                if int(sequence) in SH_SEQ_NUMBER:
                    sh_iou_list.append(iou)
                if int(sequence) in CV_SEQ_NUMBER:
                    cv_iou_list.append(iou)
                if int(sequence) in FM_SEQ_NUMBER:
                    fm_iou_list.append(iou)

    # Get average iou
    ic_average_iou = sum(ic_iou_list) / len(ic_iou_list)
    ob_average_iou = sum(ob_iou_list) / len(ob_iou_list)
    de_average_iou = sum(de_iou_list) / len(de_iou_list)
    sh_average_iou = sum(sh_iou_list) / len(sh_iou_list)
    cv_average_iou = sum(cv_iou_list) / len(cv_iou_list)
    fm_average_iou = sum(fm_iou_list) / len(fm_iou_list)

    # Sort the list
    ic_iou_list.sort()
    ob_iou_list.sort()
    de_iou_list.sort()
    sh_iou_list.sort()
    cv_iou_list.sort()
    fm_iou_list.sort()

    # Reverse the order of the list
    ic_iou_list.reverse()
    ob_iou_list.reverse()
    de_iou_list.reverse()
    sh_iou_list.reverse()
    cv_iou_list.reverse()
    fm_iou_list.reverse()

    # Get the average success ratio
    ic_success_ratio_array = get_success_ratio_array(ic_iou_list, " IC[{:.3f}]".format(ic_average_iou))
    ob_success_ratio_array = get_success_ratio_array(ob_iou_list, " OBS[{:.3f}]".format(ob_average_iou))
    de_success_ratio_array = get_success_ratio_array(de_iou_list, " DEF[{:.3f}]".format(de_average_iou))
    sh_success_ratio_array = get_success_ratio_array(sh_iou_list, " SHA[{:.3f}]".format(sh_average_iou))
    cv_success_ratio_array = get_success_ratio_array(cv_iou_list, " CVA[{:.3f}]".format(cv_average_iou))
    fm_success_ratio_array = get_success_ratio_array(fm_iou_list, " FM[{:.3f}]".format(fm_average_iou))

    attr_success_ratio_dict = {"IC[{:.3f}]".format(ic_average_iou): ic_success_ratio_array,
                               "OBS[{:.3f}]".format(ob_average_iou): ob_success_ratio_array,
                               "DEF[{:.3f}]".format(de_average_iou): de_success_ratio_array,
                               "SHA[{:.3f}]".format(sh_average_iou): sh_success_ratio_array,
                               "CVA[{:.3f}]".format(cv_average_iou): cv_success_ratio_array,
                               "FM[{:.3f}]".format(fm_average_iou): fm_success_ratio_array}

    # Create the chart
    create_chart(success_ratio_dict=attr_success_ratio_dict)


def create_average_success_ratio_chart(directory_path):
    # Get all the ".csv" files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

    # Get Iou for csv files
    all_iou_list = []
    for csv_file in csv_files:
        with open(directory_path + "/" + csv_file, "r") as f:
            # Skip the first line
            legend = [s.strip() for s in f.readline().split(",")]
            try:
                iou_column = legend.index("iou")
            except ValueError:
                try:
                    iou_column = legend.index("success_ratio")
                except ValueError:
                    iou_column = legend.index("hsvIou")

            for line in f.readlines():
                try:
                    all_iou_list.append(float(line.split(",")[iou_column]))
                except IndexError:
                    iou_column = 1
                    all_iou_list.append(float(line.split(",")[iou_column]))

    # Get average iou
    average_iou = sum(all_iou_list) / len(all_iou_list)

    # Sort the list
    all_iou_list.sort()

    # Reverse the order of the list
    all_iou_list.reverse()

    # Get the average success ratio
    success_ratio_array = get_success_ratio_array(all_iou_list, " [{:.3f}]".format(average_iou))

    # Create the chart
    create_chart(success_ratio_array=success_ratio_array)


def create_chart(success_ratio_array=None, success_ratio_dict=None, date=None):
    if date is None:
        # Get now data and time
        now = datetime.now()
        date = now.strftime("%Y-%m-%d_%H-%M-%S")

    success_ratio_array = np.array(success_ratio_array)

    if success_ratio_dict is None:
        if USE_COLOR_CYCLE:
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", COLOR_CYCLES)

        plt.rcParams['font.family'] = 'Times New Roman'

        fig = plt.figure()
        ax = fig.subplots()

        ax.plot(success_ratio_array[:, 0], success_ratio_array[:, 1], linestyle="dashed")
    else:
        # if the first letter is number, sort by first letter of key
        if list(success_ratio_dict.keys())[0][0].isdigit():
            success_ratio_dict = OrderedDict(sorted(success_ratio_dict.items(), key=lambda t: t[0][0]))
        # When key contains "\d\. \d{1,3}" in key (using regular expression), sort by this number
        elif VALUE_SORT:
            success_ratio_dict = OrderedDict(sorted(success_ratio_dict.items(),
                                                    key=lambda t: float(re.search("\d\.\d{1,3}", t[0]).group()),
                                                    reverse=True))
        elif re.search("\d\.\d{1,3}", list(success_ratio_dict.keys())[0]):
            success_ratio_dict = OrderedDict(sorted(success_ratio_dict.items(),
                                                    key=lambda t: float(re.search("\d\.\d{1,3}", t[0]).group()),
                                                    reverse=True))

            average_key = {k for k, v in success_ratio_dict.items() if "(" not in k}
            average_key = average_key.pop()
            success_ratio_dict.move_to_end(average_key, last=False)
        # If the first letter is not number, sort by key
        else:
            success_ratio_dict = OrderedDict(sorted(success_ratio_dict.items(), key=lambda t: t))

        color_cycle = []
        for key, value in success_ratio_dict.items():
            if "IC" in key:
                color_cycle.append("blue")
            elif "OBS" in key:
                color_cycle.append("green")
            elif "DEF" in key:
                color_cycle.append("red")
            elif "SHA" in key:
                color_cycle.append("orange")
            elif "CVA" in key:
                color_cycle.append("purple")
            elif "FM" in key:
                color_cycle.append("pink")
            else:
                color_cycle.append("black")

        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_cycle)

        if USE_COLOR_CYCLE:
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", COLOR_CYCLES)

        plt.rcParams['font.family'] = 'Times New Roman'

        fig = plt.figure()
        ax = fig.subplots()

        for i, (key, value) in enumerate(success_ratio_dict.items()):
            if key[0].isdigit():
                key = key[1:]

            success_ratio_array = np.array(value)

            if (DASHED_LINE_KEY == "" and i == 0) or (DASHED_LINE_KEY in key and DASHED_LINE_KEY != ""):
                ax.plot(success_ratio_array[:, 0], success_ratio_array[:, 1], linestyle="dashed", label=key)
            else:
                ax.plot(success_ratio_array[:, 0], success_ratio_array[:, 1], label=key)
            ax.legend(loc=LEGEND_LOC)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Success ratio by threshold", fontsize=16)
    ax.set_xlabel("Overlap threshold", fontsize=14)
    ax.set_ylabel("Success ratio", fontsize=14)
    ax.grid(True)

    # Save the chart
    suffix = ".jpg"
    if WRITE_PDF:
        suffix = ".pdf"
    fig.savefig(OUTPUT_DIR + "/" + date + " Chart" + suffix, dpi=300)
    print("Done")


def read_success_ratio_array(csv_file):
    success_ratio_array = []
    with open(csv_file, "r") as f:
        # Skip the first line
        f.readline()

        for line in f.readlines():
            success_ratio_array.append([float(line.split(",")[0]), float(line.split(",")[1])])

    return success_ratio_array


def get_success_ratio_array(iou_list, csv_additional_name):
    success_ratio_array = []

    # Get now data and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M-%S")

    chat_range = 0.1
    chat_step = 0.0001

    with open(OUTPUT_DIR + "/" + date + csv_additional_name + " Success Ratio.csv", 'w', newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["threshold", "success_ratio"])

        for i in range(0, int((chat_range / chat_step)) + 1):
            threshold = i / (chat_range / chat_step)
            count = 0

            for iou in iou_list:
                if iou >= threshold:
                    count += 1
                else:
                    break

            success_ratio = float(count) / len(iou_list)
            success_ratio_array.append([threshold, success_ratio])
            writer.writerow([threshold, success_ratio])

    return success_ratio_array


if __name__ == '__main__':
    main()
