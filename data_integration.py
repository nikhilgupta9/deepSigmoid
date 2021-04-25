# Importing the required packages
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import datetime
from sunpy.io import jp2
import numpy as np
import os
import sys
from PIL import Image


def header(file_path):
    """
    This function takes the JP2 file path as input and returns the spatial header info of that file.
    :param file_path: full path of the .jp2 file
    :return: spatial header info
    """
    h = jp2.get_header(file_path)
    header = h[0]
    image_w = header['NAXIS1']
    image_h = header['NAXIS2']
    center_x = header['CRPIX1']
    center_y = header['CRPIX2']
    scale_x = header['CDELT1']
    scale_y = header['CDELT2']
    header_spatial_info = [image_w, image_h, center_x, center_y, scale_x, scale_y]
    return header_spatial_info

def main():

    # Loading the sigmoid and aia-images data into dataframes
    sg = pd.read_csv('data/hek_sigmoid_data.csv', sep='\t')
    aia = pd.read_csv('data/aia_image_data.csv', sep='\t')

    # Dropping duplicates if any
    sg = sg.drop_duplicates()
    aia = aia.drop_duplicates()

    # Sigmoid temporary key for join
    sg['Date'] = pd.to_datetime(sg['sg_time'])
    sg['sgtempkey'] = pd.to_datetime(sg['Date']).dt.to_period('D')

    # AIA temporary key for join
    aia['aiatempkey'] = pd.to_datetime(aia['image_time']).dt.to_period('D')

    # Setting index and joining
    aia.index = aia.aiatempkey
    sg.index = sg.sgtempkey
    df = sg.join(aia, how='left')

    # Converting sg-time to datetime object
    df['sg_time'] = pd.to_datetime(df['sg_time'])

    # Converting image-time to datetime object
    df['image_time'] = pd.to_datetime(df['image_time'])

    # Calculating time difference between times in seconds
    df['dl'] = (df['sg_time'] - df['image_time']).astype('timedelta64[s]')

    # Calculating the absolute time difference
    df['dl'] = df.dl.abs()

    # Keeping rows which have time delta less than or equal to 3600
    keep = df.loc[df.dl <= 3600]

    # Keeping the closest value of abs for every sigmoid
    y = keep.sort_values("dl").groupby("sg_id", as_index=False).first()

    # Converting image-id to integer
    y['image_id'] = y['image_id'].astype(np.int64)

    # ndf = y[['sg-id','sg-date','sg-time','sg-shape','image-id']]
    # data = y[y['sg-time'].dt.year == 2012]
    # Extracting the header info from JP2 files

    y['header'] = y['image_path'].progress_apply(lambda x: header(x))

    # Expanding the header information extracted from files to convert them to pixel coordinates
    y[['original_w', 'original_h', 'center_x', 'center_y', 'scale_x', 'scale_y']] = pd.DataFrame(y.header.tolist(),
                                                                                                 index=y.index)

    y.to_csv('data/mapped_sigmoid_data.csv', sep='\t', index=False)

if __name__ == "__main__":
    main()