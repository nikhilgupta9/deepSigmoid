import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import datetime
from sunpy.io import jp2
import numpy as np
import os
import sys
from PIL import Image

# Convert text to datetime
def timefunc(time):
    t = datetime.datetime.strptime(time, "%Y_%m_%d__%H_%M_%S_%f")
    t2 = datetime.datetime.isoformat(t)
    return t2

def main():
    pathofAIAFile = '/data/AIA/aia.txt'
    wavelength = '131'

    # Importing the file
    df = pd.read_csv(pathofAIAFile, sep=" ", header=None)
    df.columns = ['path']

    # Removing the links which don't have .jp2 extension
    df['isFile'] = df['path'].progress_apply(lambda x: True if 'jp2' in x else False)
    img = df[df.isFile == True]

    # Adding /data/AIA to complete the path
    img['path'] = img['path'].progress_apply(lambda x: x.replace('./', '/data/AIA/'))

    img['wavelength'] = img.path.progress_apply(lambda x: (x.split('SDO_AIA_AIA_')[1]).split('.jp2')[0])

    img['imageid'] = img.path.progress_apply(lambda x: x.split('/')[-1].split('__SDO_AIA')[0].replace('_', ''))

    keep = img.loc[img['wavelength'] == wavelength]

    keep['imagetime'] = keep.path.progress_apply(lambda x: x.split('/')[-1].split('__SDO')[0])

    keep['time'] = keep.imagetime.progress_apply(lambda x: timefunc(x))

    aia = keep[['imageid', 'wavelength', 'path', 'time']]

    aia.columns = ['image_id', 'image_wavelength', 'image_path', 'image_time']

    aia = aia.reset_index(drop=True)

    aia.to_csv('data/aia_image_data.csv', sep='\t', index=False)

if __name__ == "__main__":
    main()