import pandas as pd
import numpy as np
import os
import sys
from PIL import Image
import datetime as dt
from tqdm import tqdm
from ast import literal_eval
tqdm.pandas()

#Downsize 4k image
def downsize(x, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    info = []
    image_path = x
    image_name = os.path.basename(os.path.normpath(image_path))
    image_name = image_name[:-4]  # Drop the extension
    image_name = '{0}.png'.format(image_name)  # Add new extension
    image_out_path = os.path.join(output_path, image_name)
    info.append(image_out_path)
    try:
            img = Image.open(image_path)
            img1 = img.resize((new_w, new_h), Image.ANTIALIAS)
            img1.save(image_out_path)
            info.append(new_w)
            info.append(new_h)
    except IOError as err:
            print('\nWhile processing {0}, error caught:{1}'.format(image_path, err))
            skipped.append(image_path)
    return info


#Splitting the bounding-box coordinates into a list
def points(item):
    p = item
    p1 = p[9:]
    p2 = p1[:-2]
    p3 = p2.split(',')
    p4 = [tuple([float(num) for num in item.split(" ")]) for item in p3]
    return p4

#Conversion of boundingpoints to pixelunit
def convert_boundingpoints_to_pixelunit(points, cdelt1, cdelt2, crpix1, crpix2, original_w, shrinkage_ratio):
    if not points:
        return None
    b = [(float(v[0]) / cdelt1 + crpix1, float(v[1]) / cdelt2 + crpix2) for v in points]    # Shrink and then mirror vertically
    b = [(v[0] / shrinkage_ratio, (original_w - v[1]) / shrinkage_ratio) for v in b]
    return b

#Calculating bbox area
def bbox_area(x):
    l = abs(x[0] - x[2])
    b = abs(x[1] - x[3])
    a = l * b
    return a

#X0, Y0, X1, Y1
def get_bbox(np):
    #print(np, '\n', min([item[1] for item in np]))
    res = [min([item[0] for item in np]), min([item[1] for item in np]), max([item[0] for item in np]), max([item[1] for item in np])]
    return(res)

def main():
    #Loading the sigmoid and aia-images data into dataframes
    data = pd.read_csv('mapped_sigmoid_data.csv', sep='\t')

    #Filtering 2012 data
    data['sg_date']= pd.to_datetime(data['sg_date'])
    #df = data[data['sg_date'].dt.year == 2012]
    df = data.copy()

    #Conversion of sg-bbox to pixel co-ordinates
    new_h = 1024
    new_w = 1024
    output_path = 'data/aia_sigmoid_1k_png'
    shrink_ratio = 4

    df['points'] = df['sg_bbox'].progress_apply(lambda x: points(x))

    df['new_points'] = df.progress_apply(lambda x: convert_boundingpoints_to_pixelunit((x['points']), x['scale_x'], x['scale_y'],x['center_x'], x['center_y'], x['original_w'],shrink_ratio),axis=1)

    df['diagonal-points'] = df.new_points.progress_apply(lambda x: get_bbox(x))

    #Calculating bounding box area
    df['bbox_area'] = df['diagonal-points'].progress_apply(lambda x: bbox_area(x))

    #Downsizing the images
    df['downsize']  = df['image_path'].progress_apply(lambda x: downsize(x, output_path))

    #Expanding the columns
    df[['filePath_downsize_image','new_w', 'new_h']] = pd.DataFrame(df.downsize.tolist(), index= df.index)

    #Saving everything to a CSV file
    df.to_csv('sigmoid1K.csv', sep='\t', index=False)

    #Keeping only the required columns
    sigmoid = df[['sg_id', 'sg_date', 'sg_time' , 'sg_shape', 'diagonal-points', 'bbox_area', 'image_id']]
    image = df[['image_id','image_time', 'image_wavelength', 'filePath_downsize_image' , 'new_w', 'new_h']]
    image = image.drop_duplicates()

    #Saving these files to a CSV file
    sigmoid.to_csv('data/finalSigmoid_1K.csv', sep='\t', index=False)
    image.to_csv('data/finalImage_1K.csv', sep='\t', index=False)

if __name__ == "__main__":
    main()