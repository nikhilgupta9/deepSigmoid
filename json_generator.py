#Importing the required packages
import pandas as pd
import json

def main():

    #Reading the cleaned data files
    im = pd.read_csv('data/finalImage_1K.csv', sep='\t')
    sg = pd.read_csv('data/finalSigmoid_1K.csv', sep='\t')

    #Dropping wavelength column from im dataframe
    im = im.drop(columns=['image_wavelength'])

    #Inserting new column as Filename
    im['file_name'] = im.filePath_downsize_image.apply(lambda x: x.split('/')[-1])


    #Changing the column names and inserting one column 'license' with all the values equal to 1
    im.columns = ['id', 'date_captured', 'image_path','width', 'height', 'file_name']
    im['license'] = 1

    #Changing column names as per the JSON file
    df = im[['id', 'date_captured', 'image_path', 'file_name','width', 'height', 'license']]

    #Saving the info as dictionary in images
    images = df.to_dict('records')

    #Introducing a new column with value 0
    sg['iscrowd'] = 0

    sg['category_id'] = 1
    #sg['category_id'] = sg.sg_shape.apply(lambda x: 1 if x=="Forward-S Sigmoid" else 2)

    #ndf = sg[['sg-id','category_id', 'image-id','iscrowd','diagonal-points', 'bbox_area', 'segmentation']]
    ndf = sg[['sg_id','category_id', 'image_id','iscrowd','diagonal-points', 'bbox_area']]

    #ndf.columns = ['id', 'category_id', 'image_id','iscrowd', 'bbox', 'area','segmentation']
    ndf.columns = ['id', 'category_id', 'image_id','iscrowd', 'bbox', 'area']


    #ndff = ndf[['id', 'category_id', 'image_id','iscrowd', 'area', 'bbox','segmentation']]
    ndff = ndf[['id', 'category_id', 'image_id','iscrowd', 'area', 'bbox']]


    from ast import literal_eval
    ndff['bbox'] = ndff['bbox'].apply(lambda x: literal_eval(x))


    annotations = ndff.to_dict('records')

    info =  {
            "contributor": "DMLab",
            "date_created": "2020-04-02",
            "description": "Sigmoid Dataset HEK",
            "status": "Publicly available",
            "author": "Nikhil Gupta",
            "author_email": "ngupta9@gsu.edu",
            "author_webpage": "https://github.com/nikhilgupta9",
            "Lab": "http://dmlab.cs.gsu.edu/",
            "version": "0.1",
            "year": 2020
        }

    categories = [
      {
        "id": 1,
        "name": "S",
        "supercategory": "Sigmoid"
      }
    ]

    licenses =  [
            {
                "id": 1,
                "name": "GPL License",
                "url": "https://www.gnu.org/licenses/gpl.html"
            }
    ]


    a = {"info": info, "categories": categories, "licenses": licenses, "images": images, "annotations": annotations}

    with open("train.json", "w") as write_file:
        json.dump(a, write_file, indent=1)

if __name__ == "__main__":
    main()