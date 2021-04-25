#Importing all the required packages
#Testing - Working Fine
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np

def main():
    # Reading all files and converting them into a dataframe
    x = pd.read_csv('data/2010-2012_all.csv', sep='\t')
    y = pd.read_csv('data/2012-2014_all.csv', sep='\t')
    z = pd.read_csv('data/2014-2018_all.csv', sep='\t')
    k = pd.read_csv('data/2018-2020_all.csv', sep='\t')

    # Combining all the dataframes into one
    dfs = [x, y, z, k]
    #Combining data from all the files
    df = pd.concat(dfs)
    # Dropping duplicates
    dfc = df.drop_duplicates(keep='first')
    # Keeping the required columns
    sg = dfc[['event_date', 'event_starttime', 'sg_shape', 'hpc_bbox']]
    # Inserting unique id to rows
    sg.insert(loc=0, column='sg_id', value=np.arange(len(sg)))
    #Setting column names
    sg.columns = ['sg_id', 'sg_date', 'sg_time', 'sg_shape', 'sg_bbox']
    #Resetting index
    sg = sg.reset_index(drop=True)
    #Exporting the data
    sg.to_csv('data/hek_sigmoid_data.csv', sep='\t', index=False)

if __name__ == "__main__":
    main()