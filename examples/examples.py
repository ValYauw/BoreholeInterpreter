import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classes import Project, Borehole, CPT

import pandas as pd
import csv

from typing import Literal

def open_example(filename: Literal['bh_1', 'bh-2', 'cpt_1']):
    
    if filename == 'bh_1':
        
        example_project = Project()

        dict_points = _create_points(SCRIPT_DIR + '\\' + 'borehole_example_1_points.csv', 'Borehole')

        df_raw = pd.read_csv(SCRIPT_DIR + '\\' + 'borehole_example_1_stratigraphy.csv', 
            delimiter=';'
        )
        df_raw.fillna('', inplace=True)
        df_grouped = df_raw.groupby('ID')
        for ID, data in df_grouped:
            point = dict_points[str(ID)]
            arr_data = data[data.columns[1:]].values.tolist()
            point.stratigraphy = arr_data
        
        df_raw = pd.read_csv(SCRIPT_DIR + '\\' + 'borehole_example_1_sampling.csv', 
            delimiter=';'
        )
        df_raw.fillna('', inplace=True)
        df_grouped = df_raw.groupby('ID')
        for ID, data in df_grouped:
            point = dict_points[str(ID)]
            arr_data = data[data.columns[1:]].values.tolist()
            point.sampling = arr_data
            
        for point in dict_points.values():
            example_project.add(point)
        
        return example_project

    elif filename == 'bh_2':
        
        example_project = Project()

        dict_points = _create_points(SCRIPT_DIR + '\\' + 'borehole_example_2_points.csv', 'Borehole')

        df_raw = pd.read_csv(SCRIPT_DIR + '\\' + 'borehole_example_2_stratigraphy.csv', 
            delimiter=';'
        )
        df_raw.fillna('', inplace=True)
        df_grouped = df_raw.groupby('ID')
        for ID, data in df_grouped:
            point = dict_points[str(ID)]
            arr_data = data[data.columns[1:]].values.tolist()
            point.stratigraphy = arr_data
        
        df_raw = pd.read_csv(SCRIPT_DIR + '\\' + 'borehole_example_2_sampling.csv', 
            delimiter=';'
        )
        df_raw.fillna('', inplace=True)
        df_grouped = df_raw.groupby('ID')
        for ID, data in df_grouped:
            point = dict_points[str(ID)]
            arr_data = data[data.columns[1:]].values.tolist()
            point.sampling = arr_data
            
        for point in dict_points.values():
            example_project.add(point)
        
        return example_project

    elif filename == 'cpt_1':

        example_project = Project()

        dict_points = _create_points(SCRIPT_DIR + '\\' + 'cpt_example_1_points.csv', 'CPT')

        df_raw = pd.read_csv(SCRIPT_DIR + '\\' + 'cpt_example_1_data.csv', delimiter=';')
        df_grouped = df_raw.groupby('ID')
        for ID, data in df_grouped:
            cpt_point = dict_points[str(ID)]
            cpt_point.raw_data = data[data.columns[1:]].values.tolist()
            example_project.add(cpt_point)
        
        return example_project

    else:
        raise ValueError('Please enter a valid example file.')

def _create_points(filename: str, klassname: str):
    klass = globals()[klassname]
    dict_points = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        row_no = 0
        for row in csv_reader:
            row_no += 1
            if row_no == 1:
                continue
            new_point = klass(*row)
            dict_points[row[0]] = new_point
    return dict_points