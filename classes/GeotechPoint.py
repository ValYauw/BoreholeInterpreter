import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classes.LEGEND_STRATIGRAPHY import LEGEND_COLOUR

from typing import List, Tuple, Dict, Optional, Iterable, TypeVar, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib.patches import Patch, PathPatch
from matplotlib.lines import Line2D
from matplotlib.path import Path

from math import ceil
import copy

class PointDataset(pd.DataFrame):      
    """
    Stores the dataset (e.g. stratigraphy, sampling data) of a point along its depth.
    The dataset is an extension from the Pandas Dataframe object with additional attributes.
    """  
    
    def __str__(self):
        return self.to_string()

    @property
    def depthPoints(self):
        df_filtered = self.loc[self["Depth to"].notna(), ["Depth from", "Depth to"]]
        depthPoints = []
        for idx, rw in df_filtered.iterrows():
            if len(depthPoints) == 0 or depthPoints[-1] != rw[0]:
                depthPoints.append(rw[0])
            depthPoints.append(rw[1])
        return depthPoints
        

class GeotechPoint:
    """
    An ancestor class that stores geotechnical point data.
    """
    
    # Private property to be changed in inheritance
    _pointType = "Geotechnical Point"

    def __init__(self, 
        pointID: str, 
        xCoord: 'Optional[float]' = None, 
        yCoord: 'Optional[float]' = None, 
        elevation: 'Optional[float]' = None
        ):
        """
        Iniatilzes a geotechnical point, e.g. a borehole or a CPT.

        Args:
            pointID     (string)    The ID label/name of the drilled point
            xCoord      (float)     The x-coordinate of the drilled point
            yCoord      (float)     The y-coordinate of the drilled point
            elevation   (Optional[float])
                The elevation of the drilled point 
        """
        self.pointID = str(pointID)
        self.xCoord = float(xCoord) if xCoord is not None else None
        self.yCoord = float(yCoord) if yCoord is not None else None
        if elevation:
            self._elevation = float(elevation)
        self.holedepth = 0
    
    def createDataset(self, 
        arrData: 'Iterable[Iterable[float, float, TypeVar]]', 
        dictDtypes: 'Dict[str, Union[int, float, str]]', 
        allowNan: bool = False
        ) -> PointDataset:
        """
        Factory method for point dataset.

        When this method is called, this function will automatically check the input data for various errors:
         - Invalid depths: The bottom boundary cannot have a smaller depth than the top boundary.
         - Intersecting/overlapping depths: Each point in the dataset must be discrete and not have intersecting/overlapping top/bottom boundaries.
         - Non-numeric value stored as depth
        In addition, this function will conduct typecasting of the input data as specified by the user, e.g. store soil description as string.

        Args:
            arrData     (Iterable[Iterable[float, float, TypeVar]])
                A 2D array-like data structure representing the raw dataset.
                The first and second columns of this argument should coincide with the top and bottom boundaries (in m below ground level) of each point in the dataset, i.e. "Depth from" and "Depth to".
                There must not be an invalid (i.e. None type, a NaN value, or a data type which cannot be converted to float) data in the first column.
                Invalid data (i.e. None type, a NaN value, or a data type which cannot be converted to float) is allowed in the second column.
                The third column until last column should coincide with the actual data value(s), e.g. for a point's stratigraphy this will be the soil type, soil description, and USCS.
            dictDtypes  (Dict[str, Union[int, float, str]])
                A dictionary specifying the data type to cast each column of the raw dataset to.
                The size of this dictionary is two less than the number of columns in arrData. This is because the "Depth from" and "Depth to" columns are typecast as float automatically.
                Where invalid data is detected, that invalid data is either kept as a Numpy NaN value or an empty string.
        
        Returns:
            PointDataset

        """

        arrHeadings = ["Depth from", "Depth to"]
        arrHeadings.extend(dictDtypes.keys())
        arrAssertDataTypes = tuple(dictDtypes.values())
        dictData = {
            arrHeadings[idx]:
                [ arrData[rw][idx] if idx < len(arrData[rw]) else None
                for rw in range(len(arrData)) ]
            for idx in range(len(arrHeadings))
        }

        df = PointDataset(dictData)

        # Assert data types of depth columns
        df["Depth from"] = pd.to_numeric(df["Depth from"], errors='coerce', downcast='float')
        df["Depth to"] = pd.to_numeric(df["Depth to"], errors='coerce', downcast='float')
        if df["Depth from"].isna().any():
            raise ValueError("Dataset must have numeric value for its depths:\t>>\n{}:\n{}".format(
                self.pointID,
                df[df["Depth from"].isna()]
            ))

        # Assert data types of other columns
        if not arrAssertDataTypes is None:
            arrHeadings = list(df.keys())
            for x in range(len(arrAssertDataTypes)):
                assertDataType = arrAssertDataTypes[x]
                curHeading = arrHeadings[x+2]
                if assertDataType == int:
                    df[curHeading] = pd.to_numeric(df[curHeading], errors='coerce', downcast='integer')
                elif assertDataType == float:
                    df[curHeading] = pd.to_numeric(df[curHeading], errors='coerce', downcast='float')
                elif assertDataType == str:
                    df[curHeading] = df[curHeading].astype('string')
                    df[curHeading].fillna("", inplace=True)
                elif assertDataType == 'category':
                    df[curHeading] = df[curHeading].astype('category')
                    df[curHeading].fillna("", inplace=True)
        if not allowNan:
            df_data = df[arrHeadings[2:]]
            if df_data.isna().any().any():
                raise ValueError("Failed datatype assertion for the dataset:\t>>\n{}:\n{}".format(
                    self.pointID,
                    df_data[df_data.isna()]
                ))
            del df_data

        # Sort by depth from 
        df.sort_values(by=["Depth from"], inplace=True)

        # Assert correct depth inputs ('depth to' must be more than 'depth from')
        df_intersectingDepths = df.loc[df["Depth from"] >= df["Depth to"]]

        # Assert correct depth inputs ('depth from' of a datapoint must be more than 'depth to' of the previous datapoint)
        df_parseDepths = df[df["Depth to"].notna()]
        if not df_parseDepths.empty:
            row_labels = list(df_parseDepths.index)
            for x in range(len(row_labels)):
                if x == 0:
                    continue
                if df.loc[row_labels[x], "Depth from"] < df.loc[row_labels[x-1], "Depth to"]:
                    df_intersectingDepths = pd.concat([
                        df_intersectingDepths, 
                        df.loc[row_labels[x-1]].to_frame().T,
                        df.loc[row_labels[x]].to_frame().T
                    ])
        del df_parseDepths
        df_intersectingDepths = df_intersectingDepths.drop_duplicates(inplace=False)

        if not df_intersectingDepths.empty:
            raise ValueError("Dataset must not have intersecting values for its depths:\t>>\n{}:\n{}".format(
                self.pointID,
                df_intersectingDepths
            ))
        del df_intersectingDepths

        return df 

    # Reusable OOP property atrribute methods
    def _getProperty(attr: str):
        def inner(self):
            return self.__dict__[attr]
        return inner
    def _setProperty(attr: str):
        def inner(self, val):
            self.__dict__[attr] = val
        return inner
    def _delProperty(attr: str):
        def inner(self):
            del self.__dict__[attr]
        return inner
    
    @property
    def elevation(self):
        """
        Get the elevation of the geotechnical point.

        Returns:
            Elevation (float)
        """
        if hasattr(self, "_elevation"):
            return self._elevation
        else:
            raise AttributeError(
                f"Elevation of {self.pointID} is not yet defined."
            )
    @elevation.setter
    def elevation(self, elevation):
        """
        Set the elevation of the geotechnical point.

        Args:
            elevation (float)
        """
        self._elevation = elevation

    @property
    def stratigraphy(self) -> PointDataset:
        """
        Get the stratigraphy dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_stratigraphy"):
            return self._stratigraphy
        else:
            raise AttributeError(
                f"Stratigraphy of {self.pointID} is not yet defined."
            )
    @stratigraphy.setter
    def stratigraphy(self, 
        arrStratigraphy: 'Iterable[Iterable[float, float, str, str, str]]'
        ):
        """
        Set the stratigraphy dataset of the geotechnical point.

        Args:
            arrStratigraphy     Iterable[Iterable[float, float, str, str, str]]
                A 2D array-like data structure representing the raw stratigraphy dataset of the point.
                Column 1:   (float)   The top boundary depth of the data point
                Column 2:   (float)   The bottom boundary depth of the data point
                Column 3:   (str)     The soil type as a short description, e.g. CLAY, SILTY SAND, ANDESITE
                Column 4:   (str)     The soil description, as full or brief as wanted
                Column 5:   (str)     The United Soil Classification System (USCS) Symbol, e.g. SW
                Values in columns 3-5 may be left as an empty string.
        """
        dictDtypes = {
            "Soil Type": 'category',
            "Soil Description": str,
            "USCS": 'category'
        }
        df = self.createDataset(arrStratigraphy, dictDtypes, allowNan=True)
        self._stratigraphy = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth

    def merge_datasets(self, 
        arrDatasetsToMerge: Iterable[str],
        ignore_error: bool = False
        ) -> PointDataset:
        """
        Get the merged dataset between two discrete properties, e.g. stratigraphy and consistency/density.

        Args:
            arrDatasetsToMerge  (Iterable[str])
                A Python list/array representing the names of the datasets to merge
        
        Example Input:
            test_point.merge_datasets(["stratigraphy", "consistency_density"])
        
        Returns:
            PointDataset
        """

        for attr in arrDatasetsToMerge:
            if not ignore_error and not hasattr(self, "_" + attr):
                raise AssertionError(
                    f"Dataset {attr} is not found in the object: {self.pointID}"
                )
        
        arrDatasets = [self.__dict__.get("_" + attr, None) for attr in arrDatasetsToMerge]
        arrDatasets = list(filter(lambda x: x is not None, arrDatasets))
        
        # Get information of the columns for each dataset 
        arrDatasetCol = [dataset.columns[2:] for dataset in arrDatasets]
        arrDatasetColSize = [len(arrCols) for arrCols in arrDatasetCol]
        arrDatasetDtypes = [arrDatasets[0].iloc[:,0].dtype, arrDatasets[0].iloc[:,1].dtype]
        mergedDepthPoints = []
        for dataset in arrDatasets:
            arrDatasetDtypes.extend(dataset.iloc[:, 2:].dtypes.copy())

        # Get the depth points of each discrete depth interval
        arrDepthPoints = [pd.Series(dataset.depthPoints) for dataset in arrDatasets]
        mergedDepthPoints = pd.concat(arrDepthPoints)
        mergedDepthPoints.drop_duplicates(inplace=True)
        mergedDepthPoints.sort_values(inplace=True)
        merged_depthFrom = list(mergedDepthPoints.iloc[:-1])
        merged_depthTo = list(mergedDepthPoints.iloc[1:])
        del mergedDepthPoints

        # Create skeleton DataFrame
        mergedColumns = ["Depth from", "Depth to"]
        for arrCols in arrDatasetCol:
            mergedColumns.extend(arrCols)
        mergedDtypes = {mergedColumns[i]: arrDatasetDtypes[i] for i in range(len(mergedColumns))}

        merged_df = PointDataset(
            {"Depth from":merged_depthFrom, "Depth to": merged_depthTo},
            columns=mergedColumns
        )
        del mergedColumns

        # Iterate through each discrete depth interval and each dataset to store the
        # appropriate information in the merged DataFrame
        for i in range(len(merged_depthFrom)):
            d1 = merged_depthFrom[i]
            d2 = merged_depthTo[i]
            for j in range(len(arrDatasets)):
                # Extract data from the specified datasets
                df = arrDatasets[j]
                df_extracted = df.loc[
                    (df['Depth from'] <= d1) & (df['Depth to'] >= d2),
                    arrDatasetCol[j]
                ]
                if df_extracted.empty:
                    continue
                # Store extracted data to the merged DataFrame
                idx_col_start = sum(arrDatasetColSize[:j]) + 2 if j > 0 else 2
                idx_col_end = idx_col_start + arrDatasetColSize[j] - 1
                for k in range(idx_col_start, idx_col_end+1):
                    merged_df.iloc[i, k] = df_extracted.iloc[0, k-idx_col_start]

        # Type casting
        for col_name, dtype in mergedDtypes.items():
            if is_numeric_dtype(dtype):
                #print(col_name, dtype, is_integer_dtype(dtype))
                if is_integer_dtype(dtype):
                    merged_df[col_name] = pd.to_numeric(merged_df[col_name], errors='coerce', downcast='integer')
                else:
                    merged_df[col_name] = pd.to_numeric(merged_df[col_name], errors='coerce', downcast='float')
            else:
                merged_df[col_name] = merged_df[col_name].fillna("")
                merged_df[col_name] = merged_df[col_name].astype(dtype)
        del mergedDtypes
        del arrDatasetDtypes

        return merged_df

    def plot_single_log(self, 
        ax: plt.Axes, 
        plot_by_elevation: bool = False,
        legend_colours: Dict = None,
        style_lookup_colours: Dict = None,
        plot_gwl: bool = True
        ) -> Tuple[plt.Axes, Dict]:
        """
        Draw a visual log out of the geotechnical point's stratigraphy.

        Args:
            ax                      (Matplotlib Axes)
                A Matplotlib Axes object to be passed. 
                This object will be modified.
            plot_by_elevation       (bool)
                Boolean setting to draw the visual log by the elevation or depth.
                (Default: Draw by depth)
            legend_colours          (Dict)
                If working with a collection of boreholes and wanting to draw on the same figure,
                pass this argument to share a legend.
                This object stores the stratigraphy string of each data point, as well as the 
                colour and hatching used to style the Matplotlib patch. 
                These will form legend entries of the figure.
            style_lookup_colours    (Dict)
                Pass this argument to set a custom dictionary with user-defined stratigraphy styling.
                The format of this dictionary must be:
                    Key:    The stratigraphy string corresponding to the style
                    Value:  A 2-sized tuple containing a colour string and a hatching string
                            (format must be readily accepted by Matplotlib API)
                            e.g.    ('black', '')       to style the data segment as a box with 
                                                        solid black fill colour and no hatching
                                    ("#FFF500", '')     to style the data segment as a box with 
                                                        solid fill colour #FFF500 (hexadecimal code)
                                                        and no hatching
                                    ("red", '|')        to style the data segment as a box with 
                                                        solid red fill colour and vertical hatching
        
        Returns:
            Tuple[Matplotlib Axes, Dict]
                A 2-sized tuple containing the modified Axes object, and the dictionary containing
                legend entries.
        """

        legend_colours = legend_colours.copy()

        if style_lookup_colours is None:
            style_lookup_colours = LEGEND_COLOUR
        else:
            custom_style = copy.deepcopy(LEGEND_COLOUR)
            custom_style.update(style_lookup_colours)
            style_lookup_colours = custom_style

        #axes_position = [0.2, 0.1, 0.7, 0.7]
        #ax.set_position(axes_position)

        df = self.stratigraphy
        if plot_by_elevation and not hasattr(self, "_elevation"):
            raise AttributeError(
                f"Elevation of {self.pointID} is not yet defined."
            )
        el = self.elevation if plot_by_elevation else 0

        discrete_stratigraphy_record = df.loc[df["Depth to"].notna()]
        discrete_stratigraphy_record = discrete_stratigraphy_record.reset_index()
        grouped_stratigraphy = discrete_stratigraphy_record.groupby("Soil Type")
        stratigraphy_groups = grouped_stratigraphy.size()

        for stratigraphy, group in grouped_stratigraphy:
            # Determine number of rectangles to draw
            numRectangles = int(stratigraphy_groups[stratigraphy])

            # Determine styling of stratigraphy group
            colour, hatch = ('', '')
            stratigraphy_split = stratigraphy.split()
            for item in stratigraphy_split:
                if item in style_lookup_colours.keys():
                    colour = style_lookup_colours[item][0] if colour == "" else colour
                    hatch += style_lookup_colours[item][1]
            if stratigraphy in ["", "LOSS", "CORE", "CORE LOSS"]:
                colour = "red"
                hatch = ""
                legend_colours["LOSS"] = (colour, hatch)
            elif colour == "":
                colour = "white"
                hatch = ""
                legend_colours["OTHER"] = (colour, hatch)
            else:
                if not stratigraphy in legend_colours.keys():
                    legend_colours[stratigraphy] = (colour, hatch)

            # Determine coordinates of the bottom and top lines (in Numpy array format)
            lines_bottom = el - group["Depth to"] if plot_by_elevation else group["Depth from"]
            lines_top = el - group["Depth from"] if plot_by_elevation else group["Depth to"]
            lines_bottom = lines_bottom.to_numpy()
            lines_top = lines_top.to_numpy()

            # Sets the codes and verts for matplotlib PathPatch
            codes = np.ones(numRectangles * 5, int) * Path.LINETO
            codes[0::5] = Path.MOVETO
            codes[4::5] = Path.CLOSEPOLY
            verts = np.zeros((numRectangles * 5, 2))
            verts[0::5, 0] = -7.0           # Top-left corner
            verts[0::5, 1] = lines_top
            verts[1::5, 0] = 7.0            # Top-right corner
            verts[1::5, 1] = lines_top
            verts[2::5, 0] = 7.0            # Bottom right corner
            verts[2::5, 1] = lines_bottom
            verts[3::5, 0] = -7.0           # Bottom left corner
            verts[3::5, 1] = lines_bottom
            
            # Finally add the patch
            ax.add_patch(PathPatch( Path(verts, codes),
                facecolor=colour,
                edgecolor="black",
                hatch=hatch,
                label=f"{stratigraphy}",
                alpha=0.9))

        # Draw the groundwater level
        if plot_gwl and hasattr(self, "_gwl"):
            df_gwl = self._gwl
            for idx, row in df_gwl.iterrows():
                gwl = el - row["Depth from"] if plot_by_elevation else row["Depth from"]
                ax.add_line(Line2D(
                    [-10.0, 10.0], [gwl, gwl], 
                    linewidth=3.0, linestyle='-', color='deepskyblue'
                ))

        # Format the axes
        major_y_unit, minor_y_unit = (5, 1)
        hole_depth_ceil = (ceil(self.holedepth / major_y_unit) + 1) * major_y_unit
        y_max = 0 if not plot_by_elevation else ceil(el / major_y_unit) * major_y_unit
        y_min = hole_depth_ceil if not plot_by_elevation else y_max - hole_depth_ceil
        ax.set_xlim(-10.0, 10.0)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(major_y_unit))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_y_unit))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.set_ylabel("Elevation" if plot_by_elevation else "Depth, m")
        ax.set_axisbelow(True)
        ax.grid(True, which='major', axis='y', color='black', linestyle='-')
        ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')
        ax.grid(True, which='major', axis='x', color='gray', linestyle='--')
        ax.set_title(self.pointID)

        return (ax, legend_colours)

    def __str__(self):
        xCoord_string = f"\nX:\t\t{'{:,.1f}'.format(self.xCoord)}" if self.xCoord else ""
        yCoord_string = f"\nY:\t\t{'{:,.1f}'.format(self.yCoord)}" if self.yCoord else ""
        elevation_string = "\nElevation:\t" + '{:,.1f}'.format(self.elevation) \
            if hasattr(self, "_elevation") else ""
        return_string = \
            f"""{self._pointType}\nPoint ID :\t{self.pointID}{xCoord_string}{yCoord_string}{elevation_string}"""
        return return_string