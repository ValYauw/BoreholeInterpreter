import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import sqlite3
from sqlite3 import Error

from typing import List, Tuple, Dict, Union, Iterable, Optional, Literal, Type

import numba
import numpy as np 
import pandas as pd

from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, ScalarFormatter
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from textwrap import fill

class CPT_SoilClassification():

    connection = None

    def __init__(self, 
        filepath: Optional[str] = None
        ):
        """
        Initializes an object of the class CPT_SoilClassification, a helper class to the class CPT.

        Args:
            soil_classification_method  (string)    Sets the name of the soil classification method to follow.
            filepath                    (string)    The filepath to an external database file to define a custom soil classification method.
        """
        if not filepath is None:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"{filepath}\nDatabase not found.")
            self.connection = self._connect_to_soil_classification_db(filepath)
    
    @classmethod
    def _parallel_is_point_in_polygon(self, 
        points_x: np.ndarray, 
        points_y: np.ndarray, 
        polygon: np.ndarray
        ) -> pd.Series:
        """
        Checks whether each point is in the polygon, in parallel for a list of points.

        Args:
            points_x    (np.ndarray)
                An nx1 Numpy array containing the data to check.
            points_y    (np.ndarray)
                An nx1 Numpy array containing the data to check.
            polygon     (np.ndarray)
                An nx2 Numpy array containing x- and y-coordinates, representing a closed polygon.
        
        Returns:
            Numpy array
                Contains boolean values that represent whether or not each point is inside the polygon.
        """

        shp_polygon = Polygon(polygon)
        shp_polyline = LineString(polygon)

        if points_x.size != points_y.size:
            raise IndexError("x and y of points must be equal in size.")

        D = np.empty(points_x.size, dtype=bool) 
        for i in numba.prange(0, len(D)):
            x = points_x[i]
            y = points_y[i]
            # Quick check: either x/y value is NaN
            if np.isnan(x) or np.isnan(y):
                D[i] = False
            else:
                shp_point = Point(x,y)
                is_within = shp_point.within(shp_polygon)
                if not is_within:
                    is_within = shp_polyline.distance(shp_point) < 1e-8
                D[i] = is_within
        return D

    @classmethod
    def _connect_to_soil_classification_db(self, db_filepath: str = None):
        """
        Private class method. Connects to the default database file if filepath is not specified. 
        
        Returns an SQLite connection object upon successful connection.
        """
        if db_filepath is None:
            db_filepath = SCRIPT_DIR + "\\" + 'CPT_SOIL_CLASSIFICATION.db'
        connection = sqlite3.connect(db_filepath)
        return connection

    @classmethod
    def _execute_read_query(self, connection, query):
        """
        Private class method. Executes a read query.
        """
        cursor = connection.cursor()
        result = None
        cursor.execute(query)
        result = cursor.fetchall()
        return result

    @classmethod
    def calculate_soil_classification(self, 
        method: Literal[
            'Eslami Fellenius', 
            'Robertson et al 1986', 
            'Robertson et al 1986 (nonpiezo)', 
            'Robertson 1990'
        ], 
        CPT_point: 'CPT'
        ):
        """
        Determine the soil classification zone of the CPT test point at each depth point.

        Args:
            method              (string)
                The name/label of the soil classification method to choose.
                Default database has 4 choices:
                 - Eslami Fellenius                 fs vs. qE
                 - Robertson et al 1986             Bq vs. qt, Rf vs. qt
                 - Robertson et al 1986 (nonpiezo)  Rf vs. qt/qc
                 - Robertson 1990                   Fr vs. Qt, Bq vs. Qt
                You may also choose a custom soil classification method if connecting to an external database.
            CPT_point           (CPT)
                An object of the CPT class.
                If choosing a custom soil classification method, the attributes to be compared against the graph must already be defined.
                For example, define an attribute "Ic" if the custom soil classification method uses Ic.
        
        Returns:
            Pandas Dataframe representing the soil zone number and soil zone description of each point.
        """
        if not hasattr(self, "connection") or self.connection is None:
            self.connection = self._connect_to_soil_classification_db()        

        # Check if the soil classification method has a defined record in the database
        does_method_exist = len(self._execute_read_query(
            self.connection, 
            f"""SELECT method_name FROM methods 
            WHERE method_name='{method}';"""
        )) > 0
        if not does_method_exist:
            raise Error(f"No soil classification method by the label '{method}' exists in the given database.")

        # Define general soil classification method parameters
        table_methods = self._execute_read_query(
            self.connection, 
            f"""SELECT param_x1, param_y1, param_x2, param_y2, 
            is_x1_on_log_scale, is_y1_on_log_scale, 
            is_x2_on_log_scale, is_y2_on_log_scale
            FROM methods WHERE method_name='{method}';"""
        )
        (param_x1, param_y1, param_x2, param_y2, 
         *logarithmic_scale) = table_methods[0]
        is_one_graph_only = param_x2 is None
        number_graphs = 1 if is_one_graph_only else 2

        # Define dictionary to lookup soil zone description and USCS
        table_soilzone_names = self._execute_read_query(
            self.connection, 
            f"""SELECT zone_no, description, USCS FROM zone_names 
            WHERE method_name='{method}';"""
        )
        dict_soilzone_names = {rw[0]: (rw[1], rw[2]) for rw in table_soilzone_names}

        # Sets the x- and y- data range
        compare_data = []
        if method == "Eslami Fellenius":
            compare_data.append((
                CPT_point.fs,
                CPT_point.qt - CPT_point.u2 / 1000 if hasattr(CPT_point, "_u2") else CPT_point.qt,
                bool(logarithmic_scale[0]),
                bool(logarithmic_scale[1])
            ))
        else:
            compare_data.append((
                CPT_point.__getattribute__(param_x1),
                CPT_point.__getattribute__(param_y1),
                bool(logarithmic_scale[0]),
                bool(logarithmic_scale[1])
            ))
            if not is_one_graph_only:
                compare_data.append((
                    CPT_point.__getattribute__(param_x2),
                    CPT_point.__getattribute__(param_y2),
                    bool(logarithmic_scale[2]),
                    bool(logarithmic_scale[3])
                ))

        # Initialize soil classification dataframe
        df_soil_zones = pd.DataFrame(
            index=CPT_point.qc.index, 
            columns=[
                "Soil Zone Number", "Soil Zone Description", "USCS",
                "X_graph_1", "Y_graph_1", "X_graph_2", "Y_graph_2"
            ],
            data={
                "X_graph_1":compare_data[0][0],
                "Y_graph_1":compare_data[0][1]
            },
            dtype=np.float32
        )
        if not is_one_graph_only:
            df_soil_zones["X_graph_2"] = compare_data[1][0]
            df_soil_zones["Y_graph_2"] = compare_data[1][1]
        df_soil_zones[df_soil_zones.columns[:3]] = \
            df_soil_zones[df_soil_zones.columns[:3]].astype('string')
        #print("IS NA:", pd.isna(df_soil_zones.iloc[0, 0]))

        # Determine the min & max boundaries of the zone definitions
        recalc_segment = None
        for i in range(number_graphs):
            table_bounds_graph = self._execute_read_query(
                self.connection, 
                f"""SELECT zone_no, MIN(x), MAX(x), MIN(y), MAX(y)
                FROM zone_definitions
                WHERE method_name='{method}' AND graph={i+1}
                GROUP BY zone_no;"""
            )
            dict_bounds_graph = {rw[0]: rw[1:] for rw in table_bounds_graph}
            x_data, y_data, is_x_on_log_scale, is_y_on_log_scale = compare_data[i]

            for k,v in dict_bounds_graph.items():
                
                # Does a quick check to see if the given point is out-of-bounds of the current soil zone
                bool_series_inbounds = (pd.isna(df_soil_zones["Soil Zone Number"])) & \
                    (x_data >= v[0]) & (x_data <= v[1]) & \
                    (y_data >= v[2]) & (y_data <= v[3])
                if not bool_series_inbounds.any():
                    continue
                
                # Filter the inbound data coordinates
                process_x_data = x_data[bool_series_inbounds].to_numpy()
                process_y_data = y_data[bool_series_inbounds].to_numpy()

                # Process the coordinates                
                if is_x_on_log_scale:
                    process_x_data = np.log10(process_x_data)
                if is_y_on_log_scale:
                    process_y_data = np.log10(process_y_data)
                
                # Define the soil zone definitions
                table_soilzone_coordinates = self._execute_read_query(
                    self.connection, 
                    f"""SELECT x, y FROM zone_definitions
                    WHERE method_name='{method}' 
                    AND zone_no='{k}'
                    AND graph={i+1};
                    """
                )
                table_soilzone_coordinates = np.array(table_soilzone_coordinates)

                # Process the coordinates
                if is_x_on_log_scale:
                    table_soilzone_coordinates[:, 0] = np.log10(table_soilzone_coordinates[:, 0])
                if is_y_on_log_scale:
                    table_soilzone_coordinates[:, 1] = np.log10(table_soilzone_coordinates[:, 1])

                # Get a numpy boolean array to determine whether each point is in the soil zone polygon
                bool_is_in_polygon = \
                    self._parallel_is_point_in_polygon(
                    process_x_data, process_y_data, table_soilzone_coordinates
                )

                # Special case for method Robertson et al 1986: Save the portion with calculated
                # soil zone number Zone 9,10,11,12
                if method == "Robertson et al 1986" and k == "9,10,11,12":
                    recalc_segment = df_soil_zones.loc[bool_series_inbounds]
                    recalc_segment = recalc_segment.loc[bool_is_in_polygon].index
                    continue    # Continue without storing

                # Finally store in the dataframe
                df_soil_zones["Soil Zone Number"].loc[bool_series_inbounds] = \
                    np.where(bool_is_in_polygon, k, pd.NA)
                df_soil_zones["Soil Zone Description"].loc[bool_series_inbounds] = \
                    np.where(bool_is_in_polygon, dict_soilzone_names[k][0], pd.NA)
                df_soil_zones["USCS"].loc[bool_series_inbounds] = \
                    np.where(bool_is_in_polygon, dict_soilzone_names[k][1], pd.NA)
        
        # Special case for method Robertson et al 1986: If the soil zone number in the
        # previously saved portion, calculated using the second graph is not part of
        # Zones 9, 10, 11, or 12, save as "9,10,11,12"
        if method == "Robertson et al 1986" and recalc_segment is not None:
            df_soil_zones[df_soil_zones.columns[0]].loc[recalc_segment] = \
                df_soil_zones[df_soil_zones.columns[0]].where(
                    df_soil_zones["Soil Zone Number"].isin(["9", "10", "11", "12"]),
                    "9,10,11,12"
                )
            df_soil_zones[df_soil_zones.columns[1]].loc[recalc_segment] = \
                df_soil_zones[df_soil_zones.columns[1]].where(
                    df_soil_zones["Soil Zone Number"].isin(["9", "10", "11", "12"]),
                    "Zone 9,10,11,12"
                )
            df_soil_zones[df_soil_zones.columns[2]].loc[recalc_segment] = \
                df_soil_zones[df_soil_zones.columns[2]].where(
                    df_soil_zones["Soil Zone Number"].isin(["9", "10", "11", "12"]),
                    ""
                )

        # Store the computed soil classification into private class properties
        CPT_point._soil_classification_method = method
        CPT_point._soil_classification = df_soil_zones[df_soil_zones.columns[:3]]
        CPT_point._soil_classification_graph_data = df_soil_zones[df_soil_zones.columns[3:]]
        
        return df_soil_zones[df_soil_zones.columns[:3]]

    @classmethod
    def query_legend_soilzones(self, 
        method: Literal[
            'Eslami Fellenius', 
            'Robertson et al 1986', 
            'Robertson et al 1986 (nonpiezo)', 
            'Robertson 1990'
        ]):
        table_soilzone_names = self._execute_read_query(
            self.connection, 
            f"""SELECT zone_no, description, USCS, colour FROM zone_names 
            WHERE method_name='{method}';"""
        )
        dict_soilzone_names = {rw[0]: (rw[1], rw[2], rw[3]) for rw in table_soilzone_names}
        return dict_soilzone_names

    @classmethod
    def _plot_empty_graph(self,
        method: Literal[
            'Eslami Fellenius', 
            'Robertson et al 1986', 
            'Robertson et al 1986 (nonpiezo)', 
            'Robertson 1990'
        ]
        ):
        """
        Plots an unpopulated soil classification graph for the given method.

        Args:
            method      (string)
                The label of the soil classification method.
                Default database has 4 choices:
                 - Eslami Fellenius                 fs vs. qE
                 - Robertson et al 1986             Bq vs. qt, Rf vs. qt
                 - Robertson et al 1986 (nonpiezo)  Rf vs. qt/qc
                 - Robertson 1990                   Fr vs. Qt, Bq vs. Qt
                You may also choose a custom soil classification method if connecting to an external database.
        """

        # Check if the soil classification method has a defined record in the database
        does_method_exist = len(self._execute_read_query(
            self.connection, 
            f"""SELECT method_name FROM methods 
            WHERE method_name='{method}';"""
        )) > 0
        if not does_method_exist:
            raise Error(f"No soil classification method by the label '{method}' exists in the given database.")

        # Define general soil classification method parameters
        table_methods = self._execute_read_query(
            self.connection, 
            f"""SELECT param_x1_axistext, param_y1_axistext, 
            param_x2_axistext, param_y2_axistext, 
            is_x1_on_log_scale, is_y1_on_log_scale, 
            is_x2_on_log_scale, is_y2_on_log_scale
            FROM methods WHERE method_name='{method}';"""
        )
        axis_text = table_methods[0][:4]
        logarithmic_scale = table_methods[0][4:]
        is_one_graph_only = axis_text[2] is None
        number_graphs = 1 if is_one_graph_only else 2

        # Get the names of each soil classification zone
        dict_soilzone_names = self.query_legend_soilzones(method)

        # Get the x- and y- limits
        table_bounds_graph = self._execute_read_query(
            self.connection, 
            f"""SELECT graph, MIN(x), MAX(x), MIN(y), MAX(y)
            FROM zone_definitions
            WHERE method_name='{method}'
            GROUP BY graph;"""
        )                
        dict_bounds_graph = {rw[0]: rw[1:] for rw in table_bounds_graph}

        # Get the coordinates of each soil zone, for each graph
        soilzone_graph_coords = []
        for i in range(number_graphs):
            dict_soilzone_coords = {}
            for k,v in dict_soilzone_names.items():
                table_soilzone_coordinates = self._execute_read_query(
                    self.connection, 
                    f"""SELECT x, y FROM zone_definitions
                    WHERE method_name='{method}' 
                    AND zone_no='{k}' 
                    AND graph={i+1};
                    """
                )
                dict_soilzone_coords[k] = np.array(table_soilzone_coordinates)
            soilzone_graph_coords.append(dict_soilzone_coords)

        fig = plt.figure(figsize=(15, 8), dpi=72)
        subfigs = fig.subfigures(1, 2, width_ratios=[0.85, 0.15])
        axs = subfigs[0].subplots(1, number_graphs)

        
        labels = [fill(v[0], 20) for k,v in dict_soilzone_names.items()]
        dict_soilzone_lineformat = {
            k: {
                'linestyle':'', 
                'marker':'o', 'markersize':7, 
                'markerfacecolor':v[2],
                'markeredgecolor':'black',
                'markeredgewidth':0.5
            }
            for k,v in dict_soilzone_names.items()
        }
        handles = [
            Line2D(
                [0, 1], [0, 0],
                **dict_soilzone_lineformat[k]
            ) 
            for k in dict_soilzone_lineformat.keys()
        ]

        for i in range(number_graphs):
            ax = axs if number_graphs == 1 else axs[i]
            for k,v in dict_soilzone_names.items():
                verts = soilzone_graph_coords[i][k]
                if len(verts) == 0:
                    continue
                codes = np.ones(verts.shape[0], int) * Path.LINETO
                codes[0] = Path.MOVETO
                codes[-1] = Path.CLOSEPOLY
                ax.add_patch(PathPatch(
                    Path(verts, codes),
                    fill=False,
                    #facecolor=v[2],
                    edgecolor='black',
                    #alpha=0.5,
                    linewidth=1.5
                ))
            ax.set_xscale('log' if logarithmic_scale[i*2] else 'linear')
            ax.set_yscale('log' if logarithmic_scale[i*2+1] else 'linear')
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel(axis_text[i*2])
            ax.set_ylabel(axis_text[i*2+1])
            ax.set_xlim(*dict_bounds_graph[i+1][:2])
            ax.set_ylim(*dict_bounds_graph[i+1][2:])
            ax.grid(which='minor', axis='both', color='#DBDBDB')
            ax.grid(which='major', axis='both', color='#515151')
            ax.set_axisbelow(True)
        
        subfigs[1].legend(
            handles, labels, 
            title="LEGEND", loc='upper left',
            fontsize='medium',
            bbox_to_anchor=(0.0, 0.9)
        )
        
        subfigs[0].subplots_adjust(
            left=0.1, top=0.9, bottom=0.1, right=0.95, wspace=0.25
        )
        subfigs[0].suptitle(
            f"CPT Soil Classification: {method}", 
            verticalalignment='top', horizontalalignment='center'
            )

        return (
            fig, 
            [axs] if is_one_graph_only else axs, 
            dict_soilzone_lineformat
        )

    @classmethod
    def plot_graph(self, 
        CPT_Point: 'CPT' = None,
        method: Optional[Literal[
            'Eslami Fellenius', 
            'Robertson et al 1986', 
            'Robertson et al 1986 (nonpiezo)', 
            'Robertson 1990'
        ]] = None
        ):
        """
        Plot the soil classification graph.
        """

        if method is None and CPT_Point is None:
            print("Nothing to plot.")
            return

        if method is None:
            if CPT_Point is None:
                raise ValueError(f"A soil classification method has not been chosen.")
            else:
                if not hasattr(CPT_Point, "_soil_classification_method"):
                    raise ValueError(f"Soil classification has not been computed for {CPT_Point.pointID}")
            method = CPT_Point._soil_classification_method

        if not CPT_Point is None and method != CPT_Point._soil_classification_method:
            self.calculate_soil_classification(method, CPT_Point)
        if not hasattr(CPT_Point, "_soil_classification"):
            self.calculate_soil_classification(method, CPT_Point)
        df_soil_classification = CPT_Point._soil_classification
        df_soil_classification_graph = CPT_Point._soil_classification_graph_data
        
        fig, axs, dict_soilzone_lineformat = self._plot_empty_graph(method)
        for i in range(len(axs)):
            ax = axs[i]
            for k,v in dict_soilzone_lineformat.items():
                bool_filter_df = df_soil_classification["Soil Zone Number"] == k
                ax.plot(
                    df_soil_classification_graph[f"X_graph_{i+1}"].loc[bool_filter_df],
                    df_soil_classification_graph[f"Y_graph_{i+1}"].loc[bool_filter_df],
                    label=k, **v
                )

        return fig
    
