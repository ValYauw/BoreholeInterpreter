import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classes.GeotechPoint import GeotechPoint
from classes.Borehole import Borehole
from classes.CPT import CPT

from typing import List, Tuple, Dict, Union, Iterable, Optional, Literal

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Patch

from math import ceil, floor

import warnings

class Project:

    def __init__(self):
        self.points = {
            "Borehole": {},
            "CPT": {},
            "Other": {}
        }

    @property
    def size(self):
        """
        Return the number of points stored in the project
        """
        num_points = {k: len(v) for k,v in self.points.items()}
        total = sum(num_points.values())
        return total
        
    def __len__(self):
        return self.size
    
    def __iter__(self):
        self.__counter = 0
        self.__kcounter = 0
        return self
    
    def __next__(self):
        if self.__kcounter >= len(self.points):
            raise StopIteration
        access_dict = list(self.points.values())[self.__kcounter]
        if len(access_dict) == 0:
            self.__kcounter += 1
            self.__counter = 0
            return next(self)
        point = list(access_dict.values())[self.__counter]
        self.__counter += 1
        if self.__counter >= len(access_dict):
            self.__kcounter += 1
            self.__counter = 0
        return point
    
    def get_list_of_all_points(self):
        dict_point_names = {
            "Point ID": [],
            "Test Type": [],
            "X": [],
            "Y": [],
            "Elevation": [],
            "Hole Depth": []
        }
        
        for k,v in self.points.items():
            if len(v) == 0:
                continue
            dict_point_names["Point ID"].extend(list(v.keys()))
            dict_point_names["Test Type"].extend([k for i in v])
            dict_point_names["X"].extend(
                [point_obj.xCoord for name, point_obj in v.items()]
            )
            dict_point_names["Y"].extend(
                [point_obj.yCoord for name, point_obj in v.items()]
            )
            dict_point_names["Elevation"].extend(
                [point_obj.elevation if hasattr(point_obj, "_elevation") else None
                for name, point_obj in v.items()]
            )
            dict_point_names["Hole Depth"].extend(
                [point_obj.holedepth for name, point_obj in v.items()]
            )
        df_points = pd.DataFrame(data=dict_point_names)
        return df_points
    
    def add(self, 
        point: Union[GeotechPoint, Borehole, CPT]
        ) -> Union[GeotechPoint, Borehole, CPT]:
        """
        Add a geotechnical point object to a Project
        """

        klass = point.__class__.__name__
        dict_key = klass if klass != 'GeotechPoint' else 'Other'

        store_to_dict = self.points[dict_key]
        ID = point.pointID
        if ID in store_to_dict:
            raise IndexError(
                f"A geotechnical point already exists in the project with the name {ID}."
            )
        store_to_dict.update({ID: point})

        return point

    def get(self, point_id: str):
        """
        Get the geotechnical point object with the specified label.
        """
        for k,v in self.points.items():
            if point_id in v:
                return v[point_id]
        print(f"No point with the label {point_id} found")
        return

    def collate(self,
        property: Union[str, List[str]],
        points: List[str] = 'all', 
        holetype: Literal['Borehole', 'CPT', 'Other'] = None
        ) -> pd.DataFrame:
        """
        Returns a collated Pandas Dataframe containing the collected property from each drilled hole.

        Example usage:
            Project.collate('stratigraphy')
                To collect stratigraphy information from all drilled holes 
                (i.e. boreholes, CPTs, and other points)
            Project.collate('sampling', holetype='Borehole')
                To collect sampling information from all boreholes
            Project.collate('coring', points=['BH-01', 'BH-02'])
                Only collects coring information from the two boreholes
            Project.collate(['stratigraphy', 'gsi', 'rockStrength'])
                Collects stratigraphy, GSI, and rock strength data and compiles them onto
                one merged table

        Args:
            property    (str) or List[str]
                The name of the property to collect information from
                e.g. stratigraphy, sampling, etc.

                If a list of properties is given this method will collate all 
            points      List[str]
                Filter the points to collect from.
            holetype    'Borehole'/'CPT'/'Other'
                Specifies the holetype to collect from.
        
        Returns: 
            pandas Dataframe
                A joined Dataframe containing all collected data.
        """
        list_points = self.query_points(points, holetype)

        list_df = []

        for point in list_points:
            if type(property) == str:
                point_df = getattr(point, property, None)
            else:
                point_df = point.merge_datasets(property, ignore_error=True)
            point_id = point.pointID
            
            if point_df is not None:
                point_df = point_df.copy()
                point_df.insert(0, "Point ID", point_id, True)
                if holetype == "CPT":
                    point_df.insert(1, "Depth", point_df.index, True)
                list_df.append(point_df)

        collated_df = pd.concat(list_df, ignore_index=True, axis=0)
        return collated_df

    def query_points(self, 
        points: List[str] = 'all', 
        holetype: Literal['Borehole', 'CPT', 'Other'] = None
        ) -> 'List[Union[GeotechPoint, Borehole, CPT]]':

        if holetype is None:
            query_from_dict = {}
            for k,v in self.points.items():
                query_from_dict.update(v)
        else:
            query_from_dict = self.points[holetype]

        if points == 'all':
            list_points = list(query_from_dict.values())
        else:
            unaccessed_points = list(filter(lambda pt: pt not in query_from_dict.keys(), list_points))
            if len(unaccessed_points) > 0:
                raise IndexError(f"{', '.join(unaccessed_points)} are not found.")
            list_points = [query_from_dict[k] for k in points]

        return list_points
    
    def calculate_CPT_parameters(self,
        points: List[str] = 'all',
        area_ratio: float = None,
        unit_weight: float = None,
        shared_gwl_depth: float = None,
        shared_gwl_el: float = None,
        soil_classification_method: Literal[
            'Eslami Fellenius', 
            'Robertson et al 1986', 
            'Robertson et al 1986 (nonpiezo)', 
            'Robertson 1990'
        ] = None):
        """
        A convenient shorthand method to calculate all relevant CPT parameters.
        """
        list_points = self.query_points(points, 'CPT')
        if not any([area_ratio, unit_weight, shared_gwl_depth, shared_gwl_el, soil_classification_method]):
            raise ValueError('Please enter at least one parameter, e.g. unit weight, to calculate the CPT data.')
        for point in list_points:
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                point.calculate(
                    area_ratio=area_ratio,
                    unit_weight=unit_weight,
                    gwl_depth=shared_gwl_depth,
                    gwl_el=shared_gwl_el,
                    soil_classification_method=soil_classification_method
                )

    def plot_location(self) -> plt.Figure:
        """
        Plot the location of the geotechnical points stored in the Project
        """
        df_points = self.get_list_of_all_points()

        min_x, max_x = (df_points["X"].min(), df_points["X"].max())
        min_y, max_y = (df_points["Y"].min(), df_points["Y"].max())
        round_to_unit = 100
        min_x = (floor(min_x / round_to_unit)) * round_to_unit
        min_y = (floor(min_y / round_to_unit)) * round_to_unit
        max_x = (ceil(max_x / round_to_unit)) * round_to_unit
        max_y = (ceil(max_y / round_to_unit)) * round_to_unit

        group_by_test_type = df_points.groupby('Test Type')

        fig, ax = plt.subplots(1, 1, dpi=144)
        ax.set_aspect('equal')
        for test_type, group in group_by_test_type:
            xcoord = group["X"]
            ycoord = group["Y"]
            pt_color = {'Borehole': 'chartreuse', 'CPT': 'r', 'Other': 'dimgray'}.get(test_type, 'dimgray')
            ax.plot(xcoord, ycoord, label=test_type,
                linestyle='',
                marker='o', markerfacecolor=pt_color, markersize=8,
                scalex=True, scaley=True
            )
            for idx, rw in group.iterrows():
                ax.annotate(rw["Point ID"], (rw["X"], rw["Y"]), 
                    xytext=(-20, 8), textcoords='offset pixels',
                    fontsize='xx-small'
                )
        
        format_num = FuncFormatter(
            lambda x,pos: "{:,}".format(int(x))
        )
        ax.xaxis.set_major_formatter(format_num)
        ax.yaxis.set_major_formatter(format_num)
        ax.grid(True, which='major', axis='both', color='gray', linestyle='--')
        ax.set_axisbelow(True)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        ax.legend()
        
        return fig

    def plot_boreholes(self, 
        points: List[str] = 'all',          
        plot_by_el: bool = False,
        superimpose: Literal['SPT'] = None,
        **kwargs
        ):
        """
        Plot the boreholes stored in the Project.
        """
        list_points = self.query_points(points, 'Borehole')

        style_lookup_colours = kwargs.get("style", None)
        sharey = kwargs.get("sharey", True)
        plot_gwl = kwargs.get("plot_gwl", True)

        num_subplots = len(list_points)
        max_subplots_per_row = 8
        nrows = ceil(num_subplots / max_subplots_per_row)
        ncols = max_subplots_per_row if num_subplots >= max_subplots_per_row else num_subplots

        fig = plt.figure(figsize=(2*ncols+2, nrows*6), dpi=144)
        legend_colours = {}

        # Determine the y-axis limits if sharing a y-axis
        if sharey:
            top_y = [ceil(pt.elevation / 5) * 5 if plot_by_el else 0 for pt in list_points]
            bot_y = [
                floor((pt.elevation - pt.holedepth) / 5) * 5 
                if plot_by_el else 
                (ceil(pt.holedepth / 5) + 1) * 5 
                for pt in list_points
            ]
            top_y = max(top_y)
            bot_y = min(bot_y) if plot_by_el else max(bot_y)

        # Draw the borehole visual on the first subfigure
        legend_width_ratio = 2 / (2*ncols+2)
        subfigs = fig.subfigures(1, 2, width_ratios=[1-legend_width_ratio, legend_width_ratio])
        axs = subfigs[0].subplots(nrows, ncols)
        if nrows > 1:
            axs = axs.flat
        subfigs[0].subplots_adjust(wspace=0.4, left=0.05, right=0.95)
        for i in range(num_subplots):
            ax = axs[i]
            ax, legend_colours = list_points[i].plot_single_log(
                ax, plot_by_el, legend_colours, 
                style_lookup_colours=style_lookup_colours, 
                plot_gwl=plot_gwl
            )
            # Hide y-axis label if not leftmost subplot of the row 
            if i % max_subplots_per_row > 0:
                ax.yaxis.label.set_visible(False)

            # Set up the shared y-axis
            if sharey:
                ax.set_ylim(bot_y, top_y)

            # Superimpose 
            if superimpose == "SPT" and hasattr(list_points[i], "_sampling"):
                list_points[i].superimpose_SPT(ax, plot_by_el, max_SPT=kwargs.get("max_SPT", None))
        
        # Hide all unpopulated subplots 
        if num_subplots < len(axs):
            for i in range(num_subplots, len(axs)):
                axs[i].set_visible(False)
        
        # Add the legend to the right of the axes
        legend_colours = dict(sorted(
            legend_colours.items(),
            key=lambda kv: "zzzz0" if kv[0] == "LOSS" else "zzzz1" if kv[0] == "OTHER" else kv[0]
        ))
        handles = [
            Patch(facecolor=v[0], edgecolor='black', hatch=v[1], label=k) 
            for k,v in legend_colours.items()
        ]
        #subfigs[1].set_facecolor('moccasin')
        subfigs[1].legend(
            handles=handles, title="LEGEND", 
            loc='center left'
        )

        return fig

    def plot_CPT(self, 
        points: List[str] = 'all',
        plot_by_el: bool = False, 
        superimpose: Literal[
            'qc', 'qt', 'fs', 'u2', 
            'Rf', 'Bq', 'Qt', 'Fr', 'Ic'
        ] = None,
        sharey: bool = True,
        xlim: Tuple[float] = None
        ):
        """
        Plot the CPTs stored in the Project.
        """

        dict_xlim = {
            'qc': (0, 8, 2),
            'qt': (0, 8, 2),
            'fs': (0, 2000, 500),
            'u2': (0, 2000, 500),
            'Rf': (0, 50, 10),
            'Bq': (-0.2, 0.8, 0.2),
            'Qt': (-5, 5, 5),
            'Fr': (0, 50, 10),
            'Ic': (1, 4, 1)
        }
        if xlim is None and superimpose is not None:
            xlim = dict_xlim[superimpose]

        list_points = self.query_points(points, 'CPT')

        num_subplots = len(list_points)
        max_subplots_per_row = 8
        nrows = ceil(num_subplots / max_subplots_per_row)
        ncols = max_subplots_per_row if num_subplots >= max_subplots_per_row else num_subplots

        fig = plt.figure(figsize=(2*ncols+2, nrows*6), dpi=144)

        # Determine the y-axis limits if sharing a y-axis
        top_y = [ceil(pt.elevation) + 1 if plot_by_el else 0 for pt in list_points]
        bot_y = [
            floor(pt.elevation - pt.holedepth) - 1
            if plot_by_el else 
            ceil(pt.holedepth) + 1
            for pt in list_points
        ]
        major_unit = [
            0.1 if abs(top_y[i] - bot_y[i]) <= 1 else 
            1 if abs(top_y[i] - bot_y[i]) <= 10 else 5 
            for i in range(len(top_y))
        ]
        minor_unit = [item / 5 for item in major_unit]
        if sharey:
            top_y = max(top_y)
            bot_y = min(bot_y) if plot_by_el else max(bot_y)
            y_range = abs(top_y - bot_y)
            major_unit = 0.1 if y_range <= 1 else 1 if y_range <= 10 else 5
            minor_unit = major_unit / 5

        legend_width_ratio = 2 / (2*ncols+2)
        subfigs = fig.subfigures(1, 2, 
            width_ratios=[1-legend_width_ratio, legend_width_ratio]
        )
        axs = subfigs[0].subplots(nrows, ncols, sharey=sharey)
        if nrows > 1:
            axs = axs.flat

        # Set up the shared y-axis
        if sharey:
            axs[0].set_ylim(bot_y, top_y)
            axs[0].yaxis.set_major_locator(MultipleLocator(major_unit))
            axs[0].yaxis.set_minor_locator(MultipleLocator(minor_unit))

        subfigs[0].subplots_adjust(wspace=0.4, left=0.05, right=0.95)
        for i in range(num_subplots):
            ax = axs[i]
            ax = list_points[i].plot_single_log(
                ax, plot_by_el, False
            )
            # Hide y-axis label if not leftmost subplot of the row 
            if i % max_subplots_per_row == 0:
                ax.set_ylabel("Elevation" if plot_by_el else "Depth, m")

            # Set up the individualized y-axis
            if not sharey:
                ax.set_ylim(bot_y[i], top_y[i])
                ax.yaxis.set_major_locator(MultipleLocator(major_unit[i]))
                ax.yaxis.set_minor_locator(MultipleLocator(minor_unit[i]))

            # Superimpose
            if superimpose is not None:
                data = getattr(list_points[i], superimpose)
                x_data = data.to_numpy()
                y_data = np.array(data.index)
                if plot_by_el:
                    y_data = list_points[i].elevation - y_data

                ax_sp = ax.twiny()
                ax_sp.plot(x_data, y_data, 'k-')
                ax_sp.set_xlabel(data.name)
                ax_sp.set_xlim(*xlim[:2])
                ax_sp.xaxis.set_major_locator(MultipleLocator(xlim[2]))
                ax_sp.xaxis.set_minor_locator(MultipleLocator(xlim[2] / 5))
                
        
        # Hide all unpopulated subplots
        if num_subplots < len(axs):
            for i in range(num_subplots, len(axs)):
                axs[i].set_visible(False)
        
        #subfigs[1].set_facecolor('moccasin')
        subfigs[1] = list_points[0].plot_soil_legend(subfigs[1])

        return fig


        
    """
    def save_to_db(self, human_readable: bool = False):
        \"""
        Save project to SQLite Database. 
        \"""
        filename = input('Please enter the filepath & name of the file.')
        pass

    def export_to_excel(self):
        \"""
        Save project to Excel file. 
        \"""
        pass

    def export_to_json(self):
        \"""
        Save project to JSon file.
        \"""
        pass
    """