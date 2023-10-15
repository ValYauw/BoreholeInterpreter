import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classes.GeotechPoint import GeotechPoint, PointDataset

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from math import ceil
import re
import pandas as pd

from typing import List, Tuple, Dict, Iterable, Literal, Optional

import copy

class Borehole(GeotechPoint):
    """
    A class representing a Borehole point.
    """

    _pointType = "Borehole"

    @property
    def sampling(self) -> PointDataset:
        """
        Get the sampling dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_sampling"):
            return self._sampling
        else:
            raise AttributeError(
                f"Sampling data of {self.pointID} is not yet defined."
            )
    @sampling.setter
    def sampling(self, 
        arrSampling: 'Iterable[Iterable[float, float, float, str, str, float or str, float, float, float, float]]'
        ):
        """
        Set the sampling dataset of the borehole.

        Args:
            arrSampling     Iterable[Iterable[float, float, float, str, str, float or str, float, float, float, float]]
                A 2D array-like data structure representing the raw sampling dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (float)     The recovery, in metres
                Column 4:   (str)       The sample type in a short word, e.g. SPT, UDS, UCS, BULK 
                Column 5:   (str)       The sampling description, as brief or full as wanted
                Column 6:   (str/float) A string, representing the SPT-N value to be plotted, e.g. "30", "HW", "HB", ">50"
                Column 7:   (float)     The values recorded from Pocket Penetrometer
                Column 8:   (float)     The values recorded from Torvane
                Column 9:   (float)     The peak shear strength recorded from vane shear tests 
                Column 10:  (float)     The residual shear strength recorded from vane shear tests 
                Values in columns 4-6 may be left as an empty string.
                Columns to the right may be omitted if no values are recorded for that point in the dataset, e.g. if only measuring SPTs.
        """
        dictDtypes = {
            "Recovery": float, 
            "Sample Type": 'category', 
            "Sampling Description": str, 
            "SPT-N Value": str, 
            "Pocket Penetrometer": float,
            "Torvane": float, 
            "Vane Shear Test (Peak)": float, 
            "Vane Shear Test (Residual)": float
        }
        df = self.createDataset(arrSampling, dictDtypes, allowNan=True)
        self._sampling = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    @property
    def consistency_density(self) -> PointDataset:
        """
        Get the consistency (for cohesive soils) or density (for cohesionless soils) dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_consistency_density"):
            return self._consistency_density
        else:
            raise AttributeError(
                f"Consistency/Density of {self.pointID} is not yet defined."
            )
    @consistency_density.setter
    def consistency_density(self, 
        arrConsistency: 'Iterable[Iterable[float, float, str]]'
        ):
        """
        Set the consistency (for cohesive soils) or density (for cohesionless soils) dataset of the borehole.

        Args:
            arrConsistency     Iterable[Iterable[float, float, str]]
                A 2D array-like data structure representing the raw consistency/density dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (str)       The consistency/density, e.g. "VS" (Very Soft), "VD" (Very Dense)
                Values in column 3 may be left as an empty string.
        """
        dictDtypes = {
            "Consistency/Density": 'category'
        }
        df = self.createDataset(arrConsistency, dictDtypes, allowNan=True)
        self._consistency_density = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    @property
    def moisture(self) -> PointDataset:
        """
        Get the moisture dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_moisture"):
            return self._moisture
        else:
            raise AttributeError(
                f"Moisture of {self.pointID} is not yet defined."
            )
    @moisture.setter
    def moisture(self, 
        arrMoisture: 'Iterable[Iterable[float, float, str]]'
        ):
        """
        Set the moisture dataset of the borehole.

        Args:
            arrMoisture     Iterable[Iterable[float, float, str]]
                A 2D array-like data structure representing the raw moisture dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (str)       The moisture, e.g. "M" (Moist)
                Values in column 3 may be left as an empty string.
        """
        dictDtypes = {
            "Moisture": 'category'
        }
        df = self.createDataset(arrMoisture, dictDtypes, allowNan=True)
        self._moisture = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    @property
    def gwl(self) -> PointDataset:
        """
        Get the groundwater level dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_gwl"):
            return self._gwl
        else:
            raise AttributeError(
                f"Groundwater records of {self.pointID} is not yet defined."
            )
    @gwl.setter
    def gwl(self, 
        arrGWL: 'Iterable[Iterable[float, str]]'
        ):
        """
        Set the groundwater level (GWL) dataset of the borehole.

        Args:
            arrGWL     Iterable[Iterable[float, str]]
                A 2D array-like data structure representing the raw GWL dataset of the point.
                Column 1:   (float)     The GWL depth
                Column 2:   (str)       A string containing more details, e.g. the date of recording
                Values in column 2 may be left as an empty string.
        """
        arrGWL = copy.deepcopy(arrGWL)
        arrGWL = list(map(lambda rw: [rw[0], None, *rw[1:]], arrGWL))
        dictDtypes = {
            "GWL": str
        }
        self._gwl = self.createDataset(arrGWL, dictDtypes)

    @property
    def gsi(self) -> PointDataset:
        """
        Get the Geological Strength Index (GSI) of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_gsi"):
            return self._gsi
        else:
            raise AttributeError(
                f"GSI of {self.pointID} is not yet defined."
            )
    @gsi.setter
    def gsi(self, 
        arrGSI: 'Iterable[Iterable[float, float, int]]'
        ):
        """
        Set the Geological Strength Index (GSI) dataset of the borehole.

        Args:
            arrGSI     Iterable[Iterable[float, float, int]]
                A 2D array-like data structure representing the raw GSI dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (int)       The GSI of the data point
        """
        dictDtypes = {
            "GSI": int
        }    
        df = self.createDataset(arrGSI, dictDtypes, allowNan=True)
        self._gsi = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    @property
    def weathering(self) -> PointDataset:
        """
        Get the rock weathering dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_weathering"):
            return self._weathering
        else:
            raise AttributeError(
                f"Weathering of {self.pointID} is not yet defined."
            )
    @weathering.setter
    def weathering(self, 
        arrWeathering: 'Iterable[Iterable[float, float, str]]'
        ):
        """
        Set the rock weathering dataset of the borehole.

        Args:
            arrWeathering     Iterable[Iterable[float, float, str]]
                A 2D array-like data structure representing the raw rock weathering dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (str)       A string, describing the rock weathering of the data point in brief, e.g. "EW" (Extremely Weathered)
                Values in column 3 may be left as an empty string.
        """
        dictDtypes = {
            "Weathering": 'category'
        }       
        df = self.createDataset(arrWeathering, dictDtypes, allowNan=True)
        self._weathering = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    @property
    def rockStrength(self) -> PointDataset:
        """
        Get the rock strength dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_rockStrength"):
            return self._rockStrength
        else:
            raise AttributeError(
                f"Rock Strength of {self.pointID} is not yet defined."
            )
    @rockStrength.setter
    def rockStrength(self, 
        arrRockStrength: 'Iterable[Iterable[float, float, str]]'
        ):
        """
        Set the rock strength dataset of the borehole.

        Args:
            arrRockStrength     Iterable[Iterable[float, float, str]]
                A 2D array-like data structure representing the raw rock strength dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (str)       A string, describing the rock strength of the data point in brief, e.g. "R0"
                Values in column 3 may be left as an empty string.
        """
        dictDtypes = {
            "Rock Strength": 'category'
        }       
        df = self.createDataset(arrRockStrength, dictDtypes, allowNan=True)
        self._rockStrength = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    @property
    def fractureFrequency(self) -> PointDataset:
        """
        Get the rock fracture frequency dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_fractureFrequency"):
            return self._fractureFrequency
        else:
            raise AttributeError(
                f"Fracture Frequency of {self.pointID} is not yet defined."
            )
    @fractureFrequency.setter
    def fractureFrequency(self, 
        arrFractureFrequency: 'Iterable[Iterable[float, float, float]]'
        ):
        """
        Set the rock fracture frequency dataset of the borehole.

        Args:
            arrFractureFrequency     Iterable[Iterable[float, float, float]]
                A 2D array-like data structure representing the raw rock fracture frequency dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (float)     A string, describing the fracture frequency (in number of fractures per metre run).
        """
        dictDtypes = {
            "Fracture Frequency": float
        }        
        df = self.createDataset(arrFractureFrequency, dictDtypes, allowNan=True)
        self._fractureFrequency = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    @property
    def defectDesc(self) -> PointDataset:
        """
        Get the rock defect description dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_defectDesc"):
            return self._defectDesc
        else:
            raise AttributeError(
                f"Defect Description of {self.pointID} is not yet defined."
            )
    @defectDesc.setter
    def defectDesc(self, 
        arrDefectDesc: 'Iterable[Iterable[float, float, str]]'
        ):
        """
        Set the rock fracture frequency dataset of the borehole.

        Args:
            arrDefectDesc     Iterable[Iterable[float, float, str]]
                A 2D array-like data structure representing the raw rock defect description dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (str)       A string, describing the defect description, e.g. rock structural joint logging and description.
                Values in column 3 may be left as an empty string.
        """
        dictDtypes = {
            "Defect Description": str
        }
        self._defectDesc = self.createDataset(arrDefectDesc, dictDtypes)
    
    @property
    def coring(self) -> PointDataset:
        """
        Get the rock coring (TCR, RQD, SCR) dataset of the geotechnical point.

        Returns:
            PointDataset
        """
        if hasattr(self, "_coring"):
            return self._coring
        else:
            raise AttributeError(
                f"Coring of {self.pointID} is not yet defined."
            )
    @coring.setter
    def coring(self, 
        arrCoring: 'Iterable[Iterable[float, float, float, float, float]]'
        ):
        """
        Set the rock coring dataset of the borehole.

        Args:
            arrCoring     Iterable[Iterable[float, float, float]]
                A 2D array-like data structure representing the raw rock coring description dataset of the point.
                Column 1:   (float)     The top boundary depth of the data point
                Column 2:   (float)     The bottom boundary depth of the data point
                Column 3:   (float)     The Total Core Recovery (TCR) of the data point
                Column 4:   (float)     The Rock Quality Designation (RQD) of the data point
                Column 5:   (float)     The Solid Core Recovery (SCR) of the data point
        """
        dictDtypes = {
            "TCR": float,
            "RQD": float,
            "SCR": float
        }
        df = self.createDataset(arrCoring, dictDtypes, allowNan=True)
        self._coring = df
        if not df.empty:
            bottom_depth = df.iloc[-1, 1]
            if self.holedepth < bottom_depth:
                self.holedepth = bottom_depth
    
    def superimpose_SPT(self, 
        ax: plt.Axes, 
        plot_by_elevation: bool = False,
        max_SPT: int = None
        ) -> plt.Axes:
        """
        Superimpose the SPT-N values on the given figure.

        Args:
            ax                  (Matplotlib Axes)
                A Matplotlib Axes object that will be superimposed on.
                This object will not be modified.
            plot_by_elevation   (bool)
                Boolean setting to draw the visual log by the elevation or depth.
                (Default: Draw by depth) 
        
        Returns:
            Matplotlib Axes
                A Matplotlib Axes object with twinned y-axis as the original Axes object,
                that plots the SPT-N on top of this original object
        """

        df_sampling = self.sampling
        df_SPT = self.sampling.loc[
            df_sampling["Sample Type"] == "SPT", 
            ["Depth from", "Depth to", "SPT-N Value"]
        ]
        del df_sampling

        x_labels = df_SPT["SPT-N Value"]
        #x_values = pd.to_numeric(x_labels, errors='ignore', downcast='integer')
        x_values = pd.Series(index=x_labels.index, dtype=object)
        max_xvalue = 0 if max_SPT is None else max_SPT
        for idx, x in x_labels.items():
            if x == "HW":
                x = 0
            lookup_regex = re.search(r"^>\s*(\d+)\s*$", str(x))
            if not lookup_regex is None:
                x = int(lookup_regex.group(1))
            if max_SPT is None:
                x_values.loc[idx] = x 
                if str(x).isnumeric():
                    max_xvalue = int(x) if max_xvalue < int(x) else max_xvalue
            else:
                if str(x).isnumeric():
                    x_values.loc[idx] = int(x) if int(x) < max_SPT else max_SPT
        x_values.loc[x_values == "HB"] = max_xvalue
        x_values = pd.to_numeric(x_values, errors='coerce', downcast='integer')

        y_values = self.elevation - df_SPT["Depth from"] if plot_by_elevation else df_SPT["Depth from"]

        ax_sp = ax.twiny()
        ax_sp.set_position(ax.get_position())
        ax_sp.plot(x_values, y_values, 'o-k', markerfacecolor='white', markersize=5)

        ax_sp.set_xlim(0, ceil(max_xvalue / 10) * 10)
        ax_sp.xaxis.set_major_locator(MultipleLocator(10))
        ax_sp.set_xlabel("SPT-N")
        ax.grid(False, axis='x')
        ax_sp.grid(True, which='major', axis='x', color='darkgray', linestyle=':')

        return ax_sp
    
    def plot(self, 
        plot_by_elevation: Optional[bool] = False, 
        superimpose: Optional[Literal['SPT']] = None,
        **kwargs
        ):
        """
        Create a visual representation, i.e. a visual log, of the borehole's stratigraphy.

        Args:
            plot_by_elevation   (bool)
                Boolean setting to draw the visual log by the elevation or depth.
                (Default: Draw by depth) 
            superimpose         ('SPT')
                Property to superimpose on the visual log, e.g. SPT-N
        
        Keyword arguments:
            style               (Dict)
                A custom dictionary with user-defined stratigraphy styling.
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
            Matplotlib Figure
                A Matplotlib Figure object representing the complete visual.
        """

        style_lookup_colours = kwargs.get("style", None)
        plot_gwl = kwargs.get("plot_gwl", True)

        fig = plt.figure(figsize=(4, 6))
        subfigs = fig.subfigures(1, 2, width_ratios=[0.65, 0.35])
        legend_colours = {}

        # Draw the borehole visual on the first subfigure
        ax = subfigs[0].subplots(1, 1)
        ax, legend_colours = self.plot_single_log(
            ax, plot_by_elevation, legend_colours, 
            style_lookup_colours=style_lookup_colours,
            plot_gwl=plot_gwl
        )

        # Superimpose 
        if superimpose == "SPT":
            self.superimpose_SPT(ax, plot_by_elevation, max_SPT=kwargs.get("max_SPT", None))
        
        # Add the legend to the right of the axes
        legend_colours = dict(sorted(
            legend_colours.items(),
            key=lambda kv: "zzzz0" if kv[0] == "LOSS" else "zzzz1" if kv[0] == "OTHER" else kv[0]
        ))
        handles = [
            Patch(facecolor=v[0], edgecolor='black', hatch=v[1], label=k) 
            for k,v in legend_colours.items()
        ]
        subfigs[1].legend(handles=handles, title="LEGEND", loc='center', bbox_to_anchor=(0, 0.5))

        return fig
