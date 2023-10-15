import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classes.GeotechPoint import GeotechPoint
from classes.CPT_SoilClassification import CPT_SoilClassification

from typing import List, Tuple, Dict, Union, Iterable, Optional, Literal

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from textwrap import fill
from math import ceil, floor

import warnings

GLOBAL_unit_weight_of_water = 10

class CPT(GeotechPoint):
    """
    A class representing a Cone Penetration Test (CPT) point.
    """

    _pointType = "CPT"

    def listener_dependency(self,
        list_dependencies: List[str],
        reset_classification: bool = True):
        """
        A function to define a custom listener-like function that tracks when the given property
        is changed (i.e. set or deleted) and deletes the affected dependencies in memory.
        """
        
        for dpd in list_dependencies:
            if hasattr(self, "_" + dpd):
                delattr(self, "_" + dpd)
        if reset_classification and hasattr(self, '_soil_classification_method'):
            del self.classification

    @property
    def raw_data(self) -> pd.DataFrame:
        """
        Gets the raw drilling data of the CPT, which are: cone penetration (qc), sleeve friction (fs), porewater pressure (u2)

        Returns:
            pd.Dataframe:   An nx3 Pandas Dataframe representing the drilling data of the CPT.
                            Columns correspond to qc, fs, and u2.
                            Index corresponds to the depth points of each probe recording.
        """
        if not hasattr(self, "_qc"):
            raise AttributeError(
                f"Drilling data of {self.pointID} is not yet defined."
            )
        qc = self._qc
        index = qc.index
        fs = self._fs if hasattr(self, "_fs") else pd.Series(index=index, dtype='float64', name="fs")
        u2 = self._u2 if hasattr(self, "_u2") else pd.Series(index=index, dtype='float64', name="u2")
        table_data = pd.concat([qc, fs, u2], axis=1)
        return table_data
    
    @raw_data.setter
    def raw_data(self, 
        data: 'Iterable[Iterable[float, float, float, Optional[float]]]'
        ):
        """
        Sets the raw drilling data of the CPT, which are: cone penetration (qc), sleeve friction (fs), porewater pressure (u2)

        Args:
            data (Iterable[Iterable[float, float, float, Optional[float]]]): 
                An nx4 array-like object, e.g. list or numpy array, representing the drilling data of the CPT.
                Column 1: Depth point
                Column 2: Cone penetration, qc (MPa)
                Column 3: Sleeve friction fs (kPa)
                Column 4: Porewater pressure behind piezocone, u2 (kPa)
        """

        table_data = np.array(data, dtype=np.float64)
        if table_data.ndim != 2:
            raise ValueError("The input array must be two-dimensional.")
        num_columns = table_data.shape[1]
        if num_columns < 3:
            raise ValueError(
                "The input array must have at least 3 columns: Depth, Cone penetration qc, and Sleeve friction fs. A fourth column may be given for porewater pressure u2."
            )
        
        # Store the raw data
        self._qc = pd.Series(data=table_data[:, 1], index=table_data[:, 0], name="qc", copy=True)
        self._fs = pd.Series(data=table_data[:, 2], index=table_data[:, 0], name="fs", copy=True)
        if num_columns > 3:
            self._u2 = pd.Series(data=table_data[:, 3], index=table_data[:, 0], name="u2", copy=True)
        self.holedepth = table_data[:, 0].max()

        # Removes dependencies
        self.listener_dependency([
            'qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic', 
            'total_stress', 'effective_stress', 
            'static_u0', 'elevated_u0'
        ])

        # Re-calculate total stress with the given depth points
        if hasattr(self, "_total_stress"):
            self._calculate_stress()
    
    @raw_data.deleter
    def raw_data(self):
        self.listener_dependency([
            'qc', 'fs', 'u2',
            'total_stress', 'effective_stress', 
            'static_u0', 'elevated_u0'
        ])
        del self.qt
    
    def _validate_depth_index(self, 
        seriesData: pd.Series, 
        arr_checkDatasets: List[str] = ['qc', 'fs', 'u2']
        ) -> 'Tuple[bool, Union(None, List[str])]':
        """
        Private method. Does a check with previously existing raw dataset (qc, fs, u2) to 
        see if the depth index of the given pandas series is equal to that of the stored datasets.
        """
        arr_existing_data_series = []
        arr_misaligned_dataset = []
        for attr in arr_checkDatasets:
            if hasattr(self, "_" + attr):
                arr_existing_data_series.append(self.__dict__["_" + attr])
        if len(arr_existing_data_series) == 0:
            return (True, None)
        check_ds_depth = seriesData.index
        bool_validated = True
        for existing_data_series in arr_existing_data_series:
            existing_ds_depth = existing_data_series.index
            if not check_ds_depth.equals(existing_ds_depth):
                bool_validated = False
                arr_misaligned_dataset.append(existing_data_series.name)
        return (bool_validated, None if bool_validated else arr_misaligned_dataset)

    @property
    def qc(self) -> pd.Series:
        """
        Gets the cone penetration (qc) data of the CPT

        Returns:
            pd.Series:  A Pandas Series representing the cone penetration (qc) of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if not hasattr(self, "_qc"):
            raise AttributeError(
                f"Cone penetration, qc, of {self.pointID} is not yet defined."
            )
        return self._qc  
              
    @qc.setter
    def qc(self, 
        data: 'Iterable[Iterable[float, float]]'
        ):
        """
        Sets the cone penetration (qc) data of the CPT

        Args:
            data (Iterable[Iterable[float, float]]): 
                An nx2 array-like object, e.g. list or numpy array, representing the drilling data of the CPT.
                Column 1: Depth point
                Column 2: Cone penetration, qc
        """
        table_data = np.array(data, dtype=np.float64)
        qc_series = pd.Series(data=table_data[:, 1], index=table_data[:, 0], name="qc", copy=True)
        (bool_validated, arr_mismatcheddatasets) = self._validate_depth_index(qc_series, ['fs', 'u2'])
        if not bool_validated:
            raise ValueError(
                f'Error: Input data is mismatched with existing {", ".join(arr_mismatcheddatasets)} datasets.\nTo reset the whole raw dataset, use the raw_data property.'
            )
        (bool_keepexistingindex, arr_mismatcheddatasets) = self._validate_depth_index(qc_series, ['qc'])
        self._qc = qc_series
        max_depth = qc_series.max()
        if self.holedepth < max_depth:
            self.holedepth = max_depth

        # Removes dependencies
        del self.qt

        # Re-calculate total stress with the given depth points
        if hasattr(self, "_total_stress") and not bool_keepexistingindex:
            self._calculate_stress()
    
    @qc.deleter
    def qc(self):
        del self.qt
        del self._qc

    @property
    def fs(self) -> pd.Series:
        """
        Gets the sleeve friction (fs) data of the CPT

        Returns:
            pd.Series:  A Pandas Series representing the sleeve friction (fs) of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if not hasattr(self, "_fs"):
            raise AttributeError(
                f"Sleeve friction, fs, of {self.pointID} is not yet defined."
            )
        return self._fs
    
    @fs.setter
    def fs(self,
        data: 'Iterable[Iterable[float, float]]'
        ):
        """
        Sets the sleeve friction (fs) data of the CPT

        Args:
            data (Iterable[Iterable[float, float]]): 
                An nx2 array-like object, e.g. list or numpy array, representing the drilling data of the CPT.
                Column 1: Depth point
                Column 2: Sleeve friction fs
        """
        table_data = np.array(data, dtype=np.float64)
        fs_series = pd.Series(data=table_data[:, 1], index=table_data[:, 0], name="fs", copy=True)
        (bool_validated, arr_mismatcheddatasets) = self._validate_depth_index(fs_series, ['qc', 'u2'])
        if not bool_validated:
            raise ValueError(
                f'Error: Input data is mismatched with existing {", ".join(arr_mismatcheddatasets)} datasets.\nTo reset the whole raw dataset, use the raw_data property.'
            )
        (bool_keepexistingindex, arr_mismatcheddatasets) = self._validate_depth_index(fs_series, ['fs'])
        self._fs = fs_series
        max_depth = fs_series.max()
        if self.holedepth < max_depth:
            self.holedepth = max_depth

        # Removes dependencies
        self.listener_dependency(['Rf', 'Fr', 'Ic'])

        # Re-calculate total stress with the given depth points
        if hasattr(self, "_total_stress") and not bool_keepexistingindex:
            self._calculate_stress()
    
    @fs.deleter
    def fs(self):
        self.listener_dependency(['Rf', 'Fr', 'Ic'])
        del self._fs
    
    @property
    def u2(self) -> pd.Series:
        """
        Gets the porewater pressure behind piezocone (u2) data of the CPT

        Returns:
            pd.Series:  A Pandas Series representing the porewater pressure behind piezocone (u2) of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if not hasattr(self, "_u2"):
            raise AttributeError(
                f"Pore pressure, u2, of {self.pointID} is not yet defined."
            )
        return self._u2
    
    @u2.setter
    def u2(self, 
        data: 'Iterable[Iterable[float, float]]'
        ):
        """
        Sets the porewater pressure (u2) data of the CPT

        Args:
            data (Iterable[Iterable[float, float]]): 
                An nx2 array-like object, e.g. list or numpy array, representing the drilling data of the CPT.
                Column 1: Depth point
                Column 2: Porewater pressure behind piezocone u2
        """
        table_data = np.array(data, dtype=np.float64)
        u2_series = pd.Series(data=table_data[:, 1], index=table_data[:, 0], name="u2", copy=True)
        (bool_validated, arr_mismatcheddatasets) = self._validate_depth_index(u2_series, ['qc', 'fs'])
        if not bool_validated:
            raise ValueError(
                f'Error: Input data is mismatched with existing {", ".join(arr_mismatcheddatasets)} datasets.\nTo reset the whole raw dataset, use the raw_data property.'
            )
        (bool_keepexistingindex, arr_mismatcheddatasets) = self._validate_depth_index(u2_series, ['u2'])
        self._u2 = u2_series
        max_depth = u2_series.max()
        if self.holedepth < max_depth:
            self.holedepth = max_depth

        # Removes dependencies
        del self.qt   

        # Re-calculate total stress with the given depth points
        if hasattr(self, "_total_stress") and not bool_keepexistingindex:
            self._calculate_stress()
    
    @u2.deleter
    def u2(self):
        del self.qt
        del self._u2 
    
    @property
    def area_ratio(self) -> float:
        """
        Gets the area ratio of the CPT probe

        Returns:
            float
        """
        if not hasattr(self, "_area_ratio"):
            raise AttributeError(
                f"Area ratio of {self.pointID} is not yet defined.\nFor all calculations qt will be assumed to be the same as qc"
            )
        return self._area_ratio
    
    @area_ratio.setter
    def area_ratio(self, value: float):
        """
        Sets the area ratio of the CPT probe

        Args:
            value (float)
        """
        self._area_ratio = float(value)
        
        # Removes dependencies
        del self.qt

    @area_ratio.deleter
    def area_ratio(self):
        del self.qt
        del self._area_ratio

    @property
    def gwl(self) -> float:
        """
        Gets the static groundwater level (in m below ground level) of the CPT test point

        Returns:
            float
        """
        if not hasattr(self, "_gwl"):
            raise AttributeError(
                f"Static groundwater level of {self.pointID} is not yet defined. This data is required to compute the effective stress and pore pressure ratio Bq."
            )
        return self._gwl
    
    @gwl.setter
    def gwl(self, value: float):
        """
        Sets the static groundwater level

        Args:
            value (float)
        """
        self._gwl = float(value)
        self._calculate_effective_stress()
        # Removes dependencies
        self.listener_dependency(['static_u0', 'effective_stress', 'Qt', 'Bq', 'Ic'])

    @gwl.deleter
    def gwl(self):
        self.listener_dependency(['static_u0', 'effective_stress', 'Qt', 'Bq', 'Ic'])
        del self._gwl

    @property
    def static_gwl(self) -> float:
        """
        Gets the static groundwater level (in m below ground level) of the CPT test point

        Returns:
            float
        """
        try:
            return self.gwl
        except Exception as e:
            raise e
        
    @static_gwl.setter
    def static_gwl(self, rawGWL: float):
        """
        Sets the static groundwater level

        Args:
            value (float)
        """
        self.gwl = float(rawGWL)
    
    @static_gwl.deleter
    def static_gwl(self):
        del self.gwl

    @property
    def unit_weight(self) -> np.ndarray:
        """
        Gets the unit weight of soil along depth of the CPT test point

        Returns:
            np.ndarray:  
                An nx2 numpy array, representing the unit weight at known depth points.
        
        Examples:
            An output of [[0, 16], [20, 16]] means that the unit weight of soil is 16 kN/m3 from depths 0-20 m.
        """
        if not hasattr(self, "_unit_weight"):
            raise AttributeError(
                f"Unit weight of {self.pointID} is not yet defined."
            )
        return self._unit_weight
    
    @unit_weight.setter
    def unit_weight(self, 
        value: 'Union(float, int, Iterable[Iterable[float, float]])'
        ):
        """
        Sets the unit weight of soil along depth of the CPT test point

        Args:
            value (Union(float, int, Iterable[Iterable[float, float]])):
                The unit weight can either be set using a number data type (Python float or Python int), or an nx2 array-like object.

                If set using a number data type, this library will assume that the unit weight is static (stays the same) throughout depths 0-9999 m.

                If set using an nx2 array-like object, e.g. a list or numpy array, this libary will assume a changing unit weight through depth.
        
        Examples:
            >>> cpt_point.unit_weight = 16
            >>> print(cpt_point.unit_weight)
            [ 0     16
              9999  16 ]

            >>> cpt_point.unit_weight = [[0, 14], [20, 15]]
            >>> print(cpt_point.unit_weight)
            [ 0    14
              20   15 ]

            >>> cpt_point.unit_weight = np.array([[0, 14], [20, 15]])
            >>> print(cpt_point.unit_weight)
            [ 0    14
              20   15 ]
        """
        if type(value) == int or type(value) == float:
            self._unit_weight = np.array([[0, value], [9999, value]])
        else:
            self._unit_weight = np.array(value)
        # Removes dependencies
        self.listener_dependency([
            'total_stress', 'effective_stress', 
            'Qt', 'Fr', 'Bq', 'Ic'
        ])
        self._calculate_stress()
    
    @unit_weight.deleter
    def unit_weight(self):
        self.listener_dependency([
            'total_stress', 'effective_stress', 
            'Qt', 'Fr', 'Bq', 'Ic'
        ])
        del self._unit_weight
    
    @property
    def elevated_gwl(self) -> float:
        """
        Gets the elevated groundwater level (in m below ground level) of the CPT test point

        Elevated porewater pressures may be experienced in the CPT test point during drilling. Use this property to set the approximate groundwater level that would generate this elevated porewater pressure.

        By default pore pressure ratio (Bq) and soil effective stress will be calculated using the static groundwater level if this parameter is not set. Bq and effective stress are calculated using the elevated groundwater level otherwise.

        Returns:
            float
        """
        if not hasattr(self, "_elevated_gwl"):
            raise AttributeError(
                f"""Elevated groundwater level of {self.pointID} is not yet defined.
This data represents the depth of groundwater level that is elevated during CPT drilling.
This data is used to compute the pore pressure ratio Bq, and takes precedence over the static groundwater level (CPT.gwl).
If this value is omitted this library will use the static groundwater level by default."""
            )
        return self._elevated_gwl
    
    @elevated_gwl.setter
    def elevated_gwl(self, value: float):
        """
        Sets the elevated groundwater level (in m below ground level) of the CPT test point

        Elevated porewater pressures may be experienced in the CPT test point during drilling. Use this property to set the approximate groundwater level that would generate this elevated porewater pressure.

        By default pore pressure ratio (Bq) and soil effective stress will be calculated using the static groundwater level if this parameter is not set. Bq and effective stress are calculated using the elevated groundwater level otherwise.

        Returns:
            value (float)
        """
        self._elevated_gwl = float(value)
        self._calculate_effective_stress()
        # Removes dependencies
        self.listener_dependency(['elevated_u0', 'effective_stress', 'Qt', 'Bq', 'Ic'])
    
    @elevated_gwl.deleter
    def elevated_gwl(self):
        self.listener_dependency(['elevated_u0', 'effective_stress', 'Qt', 'Bq', 'Ic'])
        del self._elevated_gwl
    
    @property
    def total_stress(self) -> pd.Series:
        """
        Gets the total stress distribution of the CPT test point

        Dependent on the property unit_weight.
        This property will calculate the total stress distribution the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_total_stress"):
            return self._total_stress
        self._calculate_stress()
        return self._total_stress

    @total_stress.deleter
    def total_stress(self):
        del self._total_stress

    @property
    def effective_stress(self) -> pd.Series:
        """
        Gets the effective stress distribution of the CPT test point

        Dependent on the property unit_weight and groundwater level.
        This property will calculate the effective stress distribution the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_effective_stress"):
            return self._effective_stress
        self._calculate_stress()
        return self._effective_stress

    @effective_stress.deleter
    def effective_stress(self):
        del self._effective_stress
    
    @property
    def static_u0(self) -> pd.Series:
        """
        Gets the static porewater pressure of the CPT test point, using the static groundwater level

        Dependent on the property gwl.
        This property will calculate the static porewater pressure the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_static_u0"):
            return self._static_u0
        if not hasattr(self, "_gwl"):
            raise AttributeError(
                f"Groundwater level of {self.pointID} not set. This is needed to calculate u0."
            )
        self._calculate_stress()
        return self._static_u0

    @static_u0.deleter
    def static_u0(self):
        del self._static_u0

    @property
    def u0(self) -> pd.Series:
        """
        Gets the static porewater pressure of the CPT test point, using the static groundwater level

        Dependent on the property gwl.
        This property will calculate the static porewater pressure the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        return self.static_u0

    @u0.deleter
    def u0(self):
        del self.static_u0
    
    @property
    def elevated_u0(self) -> pd.Series:
        """
        Gets the static porewater pressure of the CPT test point, using the elevated groundwater level

        Dependent on the property elevated_gwl.
        This property will calculate the static porewater pressure the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_elevated_u0"):
            return self._elevated_u0
        if not hasattr(self, "_elevated_gwl"):
            raise AttributeError(
                f"Elevated groundwater level of {self.pointID} not set. This is needed to calculate elevated u0."
            )
        self._calculate_stress()
        return self._elevated_u0
    
    @elevated_u0.deleter
    def elevated_u0(self):
        del self._elevated_u0
    
    def _calculate_stress(self):
        self._calculate_total_stress()
        if hasattr(self, "_gwl") or hasattr(self, "_elevated_gwl"):
            self._calculate_effective_stress()
        if hasattr(self, '_soil_classification_method'):
            try:
                self.classify_soil(self._soil_classification_method)
            except Exception:
                pass

    def _calculate_total_stress(self):
        """"
        Private method. Calculate the total stress and store to the _total_stress private property when called. 
        """
        if not hasattr(self, "_unit_weight"):
            raise AttributeError(
                f"Unit weight of {self.pointID} not set. This is needed to calculate the total stress across depth."
            )
        
        depth_index_series = None
        for attr in ["_qc", "_fs", "_u2"]:
            if hasattr(self, attr):
                depth_index_series = self.__dict__[attr].index.to_numpy()
                break
        if depth_index_series is None:
            depth_index_series = self._unit_weight[:, 0]
        
        unit_weight_dataset = self._unit_weight.copy()
        for i in range(1, unit_weight_dataset.shape[0]):
            if unit_weight_dataset[i, 0] == unit_weight_dataset[i-1, 0]:
                unit_weight_dataset[i, 0] += 0.000001

        interpolated_unit_weight = [
            np.interp(z, unit_weight_dataset[:, 0], unit_weight_dataset[:, 1]) 
            for z in depth_index_series
        ]
        interpolated_inc_total_stress = np.diff(depth_index_series) * interpolated_unit_weight[1:]
        total_stress_series = pd.Series(
            index=depth_index_series, 
            data=np.concatenate(
                [np.array([depth_index_series[0] * unit_weight_dataset[0, 1]]), 
                interpolated_inc_total_stress]
            ),
            dtype='float64', 
            name="sv0"
        )
        total_stress_series = total_stress_series.cumsum()
        self._total_stress = total_stress_series
    
    def _calculate_effective_stress(self):
        """"
        Private method. Calculate the static & elevated porewater pressure and effective stress and store to the _u0 and _effective_stress private property when called. 

        By default effective stress will be the total stress less the elevated porewater pressures, and is thus dependent on the property _elevated_gwl. 
        If _elevated_gwl is not set then the effective stress will be calculated using porewater pressures calculated using the _static_gwl property.
        """
        if not hasattr(self, "_unit_weight"):
            raise AttributeError(
                f"Unit weight of {self.pointID} is not yet defined. This is needed to calculate the total/effective stress across depth."
            )
        if not hasattr(self, "_gwl") and not hasattr(self, "_elevated_gwl"):
            raise AttributeError(
                f"Groundwater level of {self.pointID} is not yet defined. This is needed to calculate the effective stress across depth."
            )
        
        total_stress = self._total_stress
        depth_index = total_stress.index.to_numpy()
        if hasattr(self, "_elevated_gwl"):
            elevated_gwl = self._elevated_gwl
            elevated_u0 = pd.Series(
                index=depth_index,
                data=map(lambda z: 0 if z < elevated_gwl else (z-elevated_gwl) * GLOBAL_unit_weight_of_water, depth_index),
                name="u0 (elevated)"
            )
            self._elevated_u0 = elevated_u0
            effective_stress = total_stress - elevated_u0

        if hasattr(self, "_gwl"):
            static_gwl = self._gwl
            static_u0 = pd.Series(
                index=depth_index,
                data=map(lambda z: 0 if z < static_gwl else (z-static_gwl) * GLOBAL_unit_weight_of_water, depth_index),
                name="u0"
            )
            self._static_u0 = static_u0
            if not hasattr(self, "_elevated_gwl"):
                effective_stress = total_stress - static_u0
                
        effective_stress.name = 'sv0\''
        self._effective_stress = effective_stress

    @property
    def qt(self) -> pd.Series:
        """
        Gets the corrected cone penetration of the CPT test point, qt = qc + u2 * (1 - a)

        Dependent on the properties qc, u2, and area_ratio.
        This property will calculate the corrected cone penetration the first time it is called, and store that value in the object instance of the class to be returned in future calls.
        If area_ratio is not set, then this method will return qc. 

        Returns:
            pd.Series   A Pandas Series representing the corrected cone penetration (qt) of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if not hasattr(self, "_qc"):
            raise AttributeError(
                f"Cone penetration, qc, of {self.pointID} is not yet defined. This is needed to calculate the corrected cone penetration, qt."
            )
        if hasattr(self, "_qt"):
            return self._qt
        if not hasattr(self, "_area_ratio"):
            warnings.warn(
                f"Area ratio of {self.pointID} is not yet defined. qt will be assumed to be the same as qc for all calculations",
                RuntimeWarning
            )
            return self.qc
        
        s_qc = self.qc
        s_u2 = self.u2.fillna(0) if hasattr(self, "_u2") else 0
        s_qt = s_qc + (s_u2 / 1000) * (1 - self.area_ratio)
        s_qt.name = 'qt'
        self._qt = s_qt
        return self._qt

    @qt.deleter
    def qt(self):
        self.listener_dependency(['Rf', 'Qt', 'Fr', 'Bq', 'Ic'])
        if hasattr(self, "_qt"):
            del self._qt
    
    @property
    def Rf(self) -> pd.Series:
        """
        Gets the friction ratio of the CPT test point, Rf = (fs/qt) * 100%

        Dependent on the properties fs, and qt.
        This property will calculate the friction ratio the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the friction ratio (Rf) of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        # Check for errors
        dict_error_lookup = {
            '0': (f"Cone penetration, qc, and sleeve friction, fs, of {self.pointID} are not yet defined.", not hasattr(self, "_qc") and not hasattr(self, "_fs")),
            '1': (f"Cone penetration, qc, of {self.pointID} is not yet defined.", not hasattr(self, "_qc")),
            '2': (f"Sleeve friction, fs, of {self.pointID} is not yet defined.", not hasattr(self, "_fs")),
        }
        for k,v in dict_error_lookup.items():
            if v[1]:
                raise AttributeError(v[0] + "This is needed to calculate the friction ratio, Rf.")
        
        if hasattr(self, "_Rf"):
            return self._Rf
        else:
            s_fs = self._fs
            s_qt = self.qt if hasattr(self, "_area_ratio") else self._qc
            s_Rf = s_fs / (s_qt * 1000) * 100
            s_Rf.name = "Rf"
            self._Rf = s_Rf
            return self._Rf
    
    @Rf.deleter
    def Rf(self):
        del self._Rf

    @property
    def Qt(self) -> pd.Series:
        """
        Gets the normalized cone penetration of the CPT test point

        Qt = (qt - sv0) / sv0'

        Dependent on the property unit_weight, gwl, elevated_gwl, qc, and area_ratio.
        This property will calculate the normalized cone penetration the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_Qt"):
            return self._Qt
        
        if not hasattr(self, "_qc"):
            raise AttributeError(
                f"Cone penetration, qc, of {self.pointID} is not yet defined. This is needed to calculate the normalized cone penetration, Qt."
            )
        if not hasattr(self, "_unit_weight"):
            raise AttributeError(
                f"Unit weight of {self.pointID} is not yet defined. This is needed to calculate the total stress and normalized cone penetration, Qt."
            )
        if not hasattr(self, "_gwl") and not hasattr(self, "_elevated_gwl"):
            raise AttributeError(
                f"Groundwater level of {self.pointID} is not yet defined. This is needed to calculate the effective stress and normalized cone penetration, Qt."
            )
        
        s_qt = self.qt if hasattr(self, "_area_ratio") else self.qc
        s_total_stress = self.total_stress
        s_effective_stress = self.effective_stress

        s_norm_Qt = (s_qt * 1000 - s_total_stress) / s_effective_stress
        s_norm_Qt.name = "Normalized Cone Penetration, Qt"
        self._Qt = s_norm_Qt
            
        return self._Qt

    @Qt.deleter
    def Qt(self):
        del self._Qt
    
    @property
    def Fr(self) -> pd.Series:
        """
        Gets the normalized friction ratio of the CPT test point

        Fr = 100% * (fs / (qt - sv0))

        Dependent on the property unit_weight, fs, qc, and area_ratio.
        This property will calculate the normalized friction ratio the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_Fr"):
            return self._Fr
        
        if not hasattr(self, "_qc"):
            raise AttributeError(
                f"Cone penetration, qc, of {self.pointID} is not yet defined. This is needed to calculate the normalized friction ratio, Fr."
            )
        if not hasattr(self, "_unit_weight"):
            raise AttributeError(
                f"Unit weight of {self.pointID} is not yet defined. This is needed to calculate the total stress and normalized friction ratio, Fr."
            )
        
        s_fs = self.fs
        s_qt = self.qt if hasattr(self, "_area_ratio") else self.qc
        s_total_stress = self.total_stress

        s_norm_Fr = 100 * (s_fs / (s_qt * 1000 - s_total_stress))
        s_norm_Fr.name = "Normalized Friction Ratio, Fr"
        self._Fr = s_norm_Fr
            
        return self._Fr

    @Fr.deleter
    def Fr(self):
        del self._Fr

    @property
    def Bq(self) -> pd.Series:
        """
        Gets the porewater pressure ratio of the CPT test point

        Bq = (u2 - u0) / (qt - sv0)

        Dependent on the property unit_weight, fs, qc, and area_ratio.
        This property will calculate the normalized friction ratio the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_Bq"):
            return self._Bq
        
        if not hasattr(self, "_qc"):
            raise AttributeError(
                f"Cone penetration, qc, of {self.pointID} is not yet defined. This is needed to calculate the porewater pressure ratio, Bq."
            )
        if not hasattr(self, "_unit_weight"):
            raise AttributeError(
                f"Unit weight of {self.pointID} is not yet defined. This is needed to calculate the total stress and the porewater pressure ratio, Bq."
            )
        if not hasattr(self, "_gwl") and not hasattr(self, "_elevated_gwl"):
            raise AttributeError(
                f"Groundwater level of {self.pointID} is not yet defined. This is needed to calculate the porewater pressures."
            )
        
        s_qt = self.qt if hasattr(self, "_area_ratio") else self.qc
        if not hasattr(self, "_u2"):
            depth_index = s_qt.index.to_numpy()
            s_Bq = pd.Series(
                index=depth_index,
                data=map(lambda z: np.NaN, depth_index),
                name="Bq"
            )
        s_total_stress = self.total_stress
        if not hasattr(self, "_elevated_u0") and not hasattr(self, "_static_u0"):
            self._calculate_effective_stress()
        s_u2 = self.u2
        s_u0 = self.elevated_u0 if hasattr(self, "_elevated_u0") else self.static_u0

        s_Bq = (s_u2 - s_u0) / (s_qt * 1000 - s_total_stress)
        s_Bq.name = "Bq"
        self._Bq = s_Bq
            
        return self._Bq

    @Bq.deleter
    def Bq(self):
        del self._Bq

    @property
    def Ic(self) -> pd.Series:
        """
        Gets the soil behaviour of the CPT test point

        Ic = ( (3.47 - log Qt)^2 + (log Fr + 1.22)^2 )^0.5

        Dependent on the property Qt and Fr.
        This property will calculate the soil behaviour index the first time it is called, and store that value in the object instance of the class to be returned in future calls.

        Returns:
            pd.Series   A Pandas Series representing the total stress distribution of the CPT.
                        Index corresponds to the depth points of each probe recording.
        """
        if hasattr(self, "_Ic"):
            return self._Ic
        
        s_Qt = self.Qt
        s_Fr = self.Fr
        
        s_Ic = ( (3.47 - np.log10(s_Qt)) ** 2 + (np.log10(s_Fr) + 1.22) ** 2 ) ** 0.5
        s_Ic.name = "Ic"
        self._Ic = s_Ic
            
        return self._Ic
    
    @Ic.deleter
    def Ic(self):
        del self._Ic

    @property
    def classification(self) -> pd.DataFrame:
        """
        Gets the soil classification zones of the CPT point.

        Requires the method classify_soil() to be called firsthand.

        Returns:
            pd.Dataframe    A Pandas Dataframe representing the soil zone number, 
                            soil zone description, and USCS
        """
        if not hasattr(self, "_soil_classification_method"):
            raise AttributeError(
                "Soil classification has not been defined for this CPT point. " + \
                "Please call the method CPT.classify_soil() to calculate the soil zone."
            )
        
        if not hasattr(self, "_soil_classification"):
            self.classify_soil(self._soil_classification_method)
        
        return self._soil_classification
    
    @classification.deleter
    def classification(self):
        #del self._soil_classification_method
        if hasattr(self, "_soil_classification"):
            del self._soil_classification
        if hasattr(self, "_soil_classification_graph_data"):
            del self._soil_classification_graph_data
    
    @property
    def results(self) -> pd.DataFrame:
        """
        Returns the collated results of the CPT data in the form of a DataFrame.
        """
        list_order = [
            '_qc', '_fs', '_u2', '_qt', '_total_stress', 
            '_static_u0', '_elevated_u0', '_effective_stress',
            '_Rf', '_Bq', '_Qt', '_Fr', '_Ic'
        ]

        list_pdseries = list(filter(
            lambda pair: isinstance(pair[1], pd.Series) or isinstance(pair[1], pd.DataFrame),
            self.__dict__.items()
        ))
        list_orderedpd = [self.__dict__.get(k, None) for k in list_order]
        list_custompd = list(map(
            lambda pair: pair[1],
            filter(
                lambda pair: pair[0] not in list_order,
                list_pdseries
            )))

        joined_df = pd.concat(list_orderedpd + list_custompd, axis=1)
        return joined_df

    
    def calculate(self,
        area_ratio: float = None,
        unit_weight: float = None,
        gwl_depth: float = None,
        gwl_el: float = None,
        soil_classification_method: Literal[
            'Eslami Fellenius', 
            'Robertson et al 1986', 
            'Robertson et al 1986 (nonpiezo)', 
            'Robertson 1990'
        ] = None):
        """
        A convenient shorthand method to calculate all relevant CPT parameters.
        """

        if not any([area_ratio, unit_weight, gwl_depth, gwl_el, soil_classification_method]):
            raise ValueError('Please enter at least one parameter, e.g. unit weight, to calculate the CPT data.')
        
        if area_ratio is not None:
            self._area_ratio = area_ratio
            self.qt
        self.Rf
        if unit_weight is not None:
            self.unit_weight = unit_weight

        self.Fr
        
        if gwl_depth is not None:
            self.gwl = gwl_depth
        else:
            if hasattr(self, "_elevation"): 
                self.gwl = self.elevation - gwl_el
        if hasattr(self, "_gwl"):
            self.Bq
            self.Qt
            self.Ic
        
        if soil_classification_method is not None:
            self.classify_soil(soil_classification_method)
        elif hasattr(self, "_soil_classification_method"):
            self.classify_soil(self._soil_classification_method)
    
    def classify_soil(self, 
        method: Literal[
            'Eslami Fellenius', 
            'Robertson et al 1986', 
            'Robertson et al 1986 (nonpiezo)', 
            'Robertson 1990'
        ] = None,
        **kwargs
        ) -> None:
        """
        Classifies the CPT data into different soil classification numbers and zones based on 
        a recognized soil classification method.

        Args:
            method                  (string)
                The name/label of the soil classification method to choose.
                Default database has 4 choices:
                 - Eslami Fellenius                 fs vs. qE
                 - Robertson et al 1986             Bq vs. qt, Rf vs. qt
                 - Robertson et al 1986 (nonpiezo)  Rf vs. qt/qc
                 - Robertson 1990                   Fr vs. Qt, Bq vs. Qt
                You may also choose a custom soil classification method if connecting to an external database.
        
        Keyword Arguments:
            filepath                (string)
                The filepath to the external soil classification method database.
        """
        if method is None:
            method = getattr(self, "_soil_classification_method", None)
        if method is None:
            raise AttributeError("Soil classification method is not set.")

        if hasattr(self, "__helper_CPT_classification"):
            OBJ_CPT_SoilClassification = self.__helper_CPT_classification
        else:
            filepath_external_db = kwargs.get("filepath", None)
            OBJ_CPT_SoilClassification = CPT_SoilClassification(filepath_external_db)
            self.__helper_CPT_classification = OBJ_CPT_SoilClassification
        OBJ_CPT_SoilClassification.calculate_soil_classification(method, self)
    
    def plot_single_log(self, 
        ax: plt.Axes, 
        plot_by_elevation: bool = False, 
        plot_bar_values: bool = True,
        **kwargs) -> plt.Axes:
        """
        Plot the soil classification computation results.
        """
        if not hasattr(self, "_soil_classification"):
            raise AttributeError('Soil classification has not been computed.')
        
        if plot_by_elevation and not hasattr(self, "_elevation"):
            raise AttributeError('Elevation has not been set.')

        df_soil_classification = self._soil_classification
        dict_soilzones = CPT_SoilClassification.query_legend_soilzones(
            self._soil_classification_method)
        
        top = pd.Series(
            index=df_soil_classification.index, 
            data=df_soil_classification.index
        )
        dh = top.diff()
        dh.iloc[0] = dh.iloc[1]
        top = self.elevation - top if plot_by_elevation else top
        bottom = top - dh if plot_by_elevation else top + dh
        for idx in range(len(dict_soilzones)):

            k,v = list(dict_soilzones.items())[idx]
            width = idx + 1 if plot_bar_values else 1

            bool_filter_df = df_soil_classification["Soil Zone Number"] == k
            data_to_plot = df_soil_classification.loc[bool_filter_df]
            nrects = data_to_plot["Soil Zone Number"].size
            top_soilzone = top.loc[bool_filter_df]
            bottom_soilzone = bottom.loc[bool_filter_df]

            codes = np.ones(nrects * 5, int) * Path.LINETO
            codes[0::5] = Path.MOVETO
            codes[4::5] = Path.CLOSEPOLY

            verts = np.zeros((nrects * 5, 2))
            verts[0::5, 0] = 0
            verts[0::5, 1] = top_soilzone
            verts[1::5, 0] = width
            verts[1::5, 1] = top_soilzone
            verts[2::5, 0] = width
            verts[2::5, 1] = bottom_soilzone
            verts[3::5, 0] = 0
            verts[3::5, 1] = bottom_soilzone

            barpath = Path(verts, codes)
            ax.add_patch(PathPatch(
                barpath,
                facecolor=v[2],
                edgecolor=v[2]
            ))
        
        if plot_bar_values:
            ax.set_xlim(0, len(dict_soilzones))
            ax.xaxis.set_major_locator(MultipleLocator(1))
        else:
            ax.xaxis.set_visible(False)
        
        ax.grid(which='minor', axis='both', color='#DBDBDB', linestyle='--')
        ax.grid(which='major', axis='both', color='#515151', linestyle='--')
        ax.xaxis.tick_top()
        ax.set_title(self.pointID)
        ax.set_xlabel('Soil Zone Number')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_axisbelow(True)

        return ax

    def plot_soil_legend(self, subfig: plt.Figure):
        dict_soilzones = CPT_SoilClassification.query_legend_soilzones(
            self._soil_classification_method)
        labels = [fill(v[0], 20) for k,v in dict_soilzones.items()]
        handles = [
            Rectangle(
                (0, 0), 0.5, 1,
                facecolor=v[2],
                edgecolor=v[2]
            ) 
            for k,v in dict_soilzones.items()
        ]
        subfig.legend(
            handles, labels, 
            title="LEGEND", fontsize='medium',
            loc='center'
        )

        return subfig

    def plot(self):
        """
        Plot the CPT results.

        This method will plot the following:
         - Cone penetration (qt or qc)
         - Friction ratio (Rf)
         - Pore pressure (u2, u0 (static condition), u0 (elevated condition))
         - Pore pressure ratio, Bq
         - Soil Classification, found using the classify_soil() method
        This method will automatically omit drawing the respective graphs if no data is available,
        e.g. if it is not a piezocone or if soil zone numbers are not yet calculated.
        However, this method will fail if it can't get the cone penetration or friction ratio.

        Returns:
            Matplotlib Figure
                A Matplotlib Figure object representing the complete visual.
        """

        def set_mpl_param(
            data_to_plot: pd.Series, 
            depth_plots: pd.Index, 
            color: str, 
            linestyle: str,
            lim: 'Tuple[Union[int, float, None], Union[int, float, None]]',
            round: 'Union[int, float]'
            ):
            
            dict_kwargs_plot = {
                'color': color,
                'linestyle': linestyle,
                'linewidth': 2.0
            }
            dict_params_axes = {}
            if lim[0] is not None and lim[1] is not None:
                # Set absolute limit on x-axis values
                dict_params_axes["xlim"] = lim
                dict_params_axes["majorticker"] = MultipleLocator(round)
                dict_params_axes["minorticker"] = MultipleLocator(round / 5)
            else:
                x_min = lim[0] if lim[0] is not None else \
                    floor(data_to_plot.min() / round) * round
                x_max = lim[1] if lim[1] is not None else \
                    ceil(data_to_plot.max() / round) * round
                range_x = x_max - x_min
                major_unit = 1 if range_x <= 5 else \
                    2 if range_x <= 10 else \
                    5 if range_x <= 25 else \
                    10 if range_x <= 50 else 20
                dict_params_axes["xlim"] = (x_min, x_max)
                if range_x < 100:
                    dict_params_axes["majorticker"] = MultipleLocator(major_unit)
                    dict_params_axes["minorticker"] = MultipleLocator(major_unit / 5)
                else:
                    dict_params_axes["majorticker"] = MaxNLocator(5)
                    dict_params_axes["minorticker"] = MaxNLocator(25)

            return (data_to_plot, depth_plots, dict_kwargs_plot, dict_params_axes)

        magnify_qc = 10

        #Plot qt, Rf, u2/u0, Bq, soil zone
        dict_plot_data = {
            "qt": {
                "data": [], 
                "axis_text": "Cone penetration, {}, MPa".format(
                    "qt" if hasattr(self, "_area_ratio") else "qc"
                ),
                "labels": ["x1 scale", f"x{magnify_qc} scale"]
            },
            "Rf": {
                "data": [], 
                "axis_text": "Friction ratio, Rf, %"
            }
        }
        depth_points = self.qc.index
        dict_plot_data["qt"]["data"].append(
            set_mpl_param(
                self.qt if hasattr(self, "_area_ratio") else self.qc,
                depth_points,
                "black", "-", (0,20), 5
            )
        )
        dict_plot_data["qt"]["data"].append(
            set_mpl_param(
                self.qt * magnify_qc if hasattr(self, "_area_ratio") else self.qc * magnify_qc,
                depth_points,
                "red", "--", (0,20), 5
            )
        )
        dict_plot_data["Rf"]["data"].append(
            set_mpl_param(
                self.Rf,
                depth_points,
                "black", "-", (0, None), 5
            )
        )

        if hasattr(self, "_u2"):
            dict_plot_data["u2"] = {
                "data": [],
                "axis_text": "Pore Pressure, kPa",
                "labels": []
            }
            dict_plot_data["u2"]["data"].append(
                set_mpl_param(
                    self.u2,
                    depth_points,
                    "blue", "-", (None, None), 200
                )
            )
            dict_plot_data["u2"]["labels"].append("u2")
            if hasattr(self, "_gwl"):
                dict_plot_data["u2"]["data"].append(
                    set_mpl_param(
                        self.static_u0,
                        depth_points,
                        "deepskyblue", ":", 
                        dict_plot_data["u2"]["data"][0][3]["xlim"], 
                        200
                    )
                )
                dict_plot_data["u2"]["labels"].append("u0")
                if hasattr(self, "_unit_weight"):
                    dict_plot_data["Bq"] = {
                        "data": [],
                        "axis_text": "Pore Pressure Ratio, Bq"
                    }
                    dict_plot_data["Bq"]["data"].append(
                        set_mpl_param(
                            self.Bq,
                            depth_points,
                            "black", "-", 
                            (-0.2, 0.8), 
                            0.2
                        )
                    )
            if hasattr(self, "_elevated_gwl"):
                dict_plot_data["u2"]["data"].append(
                    set_mpl_param(
                        self.elevated_u0,
                        depth_points,
                        "green", ":", 
                        dict_plot_data["u2"]["data"][0][3]["xlim"], 
                        200
                    )
                )
                dict_plot_data["u2"]["labels"].append("u0 (elevated)")

        fig = plt.figure(figsize=(15, 8), dpi=144)
        subplots_position = (0.1, 0.05, 0.99, 0.85)
        if hasattr(self, "_soil_classification"):
            subfigs = fig.subfigures(1, 2, width_ratios=[0.8, 0.2])
            axs = subfigs[0].subplots(1, len(dict_plot_data) + 1, sharey=True)
            subfigs[0].subplots_adjust(*subplots_position)
        else:
            axs = fig.subplots(1, len(dict_plot_data), sharey=True)
            fig.subplots_adjust(*subplots_position)
        
        axs[0].set_ylabel("Depth, m")

        subfigs[0].suptitle(self.pointID, fontsize='xx-large', fontweight='black')

        y_min = 0
        y_max = ceil(depth_points.max())
        axs[0].set_ylim(y_max, y_min)
        major_unit = 0.1 if y_max <= 1 else 1 if y_max <= 10 else 5
        axs[0].yaxis.set_major_locator(MultipleLocator(major_unit))
        axs[0].yaxis.set_minor_locator(MultipleLocator(
            major_unit / 10
        ))

        idx = 0
        for k,v in dict_plot_data.items():

            # Set axes formatting
            ax = axs[idx]
            ax.grid(which='minor', axis='both', color='#DBDBDB', linestyle='--')
            ax.grid(which='major', axis='both', color='#515151', linestyle='--')
            ax.xaxis.tick_top()
            ax.set_xlabel(v["axis_text"])
            ax.xaxis.set_label_position('top') 

            # Plot all data on the same x-axis
            for plot in v["data"]:
                *mpl_plot, plot_params, axes_params = plot
                ax.plot(*mpl_plot, **plot_params)
            axes_params = v["data"][0][3]
            ax.set_xlim(axes_params["xlim"])
            ax.xaxis.set_major_locator(axes_params["majorticker"])
            ax.xaxis.set_minor_locator(axes_params["minorticker"])

            # Special case: draw legend for qt plot
            if k=="qt":
                ax.legend(labels=v["labels"], loc='lower right')

            # Special case: draw legend for u2 plot
            if k=="u2" and len(v["data"]) > 1:
                ax.legend(labels=v["labels"], loc='lower right')
            
            idx += 1
        
        secaxis = axs[0].secondary_xaxis(1.05, 
            functions=(lambda x: x/magnify_qc, lambda x: x*magnify_qc)
        )
        secaxis.set_color('red')
        axs[0].xaxis.labelpad = 30

        if hasattr(self, "_soil_classification"):
            axs[len(dict_plot_data)] = self.plot_single_log(
                axs[len(dict_plot_data)], set_ylim=False
            )
            subfigs[1] = self.plot_soil_legend(subfigs[1])
        
        return fig

    def plot_graph(self):
        fig = CPT_SoilClassification.plot_graph(self)
        return fig