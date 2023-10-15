import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import unittest

from classes import Borehole
from classes.GeotechPoint import PointDataset

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np

class TestBorehole(unittest.TestCase):

    def assertEqualsDataframe(self, df1, df2):
        """
        Auxiliary function to assert that the data and column names of two dataframes are equal
        """
        try:
            assert_frame_equal(
                df1, df2,
                check_dtype=False,
                check_index_type=False,
                check_column_type=False,
                check_frame_type=False,
                check_categorical=False,
                check_names=True
            )
        except AssertionError:
            self.fail("AssertionError: Resulting DataFrame does not match the expected result")
    
    def assertEqualsSeries(self, s1, s2):
        """
        Auxiliary function to assert that the data of two Series are equal
        """
        try:
            assert_series_equal(
                s1, s2,
                check_dtype=False,
                check_index_type=False,
                check_series_type=False,
                check_categorical=False,
                check_names=True
            )
        except AssertionError:
            self.fail("AssertionError: Resulting Series of depth points does not match the expected result")

    def test_initialize_stratigraphy(self):
        """
        Unit test to test the initialization of the stratigraphy in a more practical geotechnical point.
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            [0, 2, "FILL", "FILL; Fat Clay", "CH"],
            [2, 5, "FILL", "FILL; Sandy Clay", "SC"],
            [5, 6, "SAND", "Well-graded Sand, dry, subangular to angular, brown", "SW"],
            [6, 7, "", "Core loss", ""],
            [7, 10, "SAND", "Well-graded Sand, dry, subangular to angular, brown", "SW"],
            [9, None, "", "grades grey", ""],
            [10, 15, "GRAVEL", "Poorly-graded Gravel, dry, subangular to angular, grey", "GP"]
        ]

        expected_df_data = {
            "Depth from": [0.0, 2.0, 5.0, 6.0, 7.0, 9.0, 10.0],
            "Depth to": [2.0, 5.0, 6.0, 7.0, 10.0, np.nan, 15],
            "Soil Type": ["FILL", "FILL", "SAND", "", "SAND", "", "GRAVEL"],
            "Soil Description": [
                "FILL; Fat Clay",
                "FILL; Sandy Clay",
                "Well-graded Sand, dry, subangular to angular, brown",
                "Core loss",
                "Well-graded Sand, dry, subangular to angular, brown",
                "grades grey",
                "Poorly-graded Gravel, dry, subangular to angular, grey"
            ],
            "USCS": ["CH", "SC", "SW", "", "SW", "", "GP"]
        }
        expected_df = pd.DataFrame(expected_df_data)

        self.assertEqualsDataframe(expected_df, testPoint.stratigraphy)

        expected_holedepth = 15.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

    def test_wrong_input_stratigraphy_convert_to_numeric(self):
        """
        Unit test of a case of wrong input for stratigraphy: non-numeric depth points that can be converted to numeric
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            ["0", "3", "CLAY", "Fat Clay", "CH"],
            ["3", "10", "SAND", "Sandy Clay", "SC"]
        ]

    def test_wrong_input_stratigraphy_1(self):
        """
        Unit test of a case of wrong input for stratigraphy: non-numeric depth points
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        with self.assertRaises(ValueError):
            testPoint.stratigraphy = [
                ["a", 5, "CLAY", "Fat Clay", "CH"],
                [3, 10, "SAND", "Sandy Clay", "SC"]
            ]
    
    def test_wrong_input_stratigraphy_2(self):
        """
        Unit test of a case of wrong input for stratigraphy: intersecting depth points
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        with self.assertRaises(ValueError):
            testPoint.stratigraphy = [
                [0, 5, "CLAY", "Fat Clay", "CH"],
                [3, 10, "SAND", "Sandy Clay", "SC"]
            ]
    
    def test_wrong_input_stratigraphy_3(self):
        """
        Unit test of a case of wrong input for stratigraphy: intersecting depth points
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        with self.assertRaises(ValueError):
            testPoint.stratigraphy = [
                [0, 1, "CLAY", "Fat Clay", "CH"],
                [11, 10, "SAND", "Sandy Clay", "SC"]
            ]

    def test_borehole_1(self):
        """
        Unit test to initialize the creation of a simple borehole.
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        expected_str = """Borehole\nPoint ID :\tBH-1\nX:\t\t249,730.6\nY:\t\t9,231,020.1\nElevation:\t57.0"""
        self.assertEqual(expected_str, str(testPoint))
    
    def test_borehole_2(self):
        """
        Unit test to initialize the creation of a simple borehole.
        """
        testPoint = Borehole("BH-1")
        self.assertIsNone(testPoint.xCoord)
        self.assertIsNone(testPoint.yCoord)
        expected_str = """Borehole\nPoint ID :\tBH-1"""
        self.assertEqual(expected_str, str(testPoint))
    
    def test_borehole_uninitialized_properties(self):
        """
        Unit test to check the behaviour when properties are uninitialized
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.elevation
        with self.assertRaises(AttributeError):
            testPoint.stratigraphy
        with self.assertRaises(AttributeError):
            testPoint.sampling
        with self.assertRaises(AttributeError):
            testPoint.consistency_density
        with self.assertRaises(AttributeError):
            testPoint.moisture
        with self.assertRaises(AttributeError):
            testPoint.gwl
        with self.assertRaises(AttributeError):
            testPoint.gsi
        with self.assertRaises(AttributeError):
            testPoint.weathering
        with self.assertRaises(AttributeError):
            testPoint.rockStrength
        with self.assertRaises(AttributeError):
            testPoint.fractureFrequency
        with self.assertRaises(AttributeError):
            testPoint.defectDesc
        with self.assertRaises(AttributeError):
            testPoint.coring

    def test_borehole_init_sampling_1(self):
        """
        Unit test to initialize the sampling data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.sampling = [
            [0.5, 1.0, 0.3, "SPT", "(5,4,2) N=6", 6],
            [1.5, 2.0, 0.5, "SPT", "(1,3,2) N=5", 5],
            [2.5, 3.0, 0.0, "SPT", "(0,HW), HW", "HW"],
            [20.0, 20.5, 0.5, "SPT", "(25,30,12/20) N>50", ">50"],
            [21.0, 21.5, 0.5, "SPT", "(25,30,HB) HB", "HB"]
        ]
        expected_df_data = {
            "Depth from": [0.5, 1.5, 2.5, 20.0, 21.0],
            "Depth to": [1.0, 2.0, 3.0, 20.5, 21.5],
            "Recovery": [0.3, 0.5, 0.0, 0.5, 0.5],
            "Sample Type": ["SPT", "SPT", "SPT", "SPT", "SPT"],
            "Sampling Description": ["(5,4,2) N=6", "(1,3,2) N=5", "(0,HW), HW", "(25,30,12/20) N>50", "(25,30,HB) HB"],
            "SPT-N Value": ["6", "5", "HW", ">50", "HB"],
            "Pocket Penetrometer": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "Torvane": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "Vane Shear Test (Peak)": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "Vane Shear Test (Residual)": [np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        expected_df = pd.DataFrame(expected_df_data)

        self.assertEqualsDataframe(expected_df, testPoint.sampling)

        expected_holedepth = 21.5
        self.assertEqual(testPoint.holedepth, expected_holedepth)
    
    def test_borehole_init_sampling_2(self):
        """
        Unit test to initialize the sampling data of a simple borehole.
        """
        self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.sampling = [
            [0.5, 1.0, 0.3, "SPT", "(5,4,2) N=6", 6],
            [1.5, 2.0, 0.5, "UDS", "PP=200, 300, 250 kPa; TV: 100 kPa", None, 200, 250],
            [2.5, 3.0, 0.5, "VST", "Vs=300 kPa, Vr=50 kPa", None, None, None, 300, 50],
        ]
        expected_df_data = {
            "Depth from": [0.5, 1.5, 2.5],
            "Depth to": [1.0, 2.0, 3.0],
            "Recovery": [0.3, 0.5, 0.5],
            "Sample Type": ["SPT", "UDS", "VST"],
            "Sampling Description": ["(5,4,2) N=6", "PP=200, 300, 250 kPa; TV: 100 kPa", "Vs=300 kPa, Vr=50 kPa"],
            "SPT-N Value": ["6.0", "", ""],
            "Pocket Penetrometer": [np.nan, 200.0, np.nan],
            "Torvane": [np.nan, 250.0, np.nan],
            "Vane Shear Test (Peak)": [np.nan, np.nan, 300.0],
            "Vane Shear Test (Residual)": [np.nan, np.nan, 50.0]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.sampling)

        expected_holedepth = 3.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)
    
    def test_borehole_init_consistency_density(self):
        """
        Unit test to initialize the consistency/density data of a simple borehole.
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.consistency_density = [
            [0, 2, "VS-S"],
            [2, 10, "MD"]
        ]
        expected_df_data = {
            "Depth from": [0.0, 2.0],
            "Depth to": [2.0, 10.0],
            "Consistency/Density": ["VS-S", "MD"]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.consistency_density)

        expected_holedepth = 10.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

    def test_borehole_init_moisture(self):
        """
        Unit test to initialize the moisture data of a simple borehole.
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.moisture = [
            [0, 2, "M"],
            [2, 10, "W"]
        ]
        expected_df_data = {
            "Depth from": [0.0, 2.0],
            "Depth to": [2.0, 10.0],
            "Moisture": ["M", "W"]
        }
        expected_df = pd.DataFrame(expected_df_data)

        self.assertEqualsDataframe(expected_df, testPoint.moisture)

        expected_holedepth = 10.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

    def test_borehole_init_gwl(self):
        """
        Unit test to initialize the GWL data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.gwl = [
            [1, "XX Date XXXX"]
        ]
        expected_df_data = {
            "Depth from": [1.0],
            "Depth to": [np.nan],
            "GWL": ["XX Date XXXX"]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.gwl)

        expected_holedepth = 0.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

    def test_borehole_init_gsi(self):
        """
        Unit test to initialize the GSI data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.gsi = [
            [10, 15, 30],
            [15, 50, 80]
        ]
        expected_df_data = {
            "Depth from": [10.0, 15.0],
            "Depth to": [15.0, 50.0],
            "GSI": [30, 80]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.gsi)

        expected_holedepth = 50.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

    def test_borehole_init_weathering(self):
        """
        Unit test to initialize the weathering data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.weathering = [
            [10, 15, "EW"],
            [15, 50, "FR"]
        ]
        expected_df_data = {
            "Depth from": [10.0, 15.0],
            "Depth to": [15.0, 50.0],
            "Weathering": ["EW", "FR"]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.weathering)

        expected_holedepth = 50.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)
    
    def test_borehole_init_rockStrength(self):
        """
        Unit test to initialize the rock strength data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.rockStrength = [
            [10, 15, "VL"],
            [15, 50, "L"]
        ]
        expected_df_data = {
            "Depth from": [10.0, 15.0],
            "Depth to": [15.0, 50.0],
            "Rock Strength": ["VL", "L"]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.rockStrength)

        expected_holedepth = 50.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)
    
    def test_borehole_init_fractureFrequency(self):
        """
        Unit test to initialize the fracture frequency data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.fractureFrequency = [
            [10, 15, 5],
            [15, 50, 0]
        ]
        expected_df_data = {
            "Depth from": [10.0, 15.0],
            "Depth to": [15.0, 50.0],
            "Fracture Frequency": [5.0, 0.0]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.fractureFrequency)

        expected_holedepth = 50.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

    def test_borehole_init_defectDesc(self):
        """
        Unit test to initialize the defect description data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.defectDesc = [
            [10, None, "Joint 10/350"],
            [12, None, "Vein 20/40"]
        ]
        expected_df_data = {
            "Depth from": [10.0, 12.0],
            "Depth to": [np.nan, np.nan],
            "Defect Description": ["Joint 10/350", "Vein 20/40"]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.defectDesc)

        expected_holedepth = 0.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)
    
    def test_borehole_init_coring(self):
        """
        Unit test to initialize the coring data of a simple borehole.
        """
        #self.maxDiff = None
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.coring = [
            [10, 15, 50, 50, 30],
            [15, 50, 100, 100, 100]
        ]
        expected_df_data = {
            "Depth from": [10.0, 15.0],
            "Depth to": [15.0, 50.0],
            "TCR": [50.0, 100.0],
            "RQD": [50.0, 100.0],
            "SCR": [30.0, 100.0]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertEqualsDataframe(expected_df, testPoint.coring)

        expected_holedepth = 50.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

    def test_set_borehole_holedepth(self):
        """
        Unit test to check the autoatic updating of hole depth upon setting of certain properties.
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            [0, 3, "CLAY", "Fat CLAY", "CH"],
            [3, 10, "MUDSTONE", "MUDSTONE", ""],
            [10, 12, "COAL", "COAL", ""]
        ]

        expected_holedepth = 12.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

        testPoint.gsi = [
            [3, 8, 30],
            [8, 10, 45],
            [10, 12, 30],
            [12, 15, 50],
            [15, 20, 70],
            [20, 25, 80]
        ]

        expected_holedepth = 25.0
        self.assertEqual(testPoint.holedepth, expected_holedepth)

        testPoint.fractureFrequency = [
            [3, 8, 30],
            [8, 10, 20],
            [10, 12, 30],
            [12, 15, 10]
        ]

        self.assertEqual(testPoint.holedepth, expected_holedepth)

        testPoint.rockStrength = [
            [3, 10, "VL"],
            [10, 12, "EL"],
            [12, 15, "VL-L"],
            [15, 20, "L"],
            [20, 25, "L-M"]
        ]

        self.assertEqual(testPoint.holedepth, expected_holedepth)        

    def test_merge_datasets(self):
        """
        Unit test to test the merge_datasets() method
        """
        testPoint = Borehole("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            [0, 3, "CLAY", "Fat CLAY", "CH"],
            [3, 10, "MUDSTONE", "MUDSTONE", ""],
            [10, 12, "COAL", "COAL", ""],
            [12, 25, "SANDSTONE", "SANDSTONE", ""]
        ]
        testPoint.gsi = [
            [3, 8, 30],
            [8, 10, 45],
            [10, 12, 30],
            [12, 15, 50],
            [15, 20, 70],
            [20, 25, 80]
        ]
        testPoint.fractureFrequency = [
            [3, 8, 30],
            [8, 10, 20],
            [10, 12, 30],
            [12, 15, 10],
            [15, 25, 0]
        ]
        testPoint.rockStrength = [
            [3, 10, "VL"],
            [10, 12, "EL"],
            [12, 15, "VL-L"],
            [15, 20, "L"],
            [20, 25, "L-M"]
        ]

        result = testPoint.merge_datasets(["stratigraphy", "gsi", "fractureFrequency", "rockStrength"])

        expected_df_depthPoints = [0.0, 3.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]

        expected_df_data = {
            "Depth from": [0.0, 3.0, 8.0, 10.0, 12.0, 15.0, 20.0],
            "Depth to": [3.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0],
            "Soil Type": ["CLAY", "MUDSTONE", "MUDSTONE", "COAL", "SANDSTONE", "SANDSTONE", "SANDSTONE"],
            "Soil Description": ["Fat CLAY", "MUDSTONE", "MUDSTONE", "COAL", "SANDSTONE", "SANDSTONE", "SANDSTONE"],
            "USCS": ["CH", "", "", "", "", "", ""],
            "GSI": [np.nan, 30, 45, 30, 50, 70, 80],
            "Fracture Frequency": [np.nan, 30.0, 20.0, 30.0, 10.0, 0.0, 0.0],
            "Rock Strength": [np.nan, "VL", "VL", "EL", "VL-L", "L", "L-M"]
        }
        expected_df = pd.DataFrame(expected_df_data)

        self.maxDiff = None

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(result, PointDataset)
        self.assertListEqual(expected_df_depthPoints, result.depthPoints)
        self.assertEqualsDataframe(expected_df, result)

if __name__ == "__main__":
    unittest.main(exit=False)