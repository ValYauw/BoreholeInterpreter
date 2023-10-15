import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import unittest

from classes import GeotechPoint
from classes.GeotechPoint import PointDataset

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np

class TestGeotechPoint(unittest.TestCase):

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

    def test_simplepoint_1(self):
        """
        Unit test to initialize the creation of a simple geotechnical point, without the definition of the point's elevation.
        """
        testPoint = GeotechPoint("BH-1", 100, 100)
        expected_str = """Geotechnical Point\nPoint ID :\tBH-1\nX:\t\t100.0\nY:\t\t100.0"""
        self.assertEqual(expected_str, str(testPoint))

    def test_simplepoint_2(self):
        """
        Unit test to initialize the creation of a simple geotechnical point.
        """
        testPoint = GeotechPoint("BH-1", 100, 100, 10)
        expected_str = """Geotechnical Point\nPoint ID :\tBH-1\nX:\t\t100.0\nY:\t\t100.0\nElevation:\t10.0"""
        self.assertEqual(expected_str, str(testPoint))
    
    def test_simplepoint_3(self):
        """
        Unit test to initialize the creation of a simple geotechnical point.
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        expected_str = """Geotechnical Point\nPoint ID :\tBH-1\nX:\t\t249,730.6\nY:\t\t9,231,020.1\nElevation:\t57.0"""
        self.assertEqual(expected_str, str(testPoint))
    
    def test_simplepoint_4(self):
        """
        Unit test to test the behaviour when initializing a geotechnical point without coordinates.
        """
        testPoint = GeotechPoint("BH-1")
        self.assertIsNone(testPoint.xCoord)
        self.assertIsNone(testPoint.yCoord)
        expected_str = """Geotechnical Point\nPoint ID :\tBH-1"""
        self.assertEqual(expected_str, str(testPoint))
    
    def test_initialize_stratigraphy_1(self):
        """
        Unit test to test the initialization of the stratigraphy in a simple geotechnical point.
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            [0, 5, "CLAY", "Fat Clay", "CH"],
            [5, 10, "SAND", "Clayey Sand", "SC"]
        ]

        expected_df_data = {
            "Depth from": [0.0, 5.0],
            "Depth to": [5.0, 10.0],
            "Soil Type": ["CLAY", "SAND"],
            "Soil Description": ["Fat Clay", "Clayey Sand"],
            "USCS": ["CH", "SC"]
        }
        expected_df = pd.DataFrame(expected_df_data)

        self.assertIsInstance(testPoint.stratigraphy, pd.DataFrame)
        self.assertIsInstance(testPoint.stratigraphy, PointDataset)
        self.assertEqual(expected_df.shape, testPoint.stratigraphy.shape)
        self.assertEqualsDataframe(expected_df, testPoint.stratigraphy)
    
    def test_initialize_stratigraphy_2(self):
        """
        Unit test to test the initialization of the stratigraphy in a more practical geotechnical point.
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
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

        self.assertIsInstance(testPoint.stratigraphy, pd.DataFrame)
        self.assertIsInstance(testPoint.stratigraphy, PointDataset)
        self.assertEqual(expected_df.shape, testPoint.stratigraphy.shape)
        self.assertEqualsDataframe(expected_df, testPoint.stratigraphy)

    def test_wrong_input_stratigraphy_convert_to_numeric(self):
        """
        Unit test of a case of wrong input for stratigraphy: non-numeric depth points that can be converted to numeric
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            ["0", "3", "CLAY", "Fat Clay", "CH"],
            ["3", "10", "SAND", "Sandy Clay", "SC"]
        ]

        self.assertIsInstance(testPoint.stratigraphy, pd.DataFrame)
        self.assertIsInstance(testPoint.stratigraphy, PointDataset)
        self.assertEqual(testPoint.stratigraphy.loc[:, "Depth from"].dtype, np.float32)
        self.assertEqual(testPoint.stratigraphy.loc[:, "Depth to"].dtype, np.float32)

    def test_wrong_input_stratigraphy_1(self):
        """
        Unit test of a case of wrong input for stratigraphy: non-numeric depth points
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        with self.assertRaises(ValueError):
            testPoint.stratigraphy = [
                ["a", 5, "CLAY", "Fat Clay", "CH"],
                [3, 10, "SAND", "Sandy Clay", "SC"]
            ]
    
    def test_wrong_input_stratigraphy_2(self):
        """
        Unit test of a case of wrong input for stratigraphy: intersecting depth points
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        with self.assertRaises(ValueError):
            testPoint.stratigraphy = [
                [0, 5, "CLAY", "Fat Clay", "CH"],
                [3, 10, "SAND", "Sandy Clay", "SC"]
            ]
    
    def test_wrong_input_stratigraphy_3(self):
        """
        Unit test of a case of wrong input for stratigraphy: intersecting depth points
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        with self.assertRaises(ValueError):
            testPoint.stratigraphy = [
                [0, 1, "CLAY", "Fat Clay", "CH"],
                [11, 10, "SAND", "Sandy Clay", "SC"]
            ]
    
    def test_gap_in_dataset_record(self):
        """
        Unit test where a dataset with a "jump"/gap in the records is set
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            [0, 1, "CLAY", "Fat Clay", "CH"],
            [2, 10, "SAND", "Sandy Clay", "SC"]
        ]
        self.assertIsInstance(testPoint, GeotechPoint)
    
    def test_stratigraphy_depthPoints(self):
        """
        Unit test to check for potential side-effects when accessing stratigraphy.depthPoints
        """
        testPoint = GeotechPoint("BH-1", 249730.567, 9231020.145, 56.956)
        testPoint.stratigraphy = [
            [0, 2, "FILL", "FILL; Fat Clay", "CH"],
            [2, 5, "FILL", "FILL; Sandy Clay", "SC"],
            [5, 7, "SAND", "Well-graded Sand, dry, subangular to angular, brown", "SW"],
            [6, None, "", "laminations between 10-15mm", ""],
            [7, 10, "SAND", "Well-graded Sand, dry, subangular to angular, brown", "SW"],
            [9, None, "", "grades grey", ""],
            [10, 15, "GRAVEL", "Poorly-graded Gravel, dry, subangular to angular, grey", "GP"]
        ]

        expected_stratigraphy_depthPoints = [0.0, 2.0, 5.0, 7.0, 10.0, 15.0]

        expected_df_data = {
            "Depth from": [0.0, 2.0, 5.0, 6.0, 7.0, 9.0, 10.0],
            "Depth to": [2.0, 5.0, 7.0, np.NaN, 10.0, np.NaN, 15],
            "Soil Type": ["FILL", "FILL", "SAND", "", "SAND", "", "GRAVEL"],
            "Soil Description": [
                "FILL; Fat Clay",
                "FILL; Sandy Clay",
                "Well-graded Sand, dry, subangular to angular, brown",
                "laminations between 10-15mm",
                "Well-graded Sand, dry, subangular to angular, brown",
                "grades grey",
                "Poorly-graded Gravel, dry, subangular to angular, grey"
            ],
            "USCS": ["CH", "SC", "SW", "", "SW", "", "GP"]
        }
        expected_df = pd.DataFrame(expected_df_data)
        
        self.assertListEqual(expected_stratigraphy_depthPoints, testPoint.stratigraphy.depthPoints)
        self.assertEqualsDataframe(expected_df, testPoint.stratigraphy)
        

if __name__ == "__main__":
    unittest.main(exit=False)