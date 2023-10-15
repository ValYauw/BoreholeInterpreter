import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import unittest
from unittest.mock import patch
import builtins

from classes import CPT
from classes.CPT_SoilClassification import CPT_SoilClassification

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np

class TestCPT(unittest.TestCase):

    def assertEqualsDataframe(self, df1, df2, msg=None):
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
            self.fail(
                "AssertionError: Resulting DataFrame does not match the expected result" 
                if msg==None else msg
            )
    
    def assertEqualsSeries(self, s1, s2, msg=None):
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
            self.fail(
                "AssertionError: Resulting Series of depth points does not match the expected result" 
                if msg==None else msg
            )

    def test_Eslami_Fellenius_1(self):
        """
        Unit test to test soil classification (Eslami Fellenius)
        """
        testPoint = CPT("CPT-1", 0, 0)
        test_data = [
            [0, 0.8, 3, 0],
            [0.1, 1, 100, 0],
            [0.2, 2.8, 100, 0],
            [0.3, 1.5, 3, 0],
            [0.4, 90, 2, 0],
            [0.5, 0.6, 20, 0],
            [0.6, 0.4, 700, 0],
            [0.7, 8, 200, 0],
            [0.8, 10, 9, 0],
            [0.9, 0.1, 2, 0],
            [1, 100, 3, 0],
            [1.1, 10, 1, 0],
            [1.2, 0.8, 1, 0],
            [1.3, 1, 1000, 0],
            [1.4, 0.1, 0.05, 0]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 1.0

        testPoint.classify_soil("Eslami Fellenius")

        results_method, results_soilzones, results_graphdata = (
            testPoint._soil_classification_method, 
            testPoint._soil_classification, 
            testPoint._soil_classification_graph_data
        )

        self.assertEqual(results_method, "Eslami Fellenius")

        expected_depths = [
            0, 0.1, 0.2, 0.3, 0.4, 
            0.5, 0.6, 0.7, 0.8, 0.9, 
            1, 1.1, 1.2, 1.3, 1.4
        ]
        expected_qE = [
            0.8, 1, 2.8, 1.5, 90, 
            0.6, 0.4, 8, 10, 0.1, 
            100, 10, 0.8, 1, 0.1
        ]
        expected_fs = [
            3, 100, 100, 3, 2, 
            20, 700, 200, 9, 2, 
            3, 1, 1, 1000, 0.05
        ]
        expected_empty = [
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        ]
        expected_df_graphdata = pd.DataFrame(
            {
                "X_graph_1": expected_fs,
                "Y_graph_1": expected_qE,
                "X_graph_2": expected_empty,
                "Y_graph_2": expected_empty
            },
            index=expected_depths,
            dtype=np.float32
        )
        self.assertEqualsDataframe(results_graphdata, expected_df_graphdata)

        expected_zones = [
            '1', '2', '3', '4', '5', 
            '2', '2', '4', '5', '1', 
            '5', '5', '1', '2', pd.NA
        ]
        results_zones = results_soilzones["Soil Zone Number"].tolist()
        self.assertListEqual(
            results_zones, 
            expected_zones
        )
    
    @patch('builtins.print')
    def test_Eslami_Fellenius_2(self, print_):
        """
        Unit test to test soil classification (Eslami Fellenius) 
        (nonpiezocone case)
        """
        testPoint = CPT("CPT-1", 0, 0)
        test_data = [
            [0, 0.8, 3],
            [0.1, 1, 100],
            [0.2, 2.8, 100],
            [0.3, 1.5, 3],
            [0.4, 90, 2],
            [0.5, 0.6, 20],
            [0.6, 0.4, 700],
            [0.7, 8, 200],
            [0.8, 10, 9],
            [0.9, 0.1, 2],
            [1, 100, 3],
            [1.1, 10, 1],
            [1.2, 0.8, 1],
            [1.3, 1, 1000],
            [1.4, 0.1, 0.05]
        ]
        testPoint.raw_data = test_data

        testPoint.classify_soil("Eslami Fellenius")
        assert print_.called

        results_method, results_soilzones, results_graphdata = (
            testPoint._soil_classification_method, 
            testPoint._soil_classification, 
            testPoint._soil_classification_graph_data
        )

        self.assertEqual(results_method, "Eslami Fellenius")

        expected_depths = [
            0, 0.1, 0.2, 0.3, 0.4, 
            0.5, 0.6, 0.7, 0.8, 0.9, 
            1, 1.1, 1.2, 1.3, 1.4
        ]
        expected_qE = [
            0.8, 1, 2.8, 1.5, 90, 
            0.6, 0.4, 8, 10, 0.1, 
            100, 10, 0.8, 1, 0.1
        ]
        expected_fs = [
            3, 100, 100, 3, 2, 
            20, 700, 200, 9, 2, 
            3, 1, 1, 1000, 0.05
        ]
        expected_empty = [
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        ]
        expected_df_graphdata = pd.DataFrame(
            {
                "X_graph_1": expected_fs,
                "Y_graph_1": expected_qE,
                "X_graph_2": expected_empty,
                "Y_graph_2": expected_empty
            },
            index=expected_depths,
            dtype=np.float32
        )
        self.assertEqualsDataframe(results_graphdata, expected_df_graphdata)

        expected_zones = [
            '1', '2', '3', '4', '5', 
            '2', '2', '4', '5', '1', 
            '5', '5', '1', '2', pd.NA
        ]
        results_zones = results_soilzones["Soil Zone Number"].tolist()
        self.assertListEqual(
            results_zones, 
            expected_zones
        )
    
    def test_Robertson1986_qc_Rf_1(self):
        """
        Unit test to test soil classification (Robertson et al 1986 (nonpiezo))
        """
        testPoint = CPT("CPT-1", 0, 0)
        test_data = [
            [0, 0.3, 3, 0],
            [0.1, 0.8, 4, 0],
            [0.2, 0.12, 4.8, 0],
            [0.3, 0.2, 15, 0],
            [0.4, 0.8, 32, 0],
            [0.5, 0.5, 30, 0],
            [0.6, 5, 350, 0],
            [0.7, 1, 30, 0],
            [0.8, 3, 90, 0],
            [0.9, 1.5, 30, 0],
            [1, 2, 40, 0],
            [1.1, 6, 180, 0],
            [1.2, 3, 30, 0],
            [1.3, 8, 40, 0],
            [1.4, 20, 160, 0],
            [1.5, 30, 150, 0],
            [1.6, 90, 5400, 0],
            [1.7, 10, 700, 0],
            [1.8, 30, 900, 0],
            [1.9, 2, 200, 0]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 1.0

        testPoint.classify_soil("Robertson et al 1986 (nonpiezo)")

        results_method, results_soilzones, results_graphdata = (
            testPoint._soil_classification_method, 
            testPoint._soil_classification, 
            testPoint._soil_classification_graph_data
        )

        self.assertEqual(results_method, "Robertson et al 1986 (nonpiezo)")

        expected_depths = [
            0, 0.1, 0.2, 0.3, 0.4, 
            0.5, 0.6, 0.7, 0.8, 0.9, 
            1, 1.1, 1.2, 1.3, 1.4, 
            1.5, 1.6, 1.7, 1.8, 1.9
        ]
        expected_Rf = [
            1, 0.5, 4, 7.5, 4, 
            6, 7, 3, 3, 2, 
            2, 3, 1, 0.5, 0.8, 
            0.5, 6, 7, 3, 10
        ]
        expected_qc = [
            0.3, 0.8, 0.12, 0.2, 0.8, 
            0.5, 5, 1, 3, 1.5, 
            2, 6, 3, 8, 20, 
            30, 90, 10, 30, 2
        ]
        expected_empty = [
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        ]
        expected_df_graphdata = pd.DataFrame(
            {
                "X_graph_1": expected_Rf,
                "Y_graph_1": expected_qc,
                "X_graph_2": expected_empty,
                "Y_graph_2": expected_empty
            },
            index=expected_depths,
            dtype=np.float32
        )
        self.assertEqualsDataframe(results_graphdata, expected_df_graphdata)

        expected_zones = [
            '1', '1', '2', '2', '3', 
            '3', '3', '4', '5', '5', 
            '6', '6', '7', '8', '9', 
            '10', '11', '11', '12', pd.NA
        ]
        results_zones = results_soilzones["Soil Zone Number"].tolist()
        self.assertListEqual(
            results_zones, 
            expected_zones
        )
    
    def test_Robertson1986_qc_Rf_2(self):
        """
        Unit test to test soil classification (Robertson et al 1986 (nonpiezo))
        (nonpiezo case)
        """
        testPoint = CPT("CPT-1", 0, 0)
        test_data = [
            [0, 0.3, 3],
            [0.1, 0.8, 4],
            [0.2, 0.12, 4.8],
            [0.3, 0.2, 15],
            [0.4, 0.8, 32],
            [0.5, 0.5, 30],
            [0.6, 5, 350],
            [0.7, 1, 30],
            [0.8, 3, 90],
            [0.9, 1.5, 30],
            [1, 2, 40],
            [1.1, 6, 180],
            [1.2, 3, 30],
            [1.3, 8, 40],
            [1.4, 20, 160],
            [1.5, 30, 150],
            [1.6, 90, 5400],
            [1.7, 10, 700],
            [1.8, 30, 900],
            [1.9, 2, 200]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 1.0

        testPoint.classify_soil("Robertson et al 1986 (nonpiezo)")

        results_method, results_soilzones, results_graphdata = (
            testPoint._soil_classification_method, 
            testPoint._soil_classification, 
            testPoint._soil_classification_graph_data
        )

        self.assertEqual(results_method, "Robertson et al 1986 (nonpiezo)")

        expected_depths = [
            0, 0.1, 0.2, 0.3, 0.4, 
            0.5, 0.6, 0.7, 0.8, 0.9, 
            1, 1.1, 1.2, 1.3, 1.4, 
            1.5, 1.6, 1.7, 1.8, 1.9
        ]
        expected_Rf = [
            1, 0.5, 4, 7.5, 4, 
            6, 7, 3, 3, 2, 
            2, 3, 1, 0.5, 0.8, 
            0.5, 6, 7, 3, 10
        ]
        expected_qc = [
            0.3, 0.8, 0.12, 0.2, 0.8, 
            0.5, 5, 1, 3, 1.5, 
            2, 6, 3, 8, 20, 
            30, 90, 10, 30, 2
        ]
        expected_empty = [
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
            np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        ]
        expected_df_graphdata = pd.DataFrame(
            {
                "X_graph_1": expected_Rf,
                "Y_graph_1": expected_qc,
                "X_graph_2": expected_empty,
                "Y_graph_2": expected_empty
            },
            index=expected_depths,
            dtype=np.float32
        )
        self.assertEqualsDataframe(results_graphdata, expected_df_graphdata)

        expected_zones = [
            '1', '1', '2', '2', '3', 
            '3', '3', '4', '5', '5', 
            '6', '6', '7', '8', '9', 
            '10', '11', '11', '12', pd.NA
        ]
        results_zones = results_soilzones["Soil Zone Number"].tolist()
        self.assertListEqual(
            results_zones, 
            expected_zones
        )
    
    def test_Robertson1986_1(self):
        """
        Unit test to test soil classification (Robertson et al 1986)
        """
        testPoint = CPT("CPT-1", 0, 0)
        test_data = [
            [0, 0.13, 5, 65],
            [0.5, 0.3, 2, 384.6],
            [1, 2.8, 0.1, 2237.2],
            [1.5, 0.3, 0.9, 97.8],
            [2, 0.4, 2, 27.36],
            [2.5, 0.7, 4, 5.2],
            [3, 1.3, 3, 30],
            [3.5, 2.3, 1, 57.44],
            [4, 4, 4, 40],
            [4.5, 15, 7, 343.56],
            [5, 81, 3, 1668.4],
            [5.5, 0.088, 6.16, 55],
            [6, 10, 50, -138.08],
            [6.5, 30, 150, -532.92],
            [7, 80, 4800, -1527.76],
            [7.5, 30, 1200, -522.6],
            [8, 30, 600, -517.44],
            [8.5, 10, 600, 7976.2],
            [9, 2, 30, 2502.8],
            [9.5, 10, 200, 3049.4],
            [10, 0.2, 20, 96]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 1.0
        testPoint.unit_weight = 16
        testPoint.gwl = 0

        testPoint.classify_soil("Robertson et al 1986")

        results_method, results_soilzones, results_graphdata = (
            testPoint._soil_classification_method, 
            testPoint._soil_classification, 
            testPoint._soil_classification_graph_data
        )

        self.assertEqual(results_method, "Robertson et al 1986")

        expected_depths = [
            0, 0.5, 1, 1.5, 2, 
            2.5, 3, 3.5, 4, 4.5, 
            5, 5.5, 6, 6.5, 7, 
            7.5, 8, 8.5, 9, 9.5, 
            10
        ]
        expected_qt = [
            0.13, 0.3, 2.8, 0.3, 0.4, 
            0.7, 1.3, 2.3, 4, 15, 
            81, 0.088, 10, 30, 80, 
            30, 30, 10, 2, 10, 
            0.2
        ]
        expected_Bq = [
            0.5, 1.3, 0.8, 0.3, 0.02, 
            -0.03, 0, 0.01, 0, 0.02, 
            0.02, pd.NA, -0.02, -0.02, -0.02, 
            -0.02, -0.02, 0.8, 1.3, 0.3, 
            -0.1
        ]
        expected_Rf = [
            3.8461538461538463, 0.6666666666666667, 0.0035714285714285718, 0.3, 0.5, 
            0.5714285714285714, 0.23076923076923078, 0.043478260869565216, 0.1, 0.04666666666666667, 
            0.003703703703703704, 7.000000000000001, 0.5, 0.5, 6, 
            4, 2, 6, 1.5, 2, 
            10
        ]
        expected_df_graphdata = pd.DataFrame(
            {
                "X_graph_1": expected_Bq,
                "Y_graph_1": expected_qt,
                "X_graph_2": expected_Rf,
                "Y_graph_2": expected_qt
            },
            index=expected_depths,
            #dtype=np.float32
        )
        self.assertEqualsDataframe(results_graphdata, expected_df_graphdata)

        expected_zones = [
            '2', '1', '3', '3', '4', 
            '5', '6', '7', '8', '9', 
            '10', pd.NA, '9', '10', '11', 
            '12', '9,10,11,12', '11', '6', '7', 
            pd.NA
        ]
        expected_zones = pd.Series(
            expected_zones, 
            index=expected_depths,
            name='Soil Zone Number',
            dtype='string'
        )
        results_zones = results_soilzones["Soil Zone Number"]
        self.assertEqualsSeries(
            results_zones, 
            expected_zones
        )
    
    def test_Robertson1990_1(self):
        """
        Unit test to test soil classification (Robertson 1990)
        """
        testPoint = CPT("CPT-1", 0, 0)
        test_data = [
            [1, 0.032, 0.032, 0],
            [2, 0.32, 0.3456, 0],
            [3, 0.144, 0.96, 0],
            [4, 0.1472, 3.328, 0],
            [5, 0.24, 12.8, 0],
            [6, 0.384, 5.76, 0],
            [7, 1.232, 89.6, 0],
            [8, 1.408, 12.8, 0],
            [9, 3.024, 57.6, 0],
            [10, 2.56, 4.8, 0],
            [11, 5.456, 52.8, 0],
            [12, 19.392, 38.4, 0],
            [13, 166.608, 1664, 0],
            [14, 201.824, 403.2, 0],
            [15, 216.24, 6480, 0],
            [16, 89.856, 1792, 0],
            [17, 245.072, 22032, 0],
            [18, 17.568, 1382.4, 0],
            [19, 1.216, 0.456, 0],
            [20, 3.52, 1.6, 0],
            [21, 7.056, 3.36, 0],
            [22, 35.552, 17.6, 0],
            [23, 294.768, 147.2, 0],
            [24, 0.8448, 0.2304, 0]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 1.0
        testPoint.unit_weight = 16
        testPoint.gwl = 100

        testPoint.classify_soil("Robertson 1990")

        results_method, results_soilzones, results_graphdata = (
            testPoint._soil_classification_method, 
            testPoint._soil_classification, 
            testPoint._soil_classification_graph_data
        )

        self.assertEqual(results_method, "Robertson 1990")

        expected_depths = [
            1, 2, 3, 4, 5, 
            6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 
            16, 17, 18, 19, 20, 
            21, 22, 23, 24
        ]
        expected_Fr = [
            0.2, 0.12, 1, 4, 8, 
            2, 8, 1, 2, 0.2, 
            1, 0.2, 1, 0.2, 3, 
            2, 9, 8, 0.05, 0.05, 
            0.05, 0.05, 0.05, 0.05
        ]
        expected_Qt = [
            1, 9, 2, 1.3, 2, 
            3, 10, 10, 20, 15, 
            30, 100, 800, 900, 900, 
            350, 900, 60, 3, 10, 
            20, 100, 800, 1.2
        ]
        expected_Bq = [
            0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 
            0, 0, 0, 0
        ]
        expected_df_graphdata = pd.DataFrame(
            {
                "X_graph_1": expected_Fr,
                "Y_graph_1": expected_Qt,
                "X_graph_2": expected_Bq,
                "Y_graph_2": expected_Qt
            },
            index=expected_depths,
            dtype=np.float32
        )
        self.assertEqualsDataframe(results_graphdata, expected_df_graphdata)

        expected_zones = [
            '1', '1', '1', '2', '2', 
            '3', '3', '4', '4', '5', 
            '5', '6', '6', '7', '8', 
            '8', '9', '9', '3', '4', 
            '5', '6', '7', pd.NA
        ]
        results_zones = results_soilzones["Soil Zone Number"].tolist()
        self.assertListEqual(
            results_zones, 
            expected_zones
        )
    


if __name__ == "__main__":
    unittest.main(exit=False)