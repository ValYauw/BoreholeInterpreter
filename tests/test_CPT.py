import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import unittest
from unittest.mock import patch
import warnings

from classes import CPT

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np

class TestCPT(unittest.TestCase):

    def ignore_warnings(test_func):
        def do_test(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_func(self, *args, **kwargs)
        return do_test

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

    def test_cpt(self):
        """
        Unit test to initialize the creation of a simple CPT.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145, 56.956)
        expected_str = """CPT\nPoint ID :\tCPT-1\nX:\t\t249,730.6\nY:\t\t9,231,020.1\nElevation:\t57.0"""
        self.assertEqual(expected_str, str(testPoint))
    
    def test_cpt_uninitialized_properties(self):
        """
        Unit test to check the behaviour when properties are uninitialized
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.elevation
        with self.assertRaises(AttributeError):
            testPoint.raw_data
        with self.assertRaises(AttributeError):
            testPoint.qc
        with self.assertRaises(AttributeError):
            testPoint.fs
        with self.assertRaises(AttributeError):
            testPoint.u2
        with self.assertRaises(AttributeError):
            testPoint.area_ratio
        with self.assertRaises(AttributeError):
            testPoint.gwl
        with self.assertRaises(AttributeError):
            testPoint.static_gwl
        with self.assertRaises(AttributeError):
            testPoint.elevated_gwl
        with self.assertRaises(AttributeError):
            testPoint.unit_weight
    
    def test_input_raw_data_1(self):
        """
        Unit test to check the behaviour when inputting raw data for CPTu probe data, 
        i.e. drilling data for qc, fs and u2 are collected per depth.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data

        results_df = testPoint.raw_data
        results_qc = testPoint.qc
        results_fs = testPoint.fs
        results_u2 = testPoint.u2

        expected_df = pd.DataFrame(
            data={
                "qc": [1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
                "fs": [35.595, 47.25, 47.04, 36.645, 27.405, 16.065, 11.025, 10.08, 12.81],
                "u2": [-4.688, 18.751, -49.898, -45.387, -46.27, -42.837, -44.603, -35.09, -32.148]
            },
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        )
        expected_qc = expected_df.loc[:, "qc"]
        expected_fs = expected_df.loc[:, "fs"]
        expected_u2 = expected_df.loc[:, "u2"]

        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqualsDataframe(results_df, expected_df)
        self.assertIsInstance(results_qc, pd.Series)
        self.assertIsInstance(results_fs, pd.Series)
        self.assertIsInstance(results_u2, pd.Series)
        self.assertEqualsSeries(results_qc, expected_qc)
        self.assertEqualsSeries(results_fs, expected_fs)
        self.assertEqualsSeries(results_u2, expected_u2)
        self.assertEqual(list(results_df.dtypes), [np.float64, np.float64, np.float64])
    
    def test_input_raw_data_2(self):
        """
        Unit test to check the behaviour when inputting raw data for CPT probe data without piezocone, 
        i.e. drilling data for qc, fs are collected per depth, but not u2.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595],
            [0.1, 1.805, 47.25],
            [0.15, 1.486, 47.04],
            [0.2, 1.266, 36.645],
            [0.25, 1.103, 27.405],
            [0.3, 1.003, 16.065],
            [0.35, 1.042, 11.025],
            [0.4, 0.994, 10.08],
            [0.45, 1.115, 12.81]
        ]
        testPoint.raw_data = test_data

        results_df = testPoint.raw_data
        results_qc = testPoint.qc
        results_fs = testPoint.fs

        expected_df = pd.DataFrame(
            data={
                "qc": [1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
                "fs": [35.595, 47.25, 47.04, 36.645, 27.405, 16.065, 11.025, 10.08, 12.81],
                "u2": [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
            },
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        )
        expected_qc = expected_df.loc[:, "qc"]
        expected_fs = expected_df.loc[:, "fs"]
        
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqualsDataframe(results_df, expected_df)
        self.assertIsInstance(results_qc, pd.Series)
        self.assertIsInstance(results_fs, pd.Series)
        self.assertEqualsSeries(results_qc, expected_qc)
        self.assertEqualsSeries(results_fs, expected_fs)
        with self.assertRaises(AttributeError):
            print(testPoint.u2)
        self.assertEqual(list(results_df.dtypes), [np.float64, np.float64, np.float64])
    
    def test_misaligned_data(self):
        qc = [
            [0.05, 1.44],
            [0.1, 1.805],
            [0.15, 1.486],
            [0.2, 1.266],
            [0.25, 1.103],
            [0.3, 1.003],
            [0.35, 1.042],
            [0.4, 0.994],
            [0.45, 1.115]
        ]
        fs = [
            [0.1, 47.25],
            [0.2, 36.645],
            [0.3, 16.065],
            [0.4, 10.08]
        ]
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        testPoint.qc = qc
        with self.assertRaises(ValueError):
            testPoint.fs = fs

    def test_initialize_stress_1(self):
        """
        Unit test to test the calculation of total & effective stress without first initializing the test data depth points first.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        testPoint.unit_weight = 16
        
        result_unit_weight = testPoint.unit_weight
        result_total_stress = testPoint.total_stress

        expected_unit_weight = np.array([[0, 16], [9999, 16]])
        expected_total_stress = pd.Series(
            index=[0, 9999],
            data=[0, 159984],
            name='sv0'
        )

        self.assertIsInstance(result_unit_weight, np.ndarray)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertTrue(np.array_equal(result_unit_weight, expected_unit_weight))
        self.assertEqualsSeries(result_total_stress, expected_total_stress)

        testPoint.elevated_gwl = 0
        result_effective_u0 = testPoint.elevated_u0
        result_effective_stress = testPoint.effective_stress
        expected_effective_u0 = pd.Series(
            index=[0, 9999],
            data=[0, 99990],
            name='u0 (elevated)'
        )
        expected_effective_stress = pd.Series(
            index=[0, 9999],
            data=[0, 59994],
            name='sv0\''
        )
        self.assertIsInstance(result_effective_u0, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertEqualsSeries(result_effective_u0, expected_effective_u0)
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress)
    
    def test_initialize_stress_2(self):
        """
        Unit test to test the calculation of total & effective stress after first initializing the test data depth points first.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595],
            [0.1, 1.805, 47.25],
            [0.15, 1.486, 47.04],
            [0.2, 1.266, 36.645],
            [0.25, 1.103, 27.405],
            [0.3, 1.003, 16.065],
            [0.35, 1.042, 11.025],
            [0.4, 0.994, 10.08],
            [0.45, 1.115, 12.81]
        ]
        testPoint.raw_data = test_data

        testPoint.unit_weight = 16
        
        result_unit_weight = testPoint.unit_weight
        result_total_stress = testPoint.total_stress

        expected_unit_weight = np.array([[0, 16], [9999, 16]])
        expected_total_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.8, 1.6, 2.4, 3.2, 4, 4.8, 5.6, 6.4, 7.2],
            name='sv0'
        )

        self.assertIsInstance(result_unit_weight, np.ndarray)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertTrue(np.array_equal(result_unit_weight, expected_unit_weight))
        self.assertEqualsSeries(result_total_stress, expected_total_stress)

        testPoint.elevated_gwl = 0.25
        result_effective_u0 = testPoint.elevated_u0
        result_effective_stress = testPoint.effective_stress
        expected_effective_u0 = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0],
            name='u0 (elevated)'
        )
        expected_effective_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.8, 1.6, 2.4, 3.2, 4, 4.3, 4.6, 4.9, 5.2],
            name='sv0\''
        )
        self.assertIsInstance(result_effective_u0, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertEqualsSeries(result_effective_u0, expected_effective_u0)
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress)
    
    def test_initialize_stress_3(self):
        """
        Unit test to test the calculation of unit weight after first initializing the test data depth points first. Test to check floating point imprecision.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595],
            [0.1, 1.805, 47.25],
            [0.15, 1.486, 47.04],
            [0.2, 1.266, 36.645],
            [0.25, 1.103, 27.405],
            [0.3, 1.003, 16.065],
            [0.35, 1.042, 11.025],
            [0.4, 0.994, 10.08],
            [0.45, 1.115, 12.81]
        ]
        testPoint.raw_data = test_data

        testPoint.unit_weight = 14.55
        
        result_unit_weight = testPoint.unit_weight
        result_total_stress = testPoint.total_stress

        expected_unit_weight = np.array([[0, 14.55], [9999, 14.55]])
        expected_total_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.7275, 1.455, 2.1825, 2.91, 3.6375, 4.365, 5.0925, 5.82, 6.5475],
            name='sv0'
        )

        self.assertIsInstance(result_unit_weight, np.ndarray)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertTrue(np.array_equal(result_unit_weight, expected_unit_weight))
        self.assertEqualsSeries(result_total_stress, expected_total_stress)

        testPoint.elevated_gwl = 0.25
        result_effective_u0 = testPoint.elevated_u0
        result_effective_stress = testPoint.effective_stress
        expected_effective_u0 = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0],
            name='u0 (elevated)'
        )
        expected_effective_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.7275, 1.455, 2.1825, 2.91, 3.6375, 3.865, 4.0925, 4.32, 4.5475],
            name='sv0\''
        )
        self.assertIsInstance(result_effective_u0, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertEqualsSeries(result_effective_u0, expected_effective_u0)
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress)
    
    def test_initialize_stress_4(self):
        """
        Unit test to test the calculation of total and effective stress without first initializing the test data depth points first.
        Unit weight is set as a variable dataset.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        testPoint.unit_weight = [[0,14], [2,14], [20,15]]
        
        result_unit_weight = testPoint.unit_weight
        result_total_stress = testPoint.total_stress

        expected_unit_weight = np.array([[0, 14], [2, 14], [20, 15]])
        expected_total_stress = pd.Series(
            index=[0.0, 2.0, 20.0],
            data=[0.0, 28.0, 298.0],
            name='sv0'
        )

        self.assertIsInstance(result_unit_weight, np.ndarray)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertTrue(np.array_equal(result_unit_weight, expected_unit_weight))
        self.assertEqualsSeries(result_total_stress, expected_total_stress)

        testPoint.elevated_gwl = 1
        result_effective_u0 = testPoint.elevated_u0
        result_effective_stress = testPoint.effective_stress
        expected_effective_u0 = pd.Series(
            index=[0.0, 2.0, 20.0],
            data=[0.0, 10.0, 190.0],
            name='u0 (elevated)'
        )
        expected_effective_stress = pd.Series(
            index=[0.0, 2.0, 20.0],
            data=[0.0, 18.0, 108.0],
            name='sv0\''
        )
        self.assertIsInstance(result_effective_u0, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertEqualsSeries(result_effective_u0, expected_effective_u0)
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress)
    
    def test_initialize_stress_5(self):
        """
        Unit test to test the calculation of total and effective stress after first initializing the test data depth points.
        Unit weight is set as a variable dataset.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595],
            [0.1, 1.805, 47.25],
            [0.15, 1.486, 47.04],
            [0.2, 1.266, 36.645],
            [0.25, 1.103, 27.405],
            [0.3, 1.003, 16.065],
            [0.35, 1.042, 11.025],
            [0.4, 0.994, 10.08],
            [0.45, 1.115, 12.81]
        ]
        testPoint.raw_data = test_data

        testPoint.unit_weight = [[0, 14], [0.2, 14], [0.2, 15], [0.5, 15]]
        
        result_unit_weight = testPoint.unit_weight
        result_total_stress = testPoint.total_stress

        expected_unit_weight = np.array([[0, 14], [0.2, 14], [0.2, 15], [0.5, 15]])
        expected_total_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.7, 1.4, 2.1, 2.8, 3.55, 4.3, 5.05, 5.8, 6.55],
            name='sv0'
        )

        self.assertIsInstance(result_unit_weight, np.ndarray)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertTrue(np.array_equal(result_unit_weight, expected_unit_weight))
        self.assertEqualsSeries(result_total_stress, expected_total_stress)

        testPoint.elevated_gwl = 0.2
        result_effective_u0 = testPoint.elevated_u0
        result_effective_stress = testPoint.effective_stress
        expected_effective_u0 = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            name='u0 (elevated)'
        )
        expected_effective_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.7, 1.4, 2.1, 2.8, 3.05, 3.3, 3.55, 3.8, 4.05],
            name='sv0\''
        )
        self.assertIsInstance(result_effective_u0, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertEqualsSeries(result_effective_u0, expected_effective_u0)
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress)
    
    def test_initialize_stress_6(self):
        """
        Unit test to test the behaviour of the class in calculating the total stress when setting the unit weight before initializing depth points, then after initializing depth-points (without re-setting the unit weight)
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        testPoint.unit_weight = 16
        testPoint.elevated_gwl = 0

        # Calculate the total stress, before having set the depth points of the test
        result_unit_weight = testPoint.unit_weight
        result_total_stress = testPoint.total_stress
        result_effective_stress = testPoint.effective_stress
        expected_unit_weight = np.array([[0, 16], [9999, 16]])
        expected_total_stress = pd.Series(
            index=[0, 9999],
            data=[0, 159984],
            name='sv0'
        )
        expected_effective_stress = pd.Series(
            index=[0, 9999],
            data=[0, 59994],
            name='sv0\''
        )
        self.assertIsInstance(result_unit_weight, np.ndarray)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertTrue(np.array_equal(result_unit_weight, expected_unit_weight), 
            "Expected different unit weight output (pre-initialization of depth points)"
        )
        self.assertEqualsSeries(result_total_stress, expected_total_stress, 
            "Expected different total stress output (pre-initialization of depth points)"
        )
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress, 
            "Expected different effective stress output (pre-initialization of depth points)"
        )

        # Set the depth points
        test_data = [
            [0.05, 1.44, 35.595],
            [0.1, 1.805, 47.25],
            [0.15, 1.486, 47.04],
            [0.2, 1.266, 36.645],
            [0.25, 1.103, 27.405],
            [0.3, 1.003, 16.065],
            [0.35, 1.042, 11.025],
            [0.4, 0.994, 10.08],
            [0.45, 1.115, 12.81]
        ]
        testPoint.raw_data = test_data

        # Re-calculate the total stress
        result_unit_weight = testPoint.unit_weight
        result_total_stress = testPoint.total_stress
        result_effective_stress = testPoint.effective_stress
        expected_unit_weight = np.array([[0, 16], [9999, 16]])
        expected_total_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.8, 1.6, 2.4, 3.2, 4, 4.8, 5.6, 6.4, 7.2],
            name='sv0'
        )
        expected_effective_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7],
            name='sv0\''
        )
        self.assertIsInstance(result_unit_weight, np.ndarray)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertTrue(np.array_equal(result_unit_weight, expected_unit_weight), 
            "Expected different unit weight output (post-initialization of depth points)"
        )
        self.assertEqualsSeries(result_total_stress, expected_total_stress, 
            "Expected different total stress output (post-initialization of depth points)"
        )
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress, 
            "Expected different effective stress output (post-initialization of depth points)"
        )
    
    def test_effective_stresses_1(self):
        """
        Unit test to calculate the effective stress and porewater pressures of the CPT test point,
        when both static and elevated groundwater levels are defined.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        testPoint.unit_weight = 16
        testPoint.static_gwl = 5
        testPoint.elevated_gwl = 0

        result_static_u0 = testPoint.u0
        result_elevated_u0 = testPoint.elevated_u0
        result_total_stress = testPoint.total_stress
        result_effective_stress = testPoint.effective_stress

        expected_static_u0 = pd.Series(
            index=[0, 9999],
            data=[0, 99940],
            name='u0'
        )
        expected_elevated_u0 = pd.Series(
            index=[0, 9999],
            data=[0, 99990],
            name='u0 (elevated)'
        )
        expected_total_stress = pd.Series(
            index=[0, 9999],
            data=[0, 159984],
            name='sv0'
        )
        expected_effective_stress = pd.Series(
            index=[0, 9999],
            data=[0, 59994],
            name='sv0\''
        )

        self.assertIsInstance(result_static_u0, pd.Series)
        self.assertIsInstance(result_elevated_u0, pd.Series)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertEqualsSeries(result_static_u0, expected_static_u0)
        self.assertEqualsSeries(result_elevated_u0, expected_elevated_u0)
        self.assertEqualsSeries(result_total_stress, expected_total_stress)
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress)

    def test_effective_stresses_2(self):
        """
        Unit test to calculate the effective stress and porewater pressures of the CPT test point,
        when only static groundwater level is defined.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        testPoint.unit_weight = 16
        testPoint.static_gwl = 5

        result_static_u0 = testPoint.u0
        with self.assertRaises(AttributeError):
            result_elevated_u0 = testPoint.elevated_u0
        result_total_stress = testPoint.total_stress
        result_effective_stress = testPoint.effective_stress

        expected_static_u0 = pd.Series(
            index=[0, 9999],
            data=[0, 99940],
            name='u0'
        )
        expected_total_stress = pd.Series(
            index=[0, 9999],
            data=[0, 159984],
            name='sv0'
        )
        expected_effective_stress = pd.Series(
            index=[0, 9999],
            data=[0, 60044],
            name='sv0\''
        )

        self.assertIsInstance(result_static_u0, pd.Series)
        self.assertIsInstance(result_total_stress, pd.Series)
        self.assertIsInstance(result_effective_stress, pd.Series)
        self.assertEqualsSeries(result_static_u0, expected_static_u0)
        self.assertEqualsSeries(result_total_stress, expected_total_stress)
        self.assertEqualsSeries(result_effective_stress, expected_effective_stress)

    def test_qt_1(self):
        """
        Unit test to calculate the corrected cone resistance, qt

        Simple case: CPTu data with complete qc, fs, u2 data and an area ratio
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.qt
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85

        result_series = testPoint.qt
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.4392968, 1.80781265, 1.4785153, 1.25919195, 1.0960595, 0.99657445, 1.03530955, 0.9887365, 1.1101778],
            dtype=np.float64,
            name='qt'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    def test_qt_2(self):
        """
        Unit test to calculate the corrected cone resistance, qt

        Case 2: CPT data with only qc and fs, and an area ratio
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.qt
        test_data = [
            [0.05, 1.44, 35.595],
            [0.1, 1.805, 47.25],
            [0.15, 1.486, 47.04],
            [0.2, 1.266, 36.645],
            [0.25, 1.103, 27.405],
            [0.3, 1.003, 16.065],
            [0.35, 1.042, 11.025],
            [0.4, 0.994, 10.08],
            [0.45, 1.115, 12.81]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85

        result_series = testPoint.qt
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
            dtype=np.float64,
            name='qt'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    #@patch('warnings.warn')
    @ignore_warnings
    def test_qt_3(self):
        """
        Unit test to calculate the corrected cone resistance, qt

        Case 3: CPTu data with complete qc, fs, u2 data but no area ratio
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.qt
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data

        result_series = testPoint.qt
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
            dtype=np.float64,
            name='qc'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    @ignore_warnings
    def test_qt_4(self):
        """
        Unit test to calculate the corrected cone resistance, qt

        Case 3: CPTu data with only qc and fs data and no area ratio
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.qt
        test_data = [
            [0.05, 1.44, 35.595],
            [0.1, 1.805, 47.25],
            [0.15, 1.486, 47.04],
            [0.2, 1.266, 36.645],
            [0.25, 1.103, 27.405],
            [0.3, 1.003, 16.065],
            [0.35, 1.042, 11.025],
            [0.4, 0.994, 10.08],
            [0.45, 1.115, 12.81]
        ]
        testPoint.raw_data = test_data

        result_series = testPoint.qt
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
            dtype=np.float64,
            name='qc'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    def test_Rf(self):
        """
        Unit test to calculate the friction ratio, Rf
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.Rf
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85

        result_series = testPoint.Rf
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[2.47308268871299, 2.61365579005103, 3.18157005206507, 2.91019967209924, 2.50032046617907, 1.61202206217508, 1.06489889907806, 1.01948294616412, 1.15386922707336],
            dtype=np.float64,
            name='Rf'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
        
    def test_Qt(self):
        """
        Unit test to calculate the normalized cone penetration, Qt
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.Qt
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0

        result_series = testPoint.Qt
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[4794.98933333333, 3010.35441666667, 1640.12811111111, 1046.65995833333, 728.039666666667, 550.985805555556, 490.337880952381, 409.306875, 408.510296296296],
            dtype=np.float64,
            name='Normalized Cone Penetration, Qt'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    def test_Fr(self):
        """
        Unit test to calculate the normalized friction ratio, Fr
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.Fr
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16

        result_series = testPoint.Fr
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[2.4744580592741, 2.61597104859165, 3.18674293261509, 2.91761424107854, 2.50947865020175, 1.61982394283297, 1.07069027377672, 1.02612495819915, 1.16140143527821],
            dtype=np.float64,
            name='Normalized Friction Ratio, Fr'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    def test_Bq(self):
        """
        Unit test to calculate the porewater pressure ratio, Bq
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.Bq
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0

        result_series = testPoint.Bq
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[-0.00360654260753309, 0.00982774647270907, -0.0348197732250319, -0.0377287449971316, -0.0446587388324537, -0.0462171615733799, -0.0467151149564457, -0.0397928815634968, -0.0332264167057578],
            dtype=np.float64,
            name='Bq'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    def test_Ic(self):
        """
        Unit test to calculate the soil behaviour index, Ic
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        with self.assertRaises(AttributeError):
            testPoint.Ic
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0

        result_series = testPoint.Ic
        expected_series = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.62719066349404, 1.63765560715623, 1.74212868108295, 1.74413127718268, 1.72989196862372, 1.60456054177913, 1.4728499946456, 1.50064449780308, 1.54554583765418],
            dtype=np.float64,
            name='Ic'
        )

        self.assertIsInstance(result_series, pd.Series)
        self.assertEqualsSeries(result_series, expected_series)
    
    def test_del_raw_data(self):
        """
        Unit test to test the behaviour when deleting the raw_data property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = [
            'qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic',
            'total_stress', 'effective_stress', 'static_u0'
        ]

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")

        del testPoint.raw_data

        self.assertFalse(hasattr(testPoint, "_qc"))
        self.assertFalse(hasattr(testPoint, "_fs"))
        self.assertFalse(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")

        expected_u0 = pd.Series(
            index=[0, 9999],
            data=[0, 99990],
            name='u0'
        )
        self.assertEqualsSeries(
            expected_u0, testPoint.u0
        )

        expected_total_stress = pd.Series(
            index=[0, 9999],
            data=[0, 159984],
            name='sv0'
        )
        self.assertEqualsSeries(
            expected_total_stress, testPoint.total_stress
        )
        expected_effective_stress = pd.Series(
            index=[0, 9999],
            data=[0, 59994],
            name='sv0\''
        )
        self.assertEqualsSeries(
            expected_effective_stress, testPoint.effective_stress
        )
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")
    
    def test_set_raw_data(self):
        """
        Unit test to test the behaviour when setting the raw_data property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data_pre = [
            [0.05, 1.44, 35.595, -4.688]
        ]
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data_pre
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = [
            'qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic',
            'total_stress', 'effective_stress', 'static_u0'
        ]

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")

        testPoint.raw_data = test_data

        self.assertTrue(hasattr(testPoint, "_qc"))
        self.assertTrue(hasattr(testPoint, "_fs"))
        self.assertTrue(hasattr(testPoint, "_u2"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_del_qc(self):
        """
        Unit test to test the behaviour when deleting the qc property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = ['qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic']

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")

        del testPoint.qc
        self.assertFalse(hasattr(testPoint, "_qc"))
        self.assertTrue(hasattr(testPoint, "_fs"))
        self.assertTrue(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")
    
    def test_set_qc(self):
        """
        Unit test to test the behaviour when setting the qc property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        test_qc = [1.000, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = ['qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic']

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")

        depth_index = testPoint.qc
        depth_index = depth_index.index
        testPoint.qc = np.stack((depth_index, test_qc), axis=1)

        self.assertTrue(hasattr(testPoint, "_qc"))
        self.assertTrue(hasattr(testPoint, "_fs"))
        self.assertTrue(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        
        self.assertEqualsSeries(
            pd.Series(
                data=test_qc, index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                dtype=np.float64,
                name='qc'
            ),
            testPoint.qc
        )
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_del_fs(self):
        """
        Unit test to test the behaviour when deleting the fs property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = ['Rf', 'Fr', 'Ic']
        check_nondependencies = ['qt', 'Qt', 'Bq']
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.fs
        self.assertTrue(hasattr(testPoint, "_qc"))
        self.assertFalse(hasattr(testPoint, "_fs"))
        self.assertTrue(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                data_nondependencies[dpd],
                getattr(testPoint, dpd)
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")
    
    def test_set_fs(self):
        """
        Unit test to test the behaviour when setting the fs property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        test_fs = [30.123, 47.25, 47.04, 36.645, 27.405, 16.065, 11.025, 10.08, 12.81]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = ['Rf', 'Fr', 'Ic']
        check_nondependencies = ['qt', 'Qt', 'Bq']
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.fs = np.stack((testPoint.fs.index, test_fs), axis=1)

        self.assertTrue(hasattr(testPoint, "_qc"))
        self.assertTrue(hasattr(testPoint, "_fs"))
        self.assertTrue(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                data_nondependencies[dpd],
                getattr(testPoint, dpd)
            )
        
        self.assertEqualsSeries(
            pd.Series(
                data=test_fs, index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                dtype=np.float64,
                name='fs'
            ),
            testPoint.fs
        )
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_del_u2_1(self):
        """
        Unit test to test the behaviour when deleting the u2 property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = ['qt', 'Bq', 'Rf', 'Qt', 'Fr', 'Ic']
        check_nondependencies = []
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.u2
        self.assertTrue(hasattr(testPoint, "_qc"))
        self.assertTrue(hasattr(testPoint, "_fs"))
        self.assertFalse(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                data_nondependencies[dpd],
                getattr(testPoint, dpd)
            )
        
        expected_qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
            dtype=np.float64,
            name='qt'
        )
        self.assertEqualsSeries(expected_qt, testPoint.qt)

        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')

        testPoint.classify_soil()
        expected_graph_1_data = pd.DataFrame(
            data={
                "X_graph_1": [35.595, 47.25, 47.04, 36.645, 27.405, 16.065, 11.025, 10.08, 12.81],
                "Y_graph_1": [1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115]
            },
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            dtype=np.float32
            )
        self.assertEqualsDataframe(
            expected_graph_1_data, 
            testPoint._soil_classification_graph_data[["X_graph_1", "Y_graph_1"]]
        )
    
    def test_del_u2_2(self):
        """
        Unit test to test the behaviour when deleting the u2 property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = ['qt', 'Bq', 'Rf', 'Qt', 'Fr', 'Ic']
        check_nondependencies = []
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.u2
        self.assertTrue(hasattr(testPoint, "_qc"))
        self.assertTrue(hasattr(testPoint, "_fs"))
        self.assertFalse(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                data_nondependencies[dpd],
                getattr(testPoint, dpd)
            )
        
        expected_qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
            dtype=np.float64,
            name='qt'
        )
        self.assertEqualsSeries(expected_qt, testPoint.qt)

        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_set_u2(self):
        """
        Unit test to test the behaviour when setting the u2 property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        test_u2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = ['qt', 'Bq', 'Rf', 'Qt', 'Fr', 'Ic']
        check_nondependencies = []
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.u2 = np.stack((testPoint.u2.index, test_u2), axis=1)

        self.assertTrue(hasattr(testPoint, "_qc"))
        self.assertTrue(hasattr(testPoint, "_fs"))
        self.assertTrue(hasattr(testPoint, "_u2"))

        self.assertTrue(hasattr(testPoint, "area_ratio"))
        self.assertTrue(hasattr(testPoint, "unit_weight"))
        self.assertTrue(hasattr(testPoint, "gwl"))

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                data_nondependencies[dpd],
                getattr(testPoint, dpd)
            )
        
        self.assertEqualsSeries(
            pd.Series(
                data=test_u2, index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                dtype=np.float64,
                name='u2'
            ),
            testPoint.u2
        )
        
        expected_qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
            dtype=np.float64,
            name='qt'
        )
        self.assertEqualsSeries(expected_qt, testPoint.qt)

        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    @ignore_warnings
    def test_del_area_ratio(self):
        """
        Unit test to test the behaviour when deleting the area_ratio property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = ['qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic']
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'u0', 'total_stress', 'effective_stress'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.area_ratio

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        expected_qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.44, 1.805, 1.486, 1.266, 1.103, 1.003, 1.042, 0.994, 1.115],
            dtype=np.float64,
            name='qc'
        )
        self.assertEqualsSeries(expected_qt, testPoint.qt)

        expected_Rf = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[2.471875, 2.61772853185596, 3.16554508748318, 2.89454976303318, 2.48458748866727, 1.60169491525424, 1.05806142034549, 1.01408450704225, 1.14887892376682],
            dtype=np.float64,
            name='Rf'
        )
        self.assertEqualsSeries(expected_Rf, testPoint.Rf)

        expected_Qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[4797.33333333333, 3005.66666666667, 1648.44444444444, 1052.33333333333, 732.666666666667, 554.555555555556, 493.52380952381, 411.5, 410.296296296296],
            dtype=np.float64,
            name='Normalized Cone Penetration, Qt'
        )
        self.assertEqualsSeries(expected_Qt, testPoint.Qt)

        expected_Fr = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[2.47324902723735, 2.62005101474992, 3.1706659476948, 2.90188470066519, 2.49363057324841, 1.609396914446, 1.06377846391355, 1.02065613608748, 1.15634591081423],
            dtype=np.float64,
            name='Normalized Friction Ratio, Fr'
        )
        self.assertEqualsSeries(expected_Fr, testPoint.Fr)

        expected_Bq = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[-0.00360478043357421, 0.00984307419319064, -0.0346441089242383, -0.0375253405131454, -0.0443767060964513, -0.0459196553796834, -0.0464135468930915, -0.0395808019441069, -0.0330817837154721],
            dtype=np.float64,
            name='Bq'
        )
        self.assertEqualsSeries(expected_Bq, testPoint.Bq)

        expected_Ic = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.62700771415096, 1.63832899178285, 1.73963513582073, 1.74125791858062, 1.72634999614562, 1.60078841819055, 1.46897521837162, 1.49741366235088, 1.54291799900455],
            dtype=np.float64,
            name='Ic'
        )
        self.assertEqualsSeries(expected_Ic, testPoint.Ic)

        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
    
    def test_set_area_ratio(self):
        """
        Unit test to test the behaviour when setting the area_ratio property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = ['qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic']
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'u0', 'total_stress', 'effective_stress'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.area_ratio = 0.65

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        expected_qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.4383592, 1.81156285, 1.4685357, 1.25011455, 1.0868055, 0.98800705, 1.02638895, 0.9817185, 1.1037482],
            dtype=np.float64,
            name='qt'
        )
        self.assertEqualsSeries(expected_qt, testPoint.qt)

        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')

    def test_del_unit_weight_1(self):
        """
        Unit test to test the behaviour when deleting the unit_weight property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies = [
            'total_stress', 'effective_stress',
            'Qt', 'Fr', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'qt', 'u0', 'Rf'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.unit_weight

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")
    
    def test_del_unit_weight_2(self):
        """
        Unit test to test the behaviour when deleting the unit_weight property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'total_stress', 'effective_stress',
            'Qt', 'Fr', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'qt', 'u0', 'Rf'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.unit_weight

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")
    
    def test_set_unit_weight(self):
        """
        Unit test to test the behaviour when setting the unit_weight property.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Eslami Fellenius')

        check_dependencies_calc = [
            'total_stress', 'effective_stress'
        ]
        check_dependencies = [
            'Qt', 'Fr', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'qt', 'u0', 'Rf'
        ]
        data_nondependencies = {}

        
        for dpd in (check_dependencies_calc + check_dependencies):
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.unit_weight = 14

        for dpd in check_dependencies_calc:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Eslami Fellenius')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_del_gwl_1(self):
        """
        Unit test to test the behaviour when deleting the gwl property,
        when only one gwl property (static_gwl) is set.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'u0', 'effective_stress', 
            'Qt', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 
            'qt', 'Rf', 'Fr'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.gwl

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")
    
    def test_set_gwl_1(self):
        """
        Unit test to test the behaviour when setting the gwl property,
        when only one gwl property (static_gwl) is set.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.gwl = 0
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'u0', 'effective_stress', 
            'Qt', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 
            'qt', 'Rf', 'Fr'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.gwl = 0.2

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")
    
    def test_del_gwl_2(self):
        """
        Unit test to test the behaviour when deleting the gwl property,
        when only one gwl property (elevated_gwl) is set.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.elevated_gwl = 0
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'elevated_u0', 'effective_stress', 
            'Qt', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 
            'qt', 'Rf', 'Fr'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.elevated_gwl

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_set_gwl_2(self):
        """
        Unit test to test the behaviour when setting the gwl property,
        when only one gwl property (elevated_gwl) is set.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.elevated_gwl = 0
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'elevated_u0', 'effective_stress', 
            'Qt', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 
            'qt', 'Rf', 'Fr'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.elevated_gwl = 0.2

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_del_gwl_3(self):
        """
        Unit test to test the behaviour when deleting the gwl property,
        when two gwl properties is set, 
        and static_gwl is deleted.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.elevated_gwl = 0
        testPoint.gwl = 0.2
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'static_u0'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 'effective_stress',
            'qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic', 
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.gwl

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_set_gwl_3(self):
        """
        Unit test to test the behaviour when setting the gwl property,
        when two gwl properties is set, 
        and static_gwl is deleted.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.elevated_gwl = 0
        testPoint.gwl = 0.2
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'static_u0'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 'effective_stress',
            'qt', 'Rf', 'Qt', 'Fr', 'Bq', 'Ic', 
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.gwl = 0.1

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_del_gwl_4(self):
        """
        Unit test to test the behaviour when deleting the gwl property,
        when two gwl properties is set, 
        and elevated_gwl is deleted.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.elevated_gwl = 0
        testPoint.gwl = 0.2
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'elevated_u0',
            'effective_stress',
            'Qt', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 
            'qt', 'Rf', 'Fr'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        del testPoint.elevated_gwl

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        expected_effective_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.8, 1.6, 2.4, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7],
            dtype=np.float64,
            name='sv0\''
        )
        self.assertEqualsSeries(expected_effective_stress, testPoint.effective_stress)

        expected_Qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1798.121, 1128.88290625, 615.048041666667, 392.497484375, 312.017, 260.993276315789, 251.148670731707, 223.258295454545, 234.676127659574],
            dtype=np.float64,
            name='Normalized Cone Penetration, Qt'
        )
        self.assertEqualsSeries(expected_Qt, testPoint.Qt)

        expected_Bq = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[-0.00325895754512627, 0.0103813911390777, -0.0338035924429481, -0.0361363781033788, -0.0428273367888838, -0.0442005740317267, -0.0447728196752181, -0.0377569193448477, -0.031413143582763],
            dtype=np.float64,
            name='Bq'
        )
        self.assertEqualsSeries(expected_Bq, testPoint.Bq)

        expected_Ic = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.62776562319521, 1.68997738664059, 1.85305420485754, 1.8992052917518, 1.89084076662185, 1.77565988367423, 1.64520748541972, 1.66521063095027, 1.69119728943906],
            dtype=np.float64,
            name='Ic'
        )
        self.assertEqualsSeries(expected_Ic, testPoint.Ic)
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

    def test_set_gwl_4(self):
        """
        Unit test to test the behaviour when setting the gwl property,
        when two gwl properties is set, 
        and elevated_gwl is deleted.
        """
        testPoint = CPT("CPT-1", 249730.567, 9231020.145)
        test_data = [
            [0.05, 1.44, 35.595, -4.688],
            [0.1, 1.805, 47.25, 18.751],
            [0.15, 1.486, 47.04, -49.898],
            [0.2, 1.266, 36.645, -45.387],
            [0.25, 1.103, 27.405, -46.27],
            [0.3, 1.003, 16.065, -42.837],
            [0.35, 1.042, 11.025, -44.603],
            [0.4, 0.994, 10.08, -35.09],
            [0.45, 1.115, 12.81, -32.148]
        ]
        testPoint.raw_data = test_data
        testPoint.area_ratio = 0.85
        testPoint.unit_weight = 16
        testPoint.elevated_gwl = 0
        testPoint.gwl = 0.4
        testPoint.classify_soil('Robertson et al 1986')

        check_dependencies = [
            'elevated_u0',
            'effective_stress',
            'Qt', 'Bq', 'Ic'
        ]
        check_nondependencies = [
            '_qc', '_fs', '_u2', 
            'total_stress', 
            'qt', 'Rf', 'Fr'
        ]
        data_nondependencies = {}

        for dpd in check_dependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is not found.")
            data_nondependencies.update({dpd: getattr(testPoint, dpd).copy()})

        testPoint.elevated_gwl = 0.2

        for dpd in check_dependencies:
            self.assertFalse(hasattr(testPoint, "_" + dpd), f"_{dpd} is not deleted.")
        for dpd in check_nondependencies:
            self.assertTrue(hasattr(testPoint, dpd), f"{dpd} is deleted.")
            self.assertEqualsSeries(
                getattr(testPoint, dpd),
                data_nondependencies[dpd]
            )
        
        expected_effective_stress = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[0.8, 1.6, 2.4, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7],
            dtype=np.float64,
            name='sv0\''
        )
        self.assertEqualsSeries(expected_effective_stress, testPoint.effective_stress)

        expected_Qt = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1798.121, 1128.88290625, 615.048041666667, 392.497484375, 312.017, 260.993276315789, 251.148670731707, 223.258295454545, 234.676127659574],
            dtype=np.float64,
            name='Normalized Cone Penetration, Qt'
        )
        self.assertEqualsSeries(expected_Qt, testPoint.Qt)

        expected_Bq = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[-0.00325895754512627, 0.0103813911390777, -0.0338035924429481, -0.0361363781033788, -0.0428273367888838, -0.0442005740317267, -0.0447728196752181, -0.0377569193448477, -0.031413143582763],
            dtype=np.float64,
            name='Bq'
        )
        self.assertEqualsSeries(expected_Bq, testPoint.Bq)

        expected_Ic = pd.Series(
            index=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            data=[1.62776562319521, 1.68997738664059, 1.85305420485754, 1.8992052917518, 1.89084076662185, 1.77565988367423, 1.64520748541972, 1.66521063095027, 1.69119728943906],
            dtype=np.float64,
            name='Ic'
        )
        self.assertEqualsSeries(expected_Ic, testPoint.Ic)
        
        self.assertEqual(testPoint._soil_classification_method, 'Robertson et al 1986')
        self.assertFalse(hasattr(testPoint, "_classification"), "Classification not deleted.")

if __name__ == "__main__":
    unittest.main(exit=False)