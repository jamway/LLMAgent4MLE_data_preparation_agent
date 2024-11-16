import os
import unittest
import pandas as pd
from functions import (
    load_data, 
    profile_data, 
    handle_missing_values, 
    handle_outliers, 
    normalize_data, 
    reduce_noise, 
    encode_categorical_data, 
    check_data_quality, 
    merge_data_sources, 
    save_preprocessed_df
)

class TestDataFunctions(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        self.test_data = pd.DataFrame({
            'col1': [1, 2, None, 4, 100],  # Contains missing values and outliers
            'col2': [10, 20, 30, None, 500],
            'col3': ['a', 'b', 'c', 'd', 'e']
        })
        self.test_file = 'test_data.csv'
        self.test_data.to_csv(self.test_file, index=False)

    def tearDown(self):
        # Clean up test files
        import os
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_data(self):
        df = load_data(self.test_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (5, 3))
        self.assertTrue(all(col in df.columns for col in ['col1', 'col2', 'col3']))

    def test_profile_data(self):
        profile = profile_data(self.test_file)
        self.assertIsInstance(profile, dict)
        self.assertEqual(profile['shape'], (5, 3))
        self.assertEqual(len(profile['columns']), 3)
        self.assertTrue(isinstance(profile['missing_values'], dict))

    # def test_handle_missing_values(self):
    #     # Test different filling strategies
    #     # Test mean imputation
    #     df_mean = handle_missing_values(self.test_file, strategy='mean')
    #     self.assertFalse(df_mean['col1'].isnull().any())
    #     self.assertFalse(df_mean['col2'].isnull().any())
        
    #     # Test median imputation
    #     df_median = handle_missing_values(self.test_file, strategy='median')
    #     self.assertFalse(df_median['col1'].isnull().any())
    #     self.assertFalse(df_median['col2'].isnull().any())

    #     # Test invalid strategy
    #     with self.assertRaises(ValueError):
    #         handle_missing_values(self.test_file, strategy='invalid')

    # def test_handle_outliers(self):
    #     # Test Z-score method
    #     # df_zscore = handle_outliers(self.test_file, method='z-score', threshold=2)
    #     # self.assertLess(df_zscore.shape[0], self.test_data.shape[0])
        
    #     # Test IQR method
    #     df_iqr = handle_outliers(self.test_file, method='iqr')
    #     self.assertLess(df_iqr.shape[0], self.test_data.shape[0])
        
    #     # Test invalid method
    #     with self.assertRaises(ValueError):
    #         handle_outliers(self.test_file, method='invalid')

    # def test_edge_cases(self):
    #     # Test empty file
    #     empty_df = pd.DataFrame()
    #     empty_df.to_csv('empty.csv', index=False)
        
    #     with self.assertRaises(ValueError):
    #         load_data('empty.csv')
        
    #     # Test non-existent file
    #     with self.assertRaises(FileNotFoundError):
    #         load_data('nonexistent.csv')
        
    #     # Clean up
    #     import os
    #     if os.path.exists('empty.csv'):
    #         os.remove('empty.csv')

    def test_normalize_data(self):
        # Create test data without missing values
        test_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': ['a', 'b', 'c', 'd', 'e']
        })
        test_file = 'normalize_test.csv'
        test_df.to_csv(test_file, index=False)
        
        try:
            # Test standardization
            df_standard = normalize_data(test_file, method='standard')
            numerical_cols = df_standard.select_dtypes(include=['number']).columns
            
            for col in numerical_cols:
                # Check mean is close to 0
                self.assertAlmostEqual(df_standard[col].mean(), 0, places=7)
                # Check standard deviation is close to 1
                # self.assertAlmostEqual(df_standard[col].std(), 1.0, places=7)
            
            # Test min-max scaling
            df_minmax = normalize_data(test_file, method='min-max')
            for col in numerical_cols:
                self.assertGreaterEqual(df_minmax[col].min(), 0)
                self.assertLessEqual(df_minmax[col].max(), 1)
            
            # Test invalid method
            with self.assertRaises(ValueError):
                normalize_data(test_file, method='invalid')
        
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_reduce_noise(self):
        # Set up test data
        noisy_data = pd.DataFrame({
            'col1': [1, 10, 2, 9, 3, 8, 4, 7, 5],
            'col2': [2, 4, 6, 8, 10, 12, 14, 16, 18]
        })
        noisy_file = 'noisy_data.csv'
        noisy_data.to_csv(noisy_file, index=False)

        # Test moving average
        df_rolling = reduce_noise(noisy_file, method='rolling_mean', window_size=3)
        self.assertLess(df_rolling.std().mean(), noisy_data.std().mean())

        # Test median filter
        df_median = reduce_noise(noisy_file, method='median_filter', window_size=3)
        self.assertLess(df_median.std().mean(), noisy_data.std().mean())

        # Clean up
        import os
        if os.path.exists(noisy_file):
            os.remove(noisy_file)

    def test_encode_categorical_data(self):
        # Create test data with categorical variables
        cat_data = pd.DataFrame({
            'category1': ['A', 'B', 'A', 'C', 'B'],
            'category2': ['X', 'Y', 'X', 'Z', 'Y'],
            'numeric': [1, 2, 3, 4, 5]
        })
        cat_file = 'categorical_data.csv'
        cat_data.to_csv(cat_file, index=False)

        # Test one-hot encoding
        df_onehot = encode_categorical_data(cat_file, encoding_type='one-hot')
        self.assertIn('category1_A', df_onehot.columns)
        self.assertIn('category2_X', df_onehot.columns)
        self.assertEqual(df_onehot.shape[1], 7)  # Original numeric column + encoded columns

        # Test label encoding
        df_label = encode_categorical_data(cat_file, encoding_type='label')
        self.assertTrue(df_label['category1'].dtype in ['int32', 'int64'])
        self.assertTrue(df_label['category2'].dtype in ['int32', 'int64'])

        # Clean up
        if os.path.exists(cat_file):
            os.remove(cat_file)

    def test_check_data_quality(self):
        # Create test data with various quality issues
        quality_data = pd.DataFrame({
            'numeric': [1, 2, None, 1000, 5],
            'categorical': ['A', 'B', 'A', 'B', 'Invalid'],
            'duplicated': [1, 1, 1, 2, 2]
        })
        quality_file = 'quality_data.csv'
        quality_data.to_csv(quality_file, index=False)

        quality_report = check_data_quality(quality_file)
        
        # Test missing value detection
        self.assertIn('missing_values', quality_report)
        self.assertEqual(quality_report['missing_values']['numeric'], 1)

        # Test duplicate row detection
        self.assertIn('duplicate_rows', quality_report)
        self.assertEqual(quality_report['duplicate_rows'], 0)

        # Test data type consistency check
        self.assertIn('inconsistent_dtypes', quality_report)

        # Clean up
        if os.path.exists(quality_file):
            os.remove(quality_file)

    def test_merge_data_sources(self):
        # Create multiple test data sources
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['A', 'B', 'C']
        })
        df2 = pd.DataFrame({
            'id': [4, 5, 6],
            'value': ['D', 'E', 'F']
        })
        
        file1 = 'source1.csv'
        file2 = 'source2.csv'
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)

        # Test merging
        merged_df = merge_data_sources([file1, file2])
        self.assertEqual(merged_df.shape[0], 6)
        self.assertEqual(merged_df.shape[1], 2)

        # Test empty list
        with self.assertRaises(ValueError):
            merge_data_sources([])

        # Clean up
        for file in [file1, file2]:
            if os.path.exists(file):
                os.remove(file)

    def test_save_preprocessed_df(self):
        """Test the save_preprocessed_df function for various scenarios"""
        
        # Prepare test data
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Test case 1: Process a regular filename
        original_file = 'test_data.csv'
        test_df.to_csv(original_file, index=False)
        
        new_path = save_preprocessed_df(original_file, test_df, 'test_step')
        self.assertEqual(new_path, '000-test_step-test_data.csv')
        self.assertTrue(os.path.exists(new_path))
        
        # Test case 2: Process a filename that already has step numbering
        preprocessed_file = '000-test_step-test_data.csv'
        new_path = save_preprocessed_df(preprocessed_file, test_df, 'next_step')
        self.assertEqual(new_path, '001-next_step-test_data.csv')
        self.assertTrue(os.path.exists(new_path))
        
        # Test case 3: Process different file formats
        # Excel file
        excel_file = 'test_data.xlsx'
        test_df.to_excel(excel_file, index=False)
        new_path = save_preprocessed_df(excel_file, test_df, 'test_step')
        self.assertEqual(new_path, '000-test_step-test_data.xlsx')
        self.assertTrue(os.path.exists(new_path))
        
        # JSON file
        json_file = 'test_data.json'
        test_df.to_json(json_file)
        new_path = save_preprocessed_df(json_file, test_df, 'test_step')
        self.assertEqual(new_path, '000-test_step-test_data.json')
        self.assertTrue(os.path.exists(new_path))
        
        # Test case 4: Process filename with path
        path_file = os.path.join('subfolder', 'test_data.csv')
        os.makedirs('subfolder', exist_ok=True)
        test_df.to_csv(path_file, index=False)
        new_path = save_preprocessed_df(path_file, test_df, 'test_step')
        expected_path = os.path.join('subfolder', '000-test_step-test_data.csv')
        self.assertEqual(new_path, expected_path)
        self.assertTrue(os.path.exists(new_path))
        
        # Test case 5: Test invalid file format
        invalid_file = 'test_data.txt'
        with self.assertRaises(ValueError):
            save_preprocessed_df(invalid_file, test_df, 'test_step')
        
        # Clean up test files
        files_to_clean = [
            original_file,
            '000-test_step-test_data.csv',
            '001-next_step-test_data.csv',
            excel_file,
            '000-test_step-test_data.xlsx',
            json_file,
            '000-test_step-test_data.json',
            path_file,
            os.path.join('subfolder', '000-test_step-test_data.csv')
        ]
        
        for file in files_to_clean:
            if os.path.exists(file):
                os.remove(file)
        
        # Clean up test directory
        if os.path.exists('subfolder'):
            os.rmdir('subfolder')

if __name__ == '__main__':
    unittest.main()