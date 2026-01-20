"""
Aadhar Enrollment Data Analysis
Merges 3 CSV data files and performs data cleaning
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. LOAD AND MERGE DATA
# =============================================================================

def load_and_merge_data():
    """
    Load all 3 Aadhar enrollment CSV files and merge them into a single DataFrame.
    """
    print("=" * 60)
    print("LOADING AADHAR ENROLLMENT DATA FILES")
    print("=" * 60)
    
    # Define file paths
    file_paths = [
        'data/api_data_aadhar_enrolment_0_500000.csv',
        'data/api_data_aadhar_enrolment_500000_1000000.csv',
        'data/api_data_aadhar_enrolment_1000000_1006029.csv'
    ]
    
    # Load each file
    dataframes = []
    for file_path in file_paths:
        print(f"Loading: {file_path}")
        df = pd.read_csv(file_path)
        print(f"   * Rows: {len(df):,} | Columns: {len(df.columns)}")
        dataframes.append(df)
    
    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n[OK] MERGED DATA: {len(merged_df):,} total rows")
    
    return merged_df


# =============================================================================
# 2. DATA CLEANING FUNCTIONS
# =============================================================================

def check_data_quality(df):
    """
    Check and report data quality issues before cleaning.
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    print(f"\n[STATS] Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n[INFO] Columns: {list(df.columns)}")
    
    # Check data types
    print("\n[CHECK] Data Types:")
    print(df.dtypes.to_string())
    
    # Check missing values
    print("\n[CHECK] Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_info.to_string())
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\n[CHECK] Duplicate Rows: {duplicates:,} ({(duplicates/len(df)*100):.2f}%)")
    
    return missing_info


def clean_date_column(df):
    """
    Clean and standardize the date column.
    """
    print("\n" + "-" * 40)
    print("[CLEAN] Cleaning Date Column")
    print("-" * 40)
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Check for invalid dates
    invalid_dates = df['date'].isnull().sum()
    print(f"   Invalid dates found: {invalid_dates:,}")
    
    # Extract date components for analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['date'].dt.day_name()
    df['month_name'] = df['date'].dt.month_name()
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('Int64')
    
    print(f"   * Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   * Added columns: year, month, day, day_of_week, day_name, month_name, quarter, week_of_year")
    
    return df


def clean_categorical_columns(df):
    """
    Clean and standardize categorical columns (state, district).
    """
    print("\n" + "-" * 40)
    print("[CLEAN] Cleaning Categorical Columns")
    print("-" * 40)
    
    # Clean state column
    df['state'] = df['state'].astype(str).str.strip()
    df['state'] = df['state'].str.title()  # Standardize capitalization
    df['state'] = df['state'].replace(['Nan', 'None', 'nan', 'NONE', ''], np.nan)
    # Remove special characters from state names
    df['state'] = df['state'].str.replace(r'[*?#@!]', '', regex=True).str.strip()
    
    unique_states = df['state'].nunique()
    print(f"   * Unique States: {unique_states}")
    
    # Clean district column
    df['district'] = df['district'].astype(str).str.strip()
    df['district'] = df['district'].str.title()  # Standardize capitalization
    df['district'] = df['district'].replace(['Nan', 'None', 'nan', 'NONE', ''], np.nan)
    # Remove special characters from district names
    df['district'] = df['district'].str.replace(r'[*?#@!]', '', regex=True).str.strip()
    
    unique_districts = df['district'].nunique()
    print(f"   * Unique Districts: {unique_districts}")
    
    return df


def clean_pincode_column(df):
    """
    Clean and validate pincode column.
    """
    print("\n" + "-" * 40)
    print("[CLEAN] Cleaning Pincode Column")
    print("-" * 40)
    
    # Convert pincode to string and ensure 6 digits
    df['pincode'] = df['pincode'].astype(str).str.strip()
    
    # Remove invalid pincodes (not 6 digits or non-numeric)
    def validate_pincode(pincode):
        try:
            if len(str(pincode)) == 6 and str(pincode).isdigit():
                return pincode
            return np.nan
        except:
            return np.nan
    
    # Check for invalid pincodes before cleaning
    invalid_before = df['pincode'].apply(lambda x: len(str(x)) != 6 or not str(x).isdigit()).sum()
    
    df['pincode'] = df['pincode'].apply(validate_pincode)
    
    invalid_after = df['pincode'].isnull().sum()
    print(f"   * Invalid pincodes cleaned: {invalid_before:,}")
    print(f"   * Unique valid pincodes: {df['pincode'].nunique():,}")
    
    return df


def clean_numeric_columns(df):
    """
    Clean and validate age group numeric columns.
    """
    print("\n" + "-" * 40)
    print("[CLEAN] Cleaning Numeric Columns (Age Groups)")
    print("-" * 40)
    
    age_columns = ['age_0_5', 'age_5_17', 'age_18_greater']
    
    for col in age_columns:
        # Convert to numeric, coerce errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for negative values
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"   [WARNING] {col}: Found {negative_count:,} negative values - setting to NaN")
            df.loc[df[col] < 0, col] = np.nan
        
        # Fill NaN with 0 (assuming no enrollment means 0)
        df[col] = df[col].fillna(0).astype(int)
        
        print(f"   * {col}: min={df[col].min()}, max={df[col].max():,}, mean={df[col].mean():.2f}")
    
    # Calculate total enrollments per row
    df['total_enrollments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    print(f"\n   * Added 'total_enrollments' column (sum of all age groups)")
    print(f"   * Total enrollments range: {df['total_enrollments'].min()} to {df['total_enrollments'].max():,}")
    
    return df


def remove_duplicates(df):
    """
    Identify and remove duplicate rows.
    """
    print("\n" + "-" * 40)
    print("[CLEAN] Removing Duplicates")
    print("-" * 40)
    
    initial_rows = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    removed = initial_rows - len(df)
    print(f"   * Removed {removed:,} duplicate rows")
    print(f"   * Remaining rows: {len(df):,}")
    
    return df


def handle_missing_values(df):
    """
    Handle remaining missing values.
    """
    print("\n" + "-" * 40)
    print("[CLEAN] Handling Missing Values")
    print("-" * 40)
    
    # Check missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    print(f"   Total missing values: {total_missing:,}")
    
    if total_missing > 0:
        print("\n   Missing values by column:")
        for col, count in missing[missing > 0].items():
            print(f"   - {col}: {count:,}")
    
    # Drop rows with missing critical columns (date, state, district)
    critical_cols = ['date', 'state', 'district']
    rows_before = len(df)
    df = df.dropna(subset=critical_cols)
    rows_dropped = rows_before - len(df)
    
    if rows_dropped > 0:
        print(f"\n   * Dropped {rows_dropped:,} rows with missing critical values")
    
    return df


# =============================================================================
# 3. DATA SUMMARY FUNCTIONS
# =============================================================================

def generate_summary(df):
    """
    Generate summary statistics for the cleaned data.
    """
    print("\n" + "=" * 60)
    print("CLEANED DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n[STATS] Final Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print(f"\n[INFO] Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2}. {col} ({df[col].dtype})")
    
    print(f"\n[DATE] Date Range: {df['date'].min().strftime('%d-%m-%Y')} to {df['date'].max().strftime('%d-%m-%Y')}")
    
    print(f"\n[GEO] Geographic Coverage:")
    print(f"   - States: {df['state'].nunique():,}")
    print(f"   - Districts: {df['district'].nunique():,}")
    print(f"   - Pincodes: {df['pincode'].nunique():,}")
    
    print(f"\n[TREND] Enrollment Statistics:")
    print(f"   - Total Records: {len(df):,}")
    print(f"   - Total Enrollments: {df['total_enrollments'].sum():,}")
    print(f"   - Age 0-5 Enrollments: {df['age_0_5'].sum():,}")
    print(f"   - Age 5-17 Enrollments: {df['age_5_17'].sum():,}")
    print(f"   - Age 18+ Enrollments: {df['age_18_greater'].sum():,}")
    
    print(f"\n[STATS] Enrollments Per Record:")
    print(df[['age_0_5', 'age_5_17', 'age_18_greater', 'total_enrollments']].describe().round(2).to_string())
    
    return df


def get_top_states(df, n=10):
    """
    Get top states by total enrollments.
    """
    print(f"\n[TOP] Top {n} States by Total Enrollments:")
    top_states = df.groupby('state')['total_enrollments'].sum().nlargest(n)
    for i, (state, count) in enumerate(top_states.items(), 1):
        print(f"   {i:2}. {state}: {count:,}")
    return top_states


def get_top_districts(df, n=10):
    """
    Get top districts by total enrollments.
    """
    print(f"\n[TOP] Top {n} Districts by Total Enrollments:")
    top_districts = df.groupby(['state', 'district'])['total_enrollments'].sum().nlargest(n)
    for i, ((state, district), count) in enumerate(top_districts.items(), 1):
        print(f"   {i:2}. {district} ({state}): {count:,}")
    return top_districts


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to execute the data processing pipeline.
    """
    print("\n" + "*" * 30)
    print("  AADHAR ENROLLMENT DATA ANALYSIS PIPELINE")
    print("*" * 30 + "\n")
    
    # Step 1: Load and merge data
    df = load_and_merge_data()
    
    # Step 2: Check initial data quality
    check_data_quality(df)
    
    # Step 3: Data Cleaning Pipeline
    print("\n" + "=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)
    
    df = clean_date_column(df)
    df = clean_categorical_columns(df)
    df = clean_pincode_column(df)
    df = clean_numeric_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    
    # Step 4: Generate Summary
    df = generate_summary(df)
    
    # Step 5: Top Statistics
    get_top_states(df)
    get_top_districts(df)
    
    # Step 6: Save cleaned data
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)
    
    output_file = 'data/aadhar_enrollment_cleaned.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Cleaned data saved to: {output_file}")
    print(f"   File size: {len(df):,} rows × {len(df.columns)} columns")
    
    print("\n" + "=" * 60)
    print("* DATA PROCESSING COMPLETE! *")
    print("=" * 60 + "\n")
    
    return df


# Run the pipeline
if __name__ == "__main__":
    cleaned_df = main()