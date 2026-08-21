import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None
    if "cleaning_applied" not in st.session_state:
        st.session_state.cleaning_applied = False
    if "cleaning_report" not in st.session_state:
        st.session_state.cleaning_report = {}


initialize_session_state()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dataset(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load dataset from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"❌ Unsupported file format: {file_extension}")
            return None
        
        if df.empty:
            st.error("❌ The uploaded file is empty or contains only headers.")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None


# ============================================================================
# DATA ANALYSIS FUNCTIONS
# ============================================================================

def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
    }


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate detailed information about each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with column information
    """
    info_data = []
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        missing_pct = (null_count / len(df)) * 100
        unique_values = df[col].nunique()
        
        info_data.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": non_null_count,
            "Missing Count": null_count,
            "Missing %": f"{missing_pct:.2f}%",
            "Unique Values": unique_values,
        })
    
    return pd.DataFrame(info_data)


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing value analysis
    """
    missing_data = []
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            missing_data.append({
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Missing Values": missing_count,
                "Missing %": f"{missing_pct:.2f}%",
            })
    
    if missing_data:
        return pd.DataFrame(missing_data)
    else:
        return pd.DataFrame()


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect and categorize column types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with column type categories
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str,
    custom_strategies: Dict[str, str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values using specified strategy.
    
    Args:
        df: Input DataFrame
        strategy: "remove", "auto", or "custom"
        custom_strategies: Dictionary of column-specific strategies
        
    Returns:
        Tuple of (cleaned DataFrame, report dictionary)
    """
    df_copy = df.copy()
    report = {"missing_values_resolved": 0, "rows_removed": 0}
    
    if strategy == "remove_any":
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna()
        report["rows_removed"] = initial_rows - len(df_copy)
        report["missing_values_resolved"] = df.isnull().sum().sum()
    
    elif strategy == "remove_all":
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna(how="all")
        report["rows_removed"] = initial_rows - len(df_copy)
        report["missing_values_resolved"] = df.isnull().sum().sum() - df_copy.isnull().sum().sum()
    
    elif strategy == "auto":
        report = _auto_fill_missing(df_copy)
    
    elif strategy == "custom" and custom_strategies:
        report = _custom_fill_missing(df_copy, custom_strategies)
    
    return df_copy, report


def _auto_fill_missing(df: pd.DataFrame) -> Dict:
    """
    Automatically fill missing values based on column type.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Report dictionary with changes made
    """
    df_copy = df.copy()
    report = {"missing_values_resolved": 0, "strategies_applied": {}}
    
    col_types = detect_column_types(df_copy)
    
    # Handle numeric columns with median
    for col in col_types["numeric"]:
        if df_copy[col].isna().any():
            median_val = df_copy[col].median()
            if pd.notna(median_val):
                df_copy[col].fillna(median_val, inplace=True)
                report["strategies_applied"][col] = f"Median ({median_val:.2f})"
                report["missing_values_resolved"] += df[col].isna().sum()
    
    # Handle categorical columns with mode
    for col in col_types["categorical"]:
        if df_copy[col].isna().any():
            mode_val = df_copy[col].mode()
            if len(mode_val) > 0:
                df_copy[col].fillna(mode_val[0], inplace=True)
                report["strategies_applied"][col] = f"Mode ({mode_val[0]})"
            else:
                df_copy[col].fillna("Unknown", inplace=True)
                report["strategies_applied"][col] = "Unknown (no mode)"
            report["missing_values_resolved"] += df[col].isna().sum()
    
    # Handle datetime columns with forward fill
    for col in col_types["datetime"]:
        if df_copy[col].isna().any():
            df_copy[col].fillna(method="ffill", inplace=True)
            df_copy[col].fillna(method="bfill", inplace=True)
            report["strategies_applied"][col] = "Forward/Backward Fill"
            report["missing_values_resolved"] += df[col].isna().sum()
    
    return report


def _custom_fill_missing(df: pd.DataFrame, strategies: Dict) -> Dict:
    """
    Fill missing values using custom strategies.
    
    Args:
        df: Input DataFrame
        strategies: Dictionary of column names and fill strategies
        
    Returns:
        Report dictionary with changes made
    """
    df_copy = df.copy()
    report = {"missing_values_resolved": 0, "strategies_applied": {}}
    
    for col, strategy in strategies.items():
        if col not in df_copy.columns or df_copy[col].isna().sum() == 0:
            continue
        
        initial_missing = df[col].isna().sum()
        
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == "mode":
            mode_val = df_copy[col].mode()
            df_copy[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown", inplace=True)
        elif strategy == "ffill":
            df_copy[col].fillna(method="ffill", inplace=True)
        elif strategy == "bfill":
            df_copy[col].fillna(method="bfill", inplace=True)
        elif strategy.startswith("custom:"):
            value = strategy.split(":", 1)[1]
            df_copy[col].fillna(value, inplace=True)
        
        report["strategies_applied"][col] = strategy
        report["missing_values_resolved"] += initial_missing
    
    return report


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed)
    """
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df_cleaned)
    
    return df_cleaned, duplicates_removed


def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Validate the cleaned dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "remaining_missing": df.isnull().sum().sum(),
        "remaining_duplicates": df.duplicated().sum(),
        "rows": len(df),
        "columns": len(df.columns),
        "empty_columns": [col for col in df.columns if df[col].isna().all()],
        "fully_empty_rows": (df.isna().all(axis=1)).sum(),
    }
    
    return validation


# ============================================================================
# FILE EXPORT FUNCTIONS
# ============================================================================

def create_csv_download(df: pd.DataFrame, filename: str) -> bytes:
    """
    Create CSV download bytes.
    
    Args:
        df: DataFrame to export
        filename: Original filename for naming
        
    Returns:
        CSV bytes
    """
    base_name = filename.rsplit(".", 1)[0]
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()


def create_excel_download(df: pd.DataFrame, filename: str) -> bytes:
    """
    Create Excel download bytes.
    
    Args:
        df: DataFrame to export
        filename: Original filename for naming
        
    Returns:
        Excel bytes
    """
    base_name = filename.rsplit(".", 1)[0]
    excel_buffer = BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cleaned Data", index=False)
    
    excel_buffer.seek(0)
    return excel_buffer.getvalue()


# ============================================================================
# UI FUNCTIONS
# ============================================================================

def display_header():
    """Display application header and title."""
    st.set_page_config(
        page_title="Data Cleaning Platform",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🧹 DATA CLEANING & QUALITY PLATFORM</h1>
        <p>Professional automated data cleaning for CSV and Excel files</p>
    </div>
    """, unsafe_allow_html=True)


def display_dataset_overview(df: pd.DataFrame):
    """Display dataset overview with KPI cards."""
    summary = get_dataset_summary(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="📊 Total Rows", value=summary["rows"])
    with col2:
        st.metric(label="📋 Total Columns", value=summary["columns"])
    with col3:
        st.metric(label="❌ Missing Values", value=summary["missing_values"])
    with col4:
        st.metric(label="🔄 Duplicate Rows", value=summary["duplicate_rows"])
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric(label="🔢 Numeric Columns", value=summary["numeric_cols"])
    with col6:
        st.metric(label="📝 Categorical Columns", value=summary["categorical_cols"])


def display_column_information(df: pd.DataFrame):
    """Display detailed column information."""
    st.subheader("📋 Column Information")
    
    col_info = get_column_info(df)
    st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    # Dataset info summary
    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
    categorical_count = len(df.select_dtypes(include=["object"]).columns)
    
    st.info(f"""
    **Dataset Summary:**
    - Contains **{len(df)} rows** and **{len(df.columns)} columns**
    - **{numeric_count} numeric** columns and **{categorical_count} categorical** columns
    - **{df.isnull().sum().sum()} total missing values** across the dataset
    """)


def display_data_preview(df: pd.DataFrame):
    """Display dataset preview with row selection."""
    st.subheader("👁️ Data Preview")
    
    preview_rows = st.selectbox(
        "Select number of rows to display:",
        [5, 10, 25, 50, 100],
        index=1,
        key="preview_rows",
    )
    
    st.dataframe(df.head(preview_rows), use_container_width=True)


def display_missing_value_analysis(df: pd.DataFrame):
    """Display missing value analysis."""
    st.subheader("🔍 Missing Value Analysis")
    
    missing_df = analyze_missing_values(df)
    total_missing = df.isnull().sum().sum()
    cols_with_missing = len(missing_df)
    missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
    
    if not missing_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Missing Cells", total_missing)
        with col2:
            st.metric("Columns with Missing Values", cols_with_missing)
        with col3:
            st.metric("% of Dataset Missing", f"{missing_pct:.2f}%")
        
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values detected in this dataset.")


def display_cleaning_configuration(df: pd.DataFrame):
    """Display cleaning configuration options."""
    st.subheader("⚙️ Cleaning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Value Strategy**")
        missing_strategy = st.radio(
            "How would you like to handle missing values?",
            ["remove_any", "remove_all", "auto"],
            format_func=lambda x: {
                "remove_any": "🗑️ Remove rows with any missing value",
                "remove_all": "🗑️ Remove rows only when all values are missing",
                "auto": "🤖 Automatically handle (Intelligent)",
            }[x],
            key="missing_strategy",
        )
    
    with col2:
        st.write("**Duplicate Handling**")
        remove_dups = st.checkbox(
            "Remove duplicate records",
            value=True,
            key="remove_duplicates",
        )
    
    # Show preview of changes
    if st.button("📋 Preview Cleaning Operations", key="preview_cleaning"):
        preview_changes(df, missing_strategy, remove_dups)
    
    return missing_strategy, remove_dups


def preview_changes(df: pd.DataFrame, missing_strategy: str, remove_dups: bool):
    """Preview what changes will be made."""
    st.info("**Preview of Cleaning Operations:**")
    
    # Missing values preview
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        if missing_strategy == "remove_any":
            rows_to_remove = df.dropna().shape[0]
            st.write(f"- ❌ Will remove rows with missing values")
        elif missing_strategy == "remove_all":
            rows_to_remove = df.dropna(how="all").shape[0]
            st.write(f"- ❌ Will remove rows where ALL values are missing")
        elif missing_strategy == "auto":
            st.write(f"- 🤖 Will intelligently fill {missing_count} missing values")
    
    # Duplicates preview
    if remove_dups:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.write(f"- 🔄 Will remove {dup_count} duplicate rows")


def execute_cleaning(
    df: pd.DataFrame,
    missing_strategy: str,
    remove_dups: bool,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the cleaning pipeline.
    
    Args:
        df: Original DataFrame
        missing_strategy: Strategy for handling missing values
        remove_dups: Whether to remove duplicates
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    df_cleaned = df.copy()
    report = {
        "missing_values_resolved": 0,
        "rows_removed_missing": 0,
        "duplicates_removed": 0,
        "strategies_applied": {},
    }
    
    # Handle missing values
    if df_cleaned.isnull().sum().sum() > 0:
        df_cleaned, missing_report = handle_missing_values(df_cleaned, missing_strategy)
        report["missing_values_resolved"] = missing_report.get("missing_values_resolved", 0)
        report["rows_removed_missing"] = missing_report.get("rows_removed", 0)
        report["strategies_applied"] = missing_report.get("strategies_applied", {})
    
    # Remove duplicates
    if remove_dups:
        df_cleaned, dups_removed = remove_duplicates(df_cleaned)
        report["duplicates_removed"] = dups_removed
    
    return df_cleaned, report


def display_cleaning_results(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    report: Dict,
):
    """Display before and after comparison."""
    st.subheader("📊 Cleaning Results")
    
    original_summary = get_dataset_summary(original_df)
    cleaned_summary = get_dataset_summary(cleaned_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Rows",
            cleaned_summary["rows"],
            delta=cleaned_summary["rows"] - original_summary["rows"],
        )
    with col2:
        st.metric("Columns", cleaned_summary["columns"])
    with col3:
        st.metric(
            "Missing Values",
            cleaned_summary["missing_values"],
            delta=cleaned_summary["missing_values"] - original_summary["missing_values"],
        )
    with col4:
        st.metric(
            "Duplicates",
            cleaned_summary["duplicate_rows"],
            delta=cleaned_summary["duplicate_rows"] - original_summary["duplicate_rows"],
        )
    with col5:
        rows_retained_pct = (len(cleaned_df) / len(original_df)) * 100
        st.metric("Data Retained", f"{rows_retained_pct:.1f}%")
    
    # Detailed report
    st.write("**Cleaning Summary:**")
    summary_text = f"""
    - **{report.get('missing_values_resolved', 0)} missing values** resolved
    - **{report.get('rows_removed_missing', 0)} rows** removed due to missing values
    - **{report.get('duplicates_removed', 0)} duplicate rows** removed
    - **{len(cleaned_df)} rows** retained in cleaned dataset
    """
    st.success(summary_text)


def display_cleaned_data_preview(df: pd.DataFrame):
    """Display preview of cleaned data."""
    st.subheader("✨ Cleaned Dataset Preview")
    
    preview_mode = st.selectbox(
        "View:",
        ["First N Rows", "Last N Rows", "Random Sample"],
        key="cleaned_preview_mode",
    )
    
    n_rows = st.slider("Number of rows:", 5, min(100, len(df)), 10, key="cleaned_preview_rows")
    
    if preview_mode == "First N Rows":
        st.dataframe(df.head(n_rows), use_container_width=True)
    elif preview_mode == "Last N Rows":
        st.dataframe(df.tail(n_rows), use_container_width=True)
    else:
        st.dataframe(df.sample(min(n_rows, len(df))), use_container_width=True)


def display_data_quality_status(cleaned_df: pd.DataFrame):
    """Display final data quality status."""
    st.subheader("✅ Data Quality Status")
    
    validation = validate_dataset(cleaned_df)
    
    checks = []
    
    if validation["remaining_missing"] == 0:
        checks.append("✅ No missing values")
    else:
        checks.append(f"⚠️ {validation['remaining_missing']} missing values remain")
    
    if validation["remaining_duplicates"] == 0:
        checks.append("✅ No duplicate records")
    else:
        checks.append(f"⚠️ {validation['remaining_duplicates']} duplicates remain")
    
    if not validation["empty_columns"]:
        checks.append("✅ No completely empty columns")
    else:
        checks.append(f"⚠️ {len(validation['empty_columns'])} empty columns found")
    
    if validation["fully_empty_rows"] == 0:
        checks.append("✅ No completely empty rows")
    else:
        checks.append(f"⚠️ {validation['fully_empty_rows']} empty rows found")
    
    for check in checks:
        st.write(check)


def display_download_section(cleaned_df: pd.DataFrame, filename: str):
    """Display download options for cleaned data."""
    st.subheader("📥 Download Cleaned Dataset")
    
    col1, col2 = st.columns(2)
    
    base_name = filename.rsplit(".", 1)[0]
    
    with col1:
        csv_data = create_csv_download(cleaned_df, filename)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"{base_name}_cleaned.csv",
            mime="text/csv",
        )
    
    with col2:
        excel_data = create_excel_download(cleaned_df, filename)
        st.download_button(
            label="📊 Download as Excel",
            data=excel_data,
            file_name=f"{base_name}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application flow."""
    display_header()
    
    # File upload section
    st.markdown("## 📁 Upload Your Dataset")
    st.write("Drag and drop your CSV or Excel file here, or browse your computer.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    
    if uploaded_file is not None:
        # Load dataset
        df = load_dataset(uploaded_file)
        
        if df is not None:
            st.session_state.original_df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.cleaned_df = None
            st.session_state.cleaning_applied = False
            
            st.success(f"✅ Successfully loaded: **{uploaded_file.name}** ({len(df)} rows)")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Overview", "Analysis", "Cleaning", "Results", "Download"]
            )
            
            # Tab 1: Overview
            with tab1:
                st.markdown("### Dataset Overview")
                display_dataset_overview(df)
                
                st.markdown("---")
                display_column_information(df)
                
                st.markdown("---")
                st.markdown("### Data Preview")
                display_data_preview(df)
            
            # Tab 2: Analysis
            with tab2:
                st.markdown("### Missing Value Analysis")
                display_missing_value_analysis(df)
            
            # Tab 3: Cleaning
            with tab3:
                st.markdown("### Configure Cleaning")
                
                missing_strategy, remove_dups = display_cleaning_configuration(df)
                
                st.markdown("---")
                
                if st.button("🚀 Execute Cleaning", key="execute_cleaning"):
                    with st.spinner("Processing..."):
                        cleaned_df, report = execute_cleaning(df, missing_strategy, remove_dups)
                        
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.cleaning_report = report
                        st.session_state.cleaning_applied = True
                    
                    st.success("✅ Cleaning completed successfully!")
            
            # Tab 4: Results
            with tab4:
                if st.session_state.cleaning_applied and st.session_state.cleaned_df is not None:
                    display_cleaning_results(
                        st.session_state.original_df,
                        st.session_state.cleaned_df,
                        st.session_state.cleaning_report,
                    )
                    
                    st.markdown("---")
                    display_cleaned_data_preview(st.session_state.cleaned_df)
                    
                    st.markdown("---")
                    display_data_quality_status(st.session_state.cleaned_df)
                else:
                    st.info("👉 Go to the **Cleaning** tab to execute cleaning operations.")
            
            # Tab 5: Download
            with tab5:
                if st.session_state.cleaning_applied and st.session_state.cleaned_df is not None:
                    display_download_section(
                        st.session_state.cleaned_df,
                        st.session_state.file_name,
                    )
                else:
                    st.info("👉 Complete the cleaning process first to download the cleaned dataset.")
            
            # Reset button in sidebar
            st.sidebar.markdown("---")
            if st.sidebar.button("🔄 Start Over / Reset Dataset", use_container_width=True):
                st.session_state.original_df = None
                st.session_state.cleaned_df = None
                st.session_state.file_name = None
                st.session_state.cleaning_applied = False
                st.session_state.cleaning_report = {}
                st.rerun()
    
    else:
        # Show welcome message when no file is uploaded
        st.info(
            """
            👋 Welcome to the **Data Cleaning & Quality Platform**!
            
            This application helps you:
            - 📊 Analyze data quality issues
            - 🧹 Automatically clean messy datasets
            - 📈 Remove duplicates and handle missing values
            - 📥 Download cleaned data in CSV or Excel format
            
            **To get started:** Upload a CSV or Excel file above.
            """
        )


if __name__ == "__main__":
    main()
