MAPPING_VERSION = "v1.0"

@st.cache_data
def load_mapping(path):
    m = pd.read_csv(path)
    m.columns = m.columns.str.strip()
    return m

@st.cache_data
def prepare_all(df_raw, mapping_path, mapping_version):
    df_fact = prepare_fact_table(df_raw, mapping_path=mapping_path)
    df_users = build_user_view(df_fact)
    df_tests = build_test_view(df_fact)
    return df_fact, df_users, df_tests
