# app/dash_board_tab.py
'''
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast  # For safely parsing 'labels' from string

def dashboard_tab():
    st.header(" Metadata Dashboard")

    try:
        # Load data
        df = pd.read_csv("data/full_df.csv")

        # Convert 'labels' from string to list if needed
        if isinstance(df['labels'].iloc[0], str):
            df['labels'] = df['labels'].apply(ast.literal_eval)

        # Age Distribution
        st.subheader("Age Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Patient Age"], bins=15, kde=True, ax=ax1)
        st.pyplot(fig1)

        # Sex Distribution
        st.subheader("Sex Distribution")
        fig2, ax2 = plt.subplots()
        sex_counts = df["Patient Sex"].value_counts()
        ax2.pie(sex_counts.values, labels=sex_counts.index, autopct="%1.1f%%", startangle=140)
        ax2.axis("equal")
        st.pyplot(fig2)

        # Disease label counts from 'labels'
        st.subheader("Disease Label Distribution")
        all_labels = [label for sublist in df["labels"] for label in sublist]
        label_counts = pd.Series(all_labels).value_counts()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax3)
        ax3.set_xlabel("Disease Label")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error loading dashboard: {e}") '''
# app/dash_board_tab.py
'''
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import time

# Load and cache the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/full_df.csv")

# Main dashboard function
def dashboard_tab():
    st.title("📊 Retina Metadata Dashboard")
    st.markdown("Get insights and trends from the dataset used in RetinoScan. Interact with graphs, filters, and metrics.")

    df = load_data()

    # 🧮 Dataset snapshot
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("---")

    # 🔢 KPI Cards
    st.subheader("📊 Key Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧪 Total Records", len(df))
    col2.metric("🧬 Unique Diseases", df['disease'].nunique())
    col3.metric("👥 % Male", f"{round((df['sex'] == 'M').mean() * 100)}%")
    style_metric_cards()
    st.markdown("---")

    # 🎯 Filters
    st.subheader("🎛️ Filters and Custom View")
    selected_disease = st.selectbox("🔍 Select Disease", options=df["disease"].unique())
    filtered_df = df[df["disease"] == selected_disease]

    st.markdown("---")

    # 📊 Simple Chart: Disease Count Bar Plot
    st.subheader("📌 Disease Distribution")
    fig_bar = px.histogram(df, x="disease", color="sex", barmode="group",
                           title="Distribution of Disease by Gender",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 🧓 Simple Chart: Age Distribution
    st.subheader("👵 Age Distribution")
    fig_age = px.histogram(df, x="age", nbins=30, color="sex", marginal="box",
                           title="Age Distribution by Gender")
    st.plotly_chart(fig_age, use_container_width=True)

    # 🥧 Medium Level: Pie chart of left/right eye involvement
    st.subheader("👁️ Eye Involvement")
    eye_counts = df["eye"].value_counts()
    fig_pie = px.pie(names=eye_counts.index, values=eye_counts.values,
                     title="Distribution of Left vs Right Eye",
                     color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie, use_container_width=True)

    # 📉 Medium: Age vs Disease Scatter Plot
    st.subheader("📍 Age vs Disease")
    fig_scatter = px.strip(df, x="disease", y="age", color="sex", stripmode="overlay",
                           title="Age vs Disease Category")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 📈 Advanced: Animated Age Distribution Over Time (if available)
    if "timestamp" in df.columns:
        st.subheader("🎥 Animated Age Trend (Optional)")
        fig_anim = px.histogram(df, x="age", animation_frame="timestamp", nbins=20,
                                title="Age Distribution Over Time")
        st.plotly_chart(fig_anim, use_container_width=True)

    # 🧠 Advanced: Heatmap of correlation (age vs visual acuity or other numeric cols)
    st.subheader("🔬 Correlation Heatmap (Numerical Features)")
    numeric_cols = df.select_dtypes(include='number')
    if len(numeric_cols.columns) > 1:
        corr = numeric_cols.corr()
        fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Bluered_r",
                                title="Correlation Matrix")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("📌 Not enough numerical columns to generate correlation heatmap.")

    # 💬 Closing remarks
    st.markdown("---")
    st.success("✅ Dashboard loaded successfully. Explore other tabs for deeper insights.")'''
'''
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import time

# Load and cache the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/full_df.csv")

    # ✅ Clean disease name from 'labels' column: e.g., ['D'] -> D
    df["disease"] = df["labels"].str.strip("[]'")

    return df

# Main dashboard function
def dashboard_tab():
    st.title("📊 Retina Metadata Dashboard")
    st.markdown("Get insights and trends from the dataset used in RetinoScan. Interact with graphs, filters, and metrics.")

    df = load_data()

    # 🧮 Dataset snapshot
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("---")

    # 🔢 KPI Cards
    st.subheader("📊 Key Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧪 Total Records", len(df))
    col2.metric("🧬 Unique Diseases", df['disease'].nunique())
    col3.metric("👥 % Male", f"{round((df['Patient Sex'] == 'Male').mean() * 100)}%")
    style_metric_cards()
    st.markdown("---")

    # 🎯 Filters
    st.subheader("🎛️ Filters and Custom View")
    selected_disease = st.selectbox("🔍 Select Disease", options=df["disease"].unique())
    filtered_df = df[df["disease"] == selected_disease]

    st.markdown("---")

    # 📊 Simple Chart: Disease Count Bar Plot
    st.subheader("📌 Disease Distribution")
    fig_bar = px.histogram(df, x="disease", color="Patient Sex", barmode="group",
                           title="Distribution of Disease by Gender",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 🧓 Simple Chart: Age Distribution
    st.subheader("👵 Age Distribution")
    fig_age = px.histogram(df, x="Patient Age", nbins=30, color="Patient Sex", marginal="box",
                           title="Age Distribution by Gender")
    st.plotly_chart(fig_age, use_container_width=True)

    # 🥧 Medium Level: Pie chart of left/right eye involvement
    st.subheader("👁️ Eye Involvement")
    if "eye" in df.columns:
        eye_counts = df["eye"].value_counts()
        fig_pie = px.pie(names=eye_counts.index, values=eye_counts.values,
                         title="Distribution of Left vs Right Eye",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 📉 Medium: Age vs Disease Scatter Plot
    st.subheader("📍 Age vs Disease")
    fig_scatter = px.strip(df, x="disease", y="Patient Age", color="Patient Sex", stripmode="overlay",
                           title="Age vs Disease Category")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 📈 Advanced: Animated Age Distribution Over Time (if available)
    if "timestamp" in df.columns:
        st.subheader("🎥 Animated Age Trend (Optional)")
        fig_anim = px.histogram(df, x="Patient Age", animation_frame="timestamp", nbins=20,
                                title="Age Distribution Over Time")
        st.plotly_chart(fig_anim, use_container_width=True)

    # 🧠 Advanced: Heatmap of correlation (only for numeric columns)
    st.subheader("🔬 Correlation Heatmap (Numerical Features)")
    numeric_cols = df.select_dtypes(include='number')
    if len(numeric_cols.columns) > 1:
        corr = numeric_cols.corr()
        fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Bluered_r",
                                title="Correlation Matrix")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("📌 Not enough numerical columns to generate correlation heatmap.")

    # 💬 Closing remarks
    st.markdown("---")
    st.success("✅ Dashboard loaded successfully. Explore other tabs for deeper insights.")'''
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

# Load and cache the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/full_df.csv")

# Main dashboard function
def dashboard_tab():
    st.title("📊 Retina Metadata Dashboard")
    st.markdown("Get insights and trends from the dataset used in RetinoScan. Interact with graphs, filters, and metrics.")

    df = load_data()

    # 🧮 Dataset snapshot
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("---")

    # 🔢 KPI Cards
    st.subheader("📊 Key Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧪 Total Records", len(df))
    col2.metric("🧬 Unique Labels", df['labels'].nunique())
    col3.metric("👥 % Male", f"{round((df['Patient Sex'] == 'Male').mean() * 100)}%")
    style_metric_cards()
    st.markdown("---")

    # 🎯 Filters
    st.subheader("🎛️ Filters and Custom View")
    selected_label = st.selectbox("🔍 Select Label", options=df["labels"].unique())
    filtered_df = df[df["labels"] == selected_label]
    st.dataframe(filtered_df.head(), use_container_width=True)
    st.markdown("---")

    # 📊 Bar Plot: Labels Distribution
    st.subheader("📌 Label Distribution")
    fig_bar = px.histogram(df, x="labels", color="Patient Sex", barmode="group",
                           title="Distribution of Labels by Gender",
                           color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 👵 Age Distribution
    st.subheader("👵 Age Distribution by Gender")
    fig_age = px.histogram(df, x="Patient Age", nbins=30, color="Patient Sex", marginal="box",
                           title="Age Distribution by Gender")
    st.plotly_chart(fig_age, use_container_width=True)

    # 🧠 Correlation Heatmap (numerical only)
    st.subheader("🧬 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include='number')
    if len(numeric_cols.columns) > 1:
        corr = numeric_cols.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="PuOr", aspect="auto",
                             title="Correlation Matrix (Numeric Features)")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns to display correlation.")

    # 👁️ Pie chart of eye side using filename
    '''
    st.subheader("👁️ Eye Involvement")
    df["eye"] = df["filename"].apply(lambda x: "Left" if "left" in x else "Right")
    eye_counts = df["eye"].value_counts()
    fig_eye = px.pie(names=eye_counts.index, values=eye_counts.values,
                     title="Distribution of Left vs Right Eye Images",
                     color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_eye, use_container_width=True) '''
    # Confirmed counts
    left_count = 3198
    right_count = 3194
    total_count = left_count + right_count

# Calculate exact percentages
    left_percent = (left_count / total_count) * 100
    right_percent = (right_count / total_count) * 100

    st.subheader("👁️ Eye Involvement ")
    st.markdown(f"""
The total number of images is **{total_count}**.

- **Left Eye Images:** **{left_count}** ($\mathbf{{ {left_percent:.2f}\% }}$)
- **Right Eye Images:** **{right_count}** ($\mathbf{{ {right_percent:.2f}\% }}$)
""")
    
    



    # 🩺 Left vs Right Eye Diagnosis Comparison
    st.subheader("🩺 Left vs Right Eye Diagnosis")
    if "Left-Diagnostic Keywords" in df.columns and "Right-Diagnostic Keywords" in df.columns:
        diag_df = df[["Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]].melt(
            var_name="Eye", value_name="Diagnosis"
        )
        fig_diag = px.histogram(diag_df, x="Diagnosis", color="Eye", barmode="group",
                                title="Frequency of Left vs Right Eye Diagnoses")
        st.plotly_chart(fig_diag, use_container_width=True)
    else:
        st.warning("🛑 Diagnosis columns not found in the dataset.")

    # 👫 Gender vs Target Label Analysis
    st.subheader("👫 Gender vs Target Labels")
    target_df = df.copy()
    target_df['target_str'] = target_df['target'].astype(str)
    fig_gender_target = px.histogram(target_df, x="target_str", color="Patient Sex",
                                     barmode="group", title="Target Label Distribution by Gender")
    st.plotly_chart(fig_gender_target, use_container_width=True)

    # 👶 Age Groups vs Disease Types
    st.subheader("📊 Age Groups vs Disease Types")
    bins = [0, 30, 45, 60, 75, 90]
    labels = ['0-30', '31-45', '46-60', '61-75', '76-90']
    df["Age Group"] = pd.cut(df["Patient Age"], bins=bins, labels=labels, right=False)
    age_disease_df = df.copy()
    age_disease_df["Disease Label"] = df["labels"]
    fig_age_group = px.histogram(age_disease_df, x="Age Group", color="Disease Label", barmode="group",
                                 title="Age Groups vs Disease Types")
    st.plotly_chart(fig_age_group, use_container_width=True)

    # 🎥 Optional: Animation over time
    if "timestamp" in df.columns:
        st.subheader("📽️ Animated Age Distribution")
        fig_anim = px.histogram(df, x="Patient Age", animation_frame="timestamp",
                                nbins=20, title="Animated Age Distribution Over Time")
        st.plotly_chart(fig_anim, use_container_width=True)

    # ✅ Final Note
    st.markdown("---")
    st.success(" Dashboard fully loaded. Explore more later as information will be adding up later.")




