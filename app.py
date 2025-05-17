import streamlit as st
import pandas as pd
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    econ = pd.read_csv('data/Crop_Economics_per_Hectare__Nigeria_.csv')
    calendar_df = pd.read_csv('data/Crop_Planting_Calendar__Nigeria_.csv')
    soil = pd.read_csv('data/Crops_Tagged_by_Soil_Role.csv')
    demand = pd.read_csv('data/Crop_Market_Demand___Buyer_Presence.csv')
    rotation = pd.read_csv('data/Crop_Rotation_Recommendations.csv')
    portfolio = pd.read_csv('data/Crop_Portfolio_Rules_by_Condition.csv')

    calendar_df['Planting Start'] = pd.to_datetime(calendar_df['Planting Start'] + ' 2024', format='%B %d %Y', errors='coerce')
    calendar_df['Planting End'] = pd.to_datetime(calendar_df['Planting End'] + ' 2024', format='%B %d %Y', errors='coerce')

    df = calendar_df.merge(econ, on='Crop', how='inner')
    df = df.merge(soil, on='Crop', how='inner')
    df = df.merge(demand, on='Crop', how='left')

    return df, rotation, portfolio, demand, calendar_df

df = load_data()

# ========== STREAMLIT UI ==========
st.title("🌱 Smart Crop Planning Dashboard")
df, rotation_df, portfolio_df, demand_df, calendar_df = load_data()
# st.sidebar.image("logo.png", width=200)


# Sidebar options
soil_roles = df['Soil Role'].unique().tolist()
months = list(calendar.month_name[1:])  # Jan to Dec
regions = df['Region'].unique().tolist()

selected_soil = st.sidebar.selectbox("Select Soil Role", soil_roles)
selected_region = st.sidebar.selectbox("Select Region", regions)
selected_month = st.sidebar.selectbox("Planting Month", months)
top_n = st.sidebar.slider("Number of Crops to Recommend", min_value=3, max_value=15, value=5)

# Current date
today = datetime.today()
today_str = today.strftime('%B %d %Y')
# current_date = pd.to_datetime(today_str, format='%B %d %Y')

# Convert month to date range
month_idx = months.index(selected_month) + 1
start_date = datetime(2024, month_idx, 1)
end_date = datetime(2024, month_idx, 28)

# Filter logic
filtered = df[
    (df['Soil Role'] == selected_soil) &
    (df['Region'] == selected_region) &
    (df['Planting Start'] <= end_date) &
    (df['Planting End'] >= start_date)
]

recommended = filtered.sort_values('Gross Margin (₦)', ascending=False).head(top_n)

# ========== TABS ===========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Recommendations",
    "📊 Profitability",
    "🛒 Market Demand",
    "🔄 Crop Rotation",
    "🧩 Portfolio Rules",
    "🗓️ Crop Calendar"
])

# --- Tab 1: Recommendations ---
with tab1:
    # Results
    if not recommended.empty:
        st.success(f"Showing top {top_n} crops for '{selected_soil}' that are in planting season ({selected_month})")
        st.dataframe(recommended[['Crop', 'Gross Margin (₦)', 'Avg Yield/Ha', 'Planting Start', 'Planting End']])
    else:
        st.warning(f"No crops found for '{selected_soil}' that can be planted in {selected_month}.")

# --- Tab 2: Profitability ---
with tab2:
    st.subheader("Top 10 Most Profitable Crops per Hectare (Gross Margin ₦)")
    top_crops = df[['Crop', 'Gross Margin (₦)']].drop_duplicates().sort_values('Gross Margin (₦)', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_crops, x='Gross Margin (₦)', y='Crop', ax=ax, palette='YlGn')
    ax.set_title("Top 10 Most Profitable Crops")
    st.pyplot(fig)

# --- Tab 3: Market Demand ---
with tab3:
    st.subheader("📈 Market Demand and Buyer Presence")

    # Clean demand level → numeric score
    demand_level_mapping = {
        "Very High": 5,
        "High": 4,
        "Medium": 3,
        "Low": 2,
        "Very Low": 1
    }

    def extract_demand_score(text):
        for key in demand_level_mapping:
            if key.lower() in str(text).lower():
                return demand_level_mapping[key]
        return None

    demand_df['Demand Score'] = demand_df['Demand Level'].apply(extract_demand_score)

    # Extract number of buyers
    demand_df['Buyer Count'] = demand_df['Major Buyers'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )

    # Top crops by demand score
    st.markdown("### 🔝 Top Crops by Demand Level")
    top_demand = demand_df.sort_values(by='Demand Score', ascending=False).head(10)
    st.bar_chart(top_demand.set_index('Crop')['Demand Score'])

    # Top crops by buyer presence
    st.markdown("### 🛒 Crops with Most Buyers")
    top_buyers = demand_df.sort_values(by='Buyer Count', ascending=False).head(10)
    st.bar_chart(top_buyers.set_index('Crop')['Buyer Count'])

    # Show full demand data
    st.markdown("### 📄 Full Demand & Buyer Table")
    st.dataframe(demand_df)

# --- Tab 4: Crop Rotation ---
with tab4:
    st.subheader("🔁 Recommended Crop Rotations")

    import plotly.graph_objects as go

    # Create edge list
    edges = []
    for _, row in rotation_df.iterrows():
        current_crop = row['Crop A (Just Harvested)']
        next_crops = str(row['Recommended Next Crops']).split(',')
        for next_crop in next_crops:
            next_crop = next_crop.strip()
            if next_crop:
                edges.append((current_crop, next_crop))

    # Build directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, k=0.8, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightgreen',
            size=20,
            line=dict(width=2)))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Crop Rotation Network',
                        #titlefont_size=18,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # Create arrows (annotations) to show direction of edges
    annotations = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Add arrow annotation
        annotations.append(
            dict(
                ax=x0,
                ay=y0,
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#636efa",
                opacity=0.7
            )
        )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Crop Rotation Network (with arrows)',
                        #titlefont_size=18,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        annotations=annotations  # ← Add arrows here
                    ))

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rotation_df)

# --- Tab 5: Portfolio Rules ---
with tab5:
    st.subheader("📦 Crop Portfolio Recommendations by Condition")

    import plotly.graph_objects as go

    # Create list of edges: (Condition, Recommended Crop)
    edges = []
    for _, row in portfolio_df.iterrows():
        condition = row["Condition"]
        crops = [c.strip() for c in str(row["Recommended Crop Portfolio"]).split(",") if c.strip()]
        for crop in crops:
            edges.append((condition, crop))

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Assign node types
    node_types = {}
    for u, v in edges:
        node_types[u] = "Condition"
        node_types[v] = "Crop"

    # Assign bipartite layout
    pos = {}
    x_gap = 1.0
    y_gap = 0.5
    conditions = sorted(set([u for u in node_types if node_types[u] == "Condition"]))
    crops = sorted(set([u for u in node_types if node_types[u] == "Crop"]))

    for i, cond in enumerate(conditions):
        pos[cond] = (0, i * y_gap)

    for i, crop in enumerate(crops):
        pos[crop] = (x_gap, i * y_gap)

    # Build edge and node traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        mode='lines'
    )

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append('lightblue' if node_types[node] == "Condition" else 'lightgreen')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_text,
        mode='markers+text',
        textposition="middle right",
        hoverinfo='text',
        marker=dict(color=node_color, size=20, line=dict(width=2))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Conditions ↔ Recommended Crops',
                        #titlefont_size=18,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ))

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(portfolio_df)

with tab6:
    # Parse Planting Start and End into datetime (dummy year for simplicity)
    calendar_df['Planting Start'] = pd.to_datetime(calendar_df['Planting Start'], format='%B %d', errors='coerce')
    calendar_df['Planting End'] = pd.to_datetime(calendar_df['Planting End'], format='%B %d', errors='coerce')

    # Estimate Harvest End
    calendar_df['Harvest End'] = calendar_df['Planting End'] + pd.to_timedelta(calendar_df['Harvest Days (Approx.)'], unit='D')
    calendar_df['Harvest Month'] = calendar_df['Harvest End'].dt.month

    # Clean rows with missing or invalid dates
    calendar_df = calendar_df.dropna(subset=['Planting Start', 'Harvest End'])

    # Convert to month numbers for plotting
    calendar_df['Planting Month'] = calendar_df['Planting Start'].dt.month

    # Sort for better display
    calendar_df_sorted = calendar_df.sort_values('Planting Month')

    # Plot
    fig = plt.figure(figsize=(12, 8))
    for i, row in calendar_df_sorted.iterrows():
        plt.plot([row['Planting Month'], row['Harvest Month']], [row['Crop']]*2, marker='o')

    plt.xticks(range(1, 13), calendar.month_abbr[1:])
    plt.title('Crop Planting to Harvest Calendar (Estimated)')
    plt.xlabel('Month')
    plt.ylabel('Crop')
    st.pyplot(fig)

    st.dataframe(calendar_df[["Crop", "Planting Start", "Planting End", "Harvest Days (Approx.)"]])
