
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from urllib.parse import quote_plus

st.set_page_config(
    page_title="IPL StratX",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL StratX")
st.caption("Integrated Data Warehouse & Data Mining Framework")
st.markdown("### Player Performance & Match Intelligence")


@st.cache_resource
def connect_atlas():
    username = quote_plus("jamunajai02_db_user")
    password = os.getenv("MONGODB_PASSWORD")

    if not password:
        raise ValueError("MONGODB_PASSWORD is not set.")

    uri = (
        f"mongodb+srv://{username}:{quote_plus(password)}"
        "@ipl.1smt2su.mongodb.net/"
        "?retryWrites=true&w=majority"
    )

    client = MongoClient(uri, serverSelectionTimeoutMS=20000)
    client.admin.command("ping")
    return client


@st.cache_data(ttl=60)
def load_data():
    client = connect_atlas()
    db = client["IPL_StratX"]

    # Load all ball-by-ball records from Atlas
    records = list(db["ball_by_ball"].find({}, {"_id": 0}))
    return pd.DataFrame(records)


try:
    with st.spinner("Loading IPL data from MongoDB Atlas..."):
        data = load_data()

    if data.empty:
        st.error("No records found in ball_by_ball.")
        st.stop()

    st.success("Connected to MongoDB Atlas")

    # Ensure numeric columns are usable for calculations
    for column in [
        "total_runs", "runs_off_bat", "is_wicket",
        "is_boundary", "team_score", "season"
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    # Sidebar filters
    st.sidebar.header("Dashboard Filters")

    filtered = data.copy()

    if "season" in filtered.columns:
        seasons = sorted(filtered["season"].dropna().unique())
        selected_seasons = st.sidebar.multiselect(
            "Select Season",
            seasons,
            default=seasons
        )
        if selected_seasons:
            filtered = filtered[
                filtered["season"].isin(selected_seasons)
            ]

    if "batting_team" in filtered.columns:
        teams = sorted(filtered["batting_team"].dropna().unique())
        selected_team = st.sidebar.selectbox(
            "Batting Team",
            ["All Teams"] + teams
        )
        if selected_team != "All Teams":
            filtered = filtered[
                filtered["batting_team"] == selected_team
            ]

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Ball Records", f"{len(filtered):,}")

    total_runs = (
        filtered["total_runs"].sum()
        if "total_runs" in filtered.columns else 0
    )
    c2.metric("Total Runs", f"{total_runs:,.0f}")

    wickets = (
        filtered["is_wicket"].sum()
        if "is_wicket" in filtered.columns else 0
    )
    c3.metric("Wickets", f"{wickets:,.0f}")

    matches = (
        filtered["match_id"].nunique()
        if "match_id" in filtered.columns else 0
    )
    c4.metric("Matches", f"{matches:,}")

    st.divider()

    # Team analysis
    left, right = st.columns(2)

    with left:
        st.subheader("Runs by Batting Team")
        if {"batting_team", "total_runs"}.issubset(filtered.columns):
            team_runs = (
                filtered.groupby("batting_team", as_index=False)
                ["total_runs"].sum()
                .sort_values("total_runs", ascending=False)
            )
            fig = px.bar(
                team_runs,
                x="batting_team",
                y="total_runs",
                color="batting_team",
                labels={
                    "batting_team": "Team",
                    "total_runs": "Runs"
                }
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Runs by Season")
        if {"season", "total_runs"}.issubset(filtered.columns):
            season_runs = (
                filtered.groupby("season", as_index=False)
                ["total_runs"].sum()
                .sort_values("season")
            )
            fig = px.line(
                season_runs,
                x="season",
                y="total_runs",
                markers=True,
                labels={
                    "season": "Season",
                    "total_runs": "Runs"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

    # Player performance
    st.subheader("Top 10 Run Scorers")

    if {"striker", "runs_off_bat"}.issubset(filtered.columns):
        players = (
            filtered.groupby("striker", as_index=False)
            ["runs_off_bat"].sum()
            .sort_values("runs_off_bat", ascending=False)
            .head(10)
        )

        fig = px.bar(
            players,
            x="runs_off_bat",
            y="striker",
            orientation="h",
            labels={
                "striker": "Player",
                "runs_off_bat": "Runs"
            }
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Match data explorer
    st.subheader("IPL Data Explorer")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.caption(
        f"Displaying data from MongoDB Atlas | "
        f"{len(filtered):,} records after filtering"
    )

except Exception as e:
    st.error(f"Dashboard error: {e}")
    st.info(
        "Check your Atlas credentials and collection data."
    )
