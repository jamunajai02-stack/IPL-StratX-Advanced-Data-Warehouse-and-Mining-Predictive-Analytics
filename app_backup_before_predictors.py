
import os
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
import plotly.express as px
from pymongo import MongoClient

st.set_page_config(
    page_title="IPL StratX",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL StratX")
st.caption("Integrated Data Warehouse & Data Mining Framework")
st.markdown("### Player Performance & Match Intelligence")


# --------------------------------------------------
# 1. MONGODB ATLAS CONNECTION
# --------------------------------------------------

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


@st.cache_data(ttl=300)
def load_collection(collection_name):
    client = connect_atlas()
    db = client["IPL_StratX"]

    records = list(
        db[collection_name].find({}, {"_id": 0})
    )

    return pd.DataFrame(records)


def numeric_column(df, column):
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )


def find_column(df, possible_names):
    """Find a column without depending on capitalization."""
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def show_table(df, height=350):
    if df is not None and not df.empty:
        st.dataframe(
            df,
            use_container_width=True,
            height=height
        )
    else:
        st.info("No data is available for this section yet.")


# --------------------------------------------------
# 2. LOAD ATLAS DATA
# --------------------------------------------------

try:
    with st.spinner("Connecting to MongoDB Atlas..."):
        data = load_collection("ball_by_ball")

    if data.empty:
        st.error("No records found in the ball_by_ball collection.")
        st.stop()

    st.success("Connected to MongoDB Atlas")

except Exception as error:
    st.error(f"Atlas connection error: {error}")
    st.info(
        "Check MONGODB_PASSWORD and your MongoDB Atlas connection."
    )
    st.stop()


# --------------------------------------------------
# 3. PREPARE DATA
# --------------------------------------------------

for column in [
    "season",
    "total_runs",
    "runs_off_bat",
    "extras",
    "is_wicket",
    "is_boundary",
    "legal_delivery",
    "innings",
    "match_id",
]:
    numeric_column(data, column)

for column in ["striker", "bowler", "batting_team", "bowling_team"]:
    if column in data.columns:
        data[column] = data[column].fillna("Unknown").astype(str)

# Remove rows without a usable season from season filtering.
if "season" in data.columns:
    data["season"] = data["season"].astype("Int64")


# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

filtered = data.copy()

if "season" in filtered.columns:
    seasons = sorted(
        filtered["season"].dropna().astype(int).unique().tolist()
    )

    selected_seasons = st.sidebar.multiselect(
        "Select Season",
        options=seasons,
        default=seasons
    )

    if selected_seasons:
        filtered = filtered[
            filtered["season"].isin(selected_seasons)
        ]
    else:
        filtered = filtered.iloc[0:0]

if "batting_team" in filtered.columns:
    teams = sorted(
        filtered["batting_team"].dropna().unique().tolist()
    )

    selected_team = st.sidebar.selectbox(
        "Batting Team",
        ["All Teams"] + teams
    )

    if selected_team != "All Teams":
        filtered = filtered[
            filtered["batting_team"] == selected_team
        ]

st.sidebar.caption(
    f"{len(filtered):,} ball records selected"
)


# --------------------------------------------------
# 5. DASHBOARD TABS
# --------------------------------------------------

overview_tab, player_tab, ml_tab, mining_tab, olap_tab = st.tabs(
    [
        "📊 Overview",
        "🏏 Player Analytics",
        "🤖 ML Insights",
        "🔎 Data Mining",
        "📈 OLAP Analytics",
    ]
)


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with overview_tab:
    st.subheader("IPL Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Ball Records", f"{len(filtered):,}")

    total_runs = (
        filtered["total_runs"].sum()
        if "total_runs" in filtered.columns else 0
    )
    c2.metric("Total Runs", f"{total_runs:,.0f}")

    dismissals = (
        filtered["is_wicket"].sum()
        if "is_wicket" in filtered.columns else 0
    )
    c3.metric("Dismissal Events", f"{dismissals:,.0f}")

    matches = (
        filtered["match_id"].nunique()
        if "match_id" in filtered.columns else 0
    )
    c4.metric("Matches", f"{matches:,}")

    st.caption(
        "Dismissal events are counted from is_wicket; "
        "they are not necessarily wickets credited to bowlers."
    )

    st.divider()

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
                title="Total Runs by Team",
                labels={
                    "batting_team": "Batting Team",
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
                title="Season-wise Runs",
                labels={
                    "season": "Season",
                    "total_runs": "Runs"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Data Explorer")
    show_table(filtered.head(100))

    st.caption(
        f"MongoDB Atlas | {len(filtered):,} records after filtering"
    )


# ==================================================
# TAB 2: PLAYER ANALYTICS AND CLUSTERING
# ==================================================

with player_tab:
    st.subheader("Player Performance Analytics")

    required = {"striker", "runs_off_bat"}

    if required.issubset(filtered.columns):
        player_stats = (
            filtered.groupby("striker")
            .agg(
                Runs=("runs_off_bat", "sum"),
                Matches=("match_id", "nunique")
                if "match_id" in filtered.columns
                else ("runs_off_bat", "count"),
            )
            .reset_index()
        )

        if "is_boundary" in filtered.columns:
            boundaries = (
                filtered.groupby("striker")["is_boundary"]
                .sum()
                .rename("Boundary Deliveries")
                .reset_index()
            )
            player_stats = player_stats.merge(
                boundaries, on="striker", how="left"
            )
        else:
            player_stats["Boundary Deliveries"] = 0

        if "legal_delivery" in filtered.columns:
            balls = (
                filtered[
                    (filtered["legal_delivery"] == 1)
                    & (filtered["runs_off_bat"].notna())
                ]
                .groupby("striker")
                .size()
                .rename("Balls")
                .reset_index()
            )
            player_stats = player_stats.merge(
                balls, on="striker", how="left"
            )
        else:
            player_stats["Balls"] = 0

        player_stats["Balls"] = player_stats["Balls"].fillna(0)
        player_stats["Strike Rate"] = (
            player_stats["Runs"]
            .div(player_stats["Balls"].replace(0, pd.NA))
            * 100
        )

        player_stats = player_stats.sort_values(
            "Runs", ascending=False
        )

        st.subheader("Top 10 Run Scorers")

        top_players = player_stats.head(10).sort_values("Runs")

        fig = px.bar(
            top_players,
            x="Runs",
            y="striker",
            orientation="h",
            color="Runs",
            title="Top Run Scorers",
            labels={"striker": "Player"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Compare Players")

        player_options = sorted(
            player_stats["striker"].dropna().unique().tolist()
        )

        selected_players = st.multiselect(
            "Choose players to compare",
            player_options,
            default=player_stats.head(3)["striker"].tolist()
        )

        comparison = player_stats[
            player_stats["striker"].isin(selected_players)
        ]

        show_table(
            comparison[
                [
                    "striker", "Runs", "Matches", "Balls",
                    "Strike Rate", "Boundary Deliveries"
                ]
            ].round(2)
        )

        # Player clustering based on observable batting statistics.
        st.divider()
        st.subheader("Player Clustering")

        st.write(
            "K-means groups players with similar runs, matches, "
            "strike rates, and boundary-delivery counts. "
            "Clusters are statistical groups, not official player rankings."
        )

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            cluster_data = player_stats[
                [
                    "Runs", "Matches", "Strike Rate",
                    "Boundary Deliveries"
                ]
            ].copy()

            cluster_data = cluster_data.replace(
                [float("inf"), -float("inf")], pd.NA
            ).fillna(0)

            if len(cluster_data) >= 3:
                scaled = StandardScaler().fit_transform(cluster_data)

                model = KMeans(
                    n_clusters=3,
                    random_state=42,
                    n_init=10
                )

                player_stats["Cluster"] = model.fit_predict(scaled)

                cluster_filter = st.selectbox(
                    "Select cluster",
                    ["All Clusters", 0, 1, 2],
                    key="player_cluster"
                )

                cluster_view = player_stats.copy()

                if cluster_filter != "All Clusters":
                    cluster_view = cluster_view[
                        cluster_view["Cluster"] == cluster_filter
                    ]

                fig = px.scatter(
                    cluster_view,
                    x="Runs",
                    y="Strike Rate",
                    color=cluster_view["Cluster"].astype(str),
                    hover_name="striker",
                    size="Matches",
                    title="Player Clusters: Runs vs Strike Rate",
                    labels={"color": "Cluster"}
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Cluster Members")
                show_table(
                    cluster_view[
                        [
                            "striker", "Runs", "Matches",
                            "Strike Rate", "Boundary Deliveries",
                            "Cluster"
                        ]
                    ].sort_values("Runs", ascending=False)
                )
            else:
                st.info("Not enough players to calculate clusters.")

        except Exception as error:
            st.warning(f"Player clustering could not run: {error}")

    else:
        st.info("Player statistics columns are missing in this dataset.")


# ==================================================
# TAB 3: MACHINE LEARNING INSIGHTS
# ==================================================

with ml_tab:
    st.subheader("Machine Learning Results")

    st.markdown("#### Match Outcome Classification")

    st.write(
        "The current Random Forest classifier estimates whether the "
        "first-innings team wins using first-innings features. "
        "These are retrospective evaluation results from the current "
        "offline model run, not a fresh evaluation during dashboard use."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy", "65.36%")
    m2.metric("Test Samples", "306")
    m3.metric("Model", "Random Forest")

    st.caption(
        "Evaluation used a 75/25 random split. Results may differ on "
        "future seasons or when the training data changes."
    )

    st.divider()
    st.markdown("#### Score Prediction Model")

    s1, s2, s3 = st.columns(3)
    s1.metric("Mean Absolute Error", "18.56 runs")
    s2.metric("Root Mean Squared Error", "25.37 runs")
    s3.metric("R² Score", "0.3452")

    st.write(
        "These metrics come from the current score-prediction model "
        "evaluation. Predictions are estimates and become more useful "
        "as more innings data is available."
    )

    st.divider()
    st.subheader("Feature Importance")

    try:
        importance = load_collection("feature_importance")

        if not importance.empty:
            feature_col = find_column(
                importance,
                ["feature", "Feature", "feature_name", "variable"]
            )
            value_col = find_column(
                importance,
                ["importance", "Importance", "importance_score"]
            )

            if feature_col and value_col:
                importance[value_col] = pd.to_numeric(
                    importance[value_col], errors="coerce"
                )
                importance = importance.dropna(
                    subset=[value_col]
                ).sort_values(value_col, ascending=False)

                fig = px.bar(
                    importance,
                    x=value_col,
                    y=feature_col,
                    orientation="h",
                    title="Features Used by the Classifier",
                    labels={
                        feature_col: "Feature",
                        value_col: "Importance"
                    }
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig, use_container_width=True)
                show_table(importance)
            else:
                st.warning(
                    "Feature importance data was found, but its column "
                    "names were not recognized."
                )
                show_table(importance)
        else:
            st.info("The feature_importance collection is empty.")

    except Exception as error:
        st.warning(f"Could not load feature importance: {error}")

    st.divider()
    st.subheader("Score Prediction Records")

    try:
        predictions = load_collection("score_predictions")

        if not predictions.empty:
            st.write(
                "Prediction records from Atlas. Column names depend "
                "on the saved model output."
            )

            actual_col = find_column(
                predictions,
                ["actual_final_score", "final_score", "actual_score"]
            )
            predicted_col = find_column(
                predictions,
                [
                    "predicted_final_score",
                    "predicted_score",
                    "prediction"
                ]
            )

            if actual_col and predicted_col:
                predictions[actual_col] = pd.to_numeric(
                    predictions[actual_col], errors="coerce"
                )
                predictions[predicted_col] = pd.to_numeric(
                    predictions[predicted_col], errors="coerce"
                )

                chart_data = predictions.dropna(
                    subset=[actual_col, predicted_col]
                )

                if not chart_data.empty:
                    fig = px.scatter(
                        chart_data.sample(
                            min(2000, len(chart_data)),
                            random_state=42
                        ),
                        x=actual_col,
                        y=predicted_col,
                        title="Actual vs Predicted Scores",
                        labels={
                            actual_col: "Actual Score",
                            predicted_col: "Predicted Score"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

            show_table(predictions.head(100))
        else:
            st.info("No score prediction records were found.")

    except Exception as error:
        st.warning(f"Could not load score predictions: {error}")


# ==================================================
# TAB 4: ASSOCIATION RULES AND FREQUENT ITEMSETS
# ==================================================

with mining_tab:
    st.subheader("Association Rule Mining")

    st.write(
        "Association rules identify patterns that occur together in "
        "historical match data. They show associations, not proof of "
        "cause and effect."
    )

    try:
        rules = load_collection("association_rules")

        if not rules.empty:
            st.markdown("#### Association Rules")

            confidence_col = find_column(
                rules, ["confidence", "Confidence"]
            )
            lift_col = find_column(rules, ["lift", "Lift"])
            support_col = find_column(rules, ["support", "Support"])

            rules_view = rules.copy()

            for col in [confidence_col, lift_col, support_col]:
                if col:
                    rules_view[col] = pd.to_numeric(
                        rules_view[col], errors="coerce"
                    )

            if lift_col:
                rules_view = rules_view.sort_values(
                    lift_col, ascending=False
                )

            if confidence_col:
                minimum_confidence = st.slider(
                    "Minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05
                )

                rules_view = rules_view[
                    rules_view[confidence_col] >= minimum_confidence
                ]

            show_table(rules_view.head(100))

            if lift_col and confidence_col and not rules_view.empty:
                antecedent_col = find_column(
                    rules_view,
                    ["antecedents", "antecedent", "lhs"]
                )

                consequent_col = find_column(
                    rules_view,
                    ["consequents", "consequent", "rhs"]
                )

                if antecedent_col and consequent_col:
                    rules_view["Rule"] = (
                        rules_view[antecedent_col].astype(str)
                        + " → "
                        + rules_view[consequent_col].astype(str)
                    )

                    fig = px.scatter(
                        rules_view,
                        x=confidence_col,
                        y=lift_col,
                        hover_name="Rule",
                        size=support_col if support_col else None,
                        title="Association Rules: Confidence vs Lift",
                        labels={
                            confidence_col: "Confidence",
                            lift_col: "Lift"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("The association_rules collection is empty.")

    except Exception as error:
        st.warning(f"Could not load association rules: {error}")

    st.divider()
    st.subheader("Frequent Itemsets")

    try:
        itemsets = load_collection("frequent_itemsets")

        if not itemsets.empty:
            show_table(itemsets.head(100))
        else:
            st.info("No frequent itemsets were found.")

    except Exception as error:
        st.warning(f"Could not load frequent itemsets: {error}")


# ==================================================
# TAB 5: OLAP ANALYTICS
# ==================================================

with olap_tab:
    st.subheader("OLAP Analytics")

    st.write(
        "Explore historical runs through roll-up, drill-down, slice, "
        "and dice operations. These summaries are calculated from the "
        "currently filtered ball-by-ball data."
    )

    if {"batting_team", "season", "total_runs"}.issubset(filtered.columns):
        olap_left, olap_right = st.columns(2)

        with olap_left:
            st.markdown("#### Roll-up: Runs by Team")

            rollup = (
                filtered.groupby("batting_team", as_index=False)
                ["total_runs"].sum()
                .sort_values("total_runs", ascending=False)
            )

            fig = px.bar(
                rollup,
                x="batting_team",
                y="total_runs",
                title="Team-level Aggregation",
                labels={
                    "batting_team": "Team",
                    "total_runs": "Runs"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            show_table(rollup)

        with olap_right:
            st.markdown("#### Roll-up: Runs by Season")

            season_rollup = (
                filtered.groupby("season", as_index=False)
                ["total_runs"].sum()
                .sort_values("season")
            )

            fig = px.line(
                season_rollup,
                x="season",
                y="total_runs",
                markers=True,
                title="Season-level Aggregation"
            )
            st.plotly_chart(fig, use_container_width=True)
            show_table(season_rollup)

        st.divider()
        st.markdown("#### Dice: Team by Season")

        pivot = pd.pivot_table(
            filtered,
            index="batting_team",
            columns="season",
            values="total_runs",
            aggfunc="sum",
            fill_value=0
        )

        show_table(pivot.reset_index(), height=450)

        st.download_button(
            "Download Team-Season OLAP Table",
            data=pivot.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="IPL_StratX_team_season_olap.csv",
            mime="text/csv"
        )

        st.divider()
        st.markdown("#### Slice: Inspect One Season")

        available_seasons = sorted(
            filtered["season"].dropna().astype(int).unique().tolist()
        )

        if available_seasons:
            slice_season = st.selectbox(
                "Choose season for slice",
                available_seasons,
                key="olap_slice_season"
            )

            sliced = filtered[
                filtered["season"] == slice_season
            ]

            slice_summary = (
                sliced.groupby("batting_team", as_index=False)
                ["total_runs"].sum()
                .sort_values("total_runs", ascending=False)
            )

            st.write(f"Season selected: {slice_season}")
            show_table(slice_summary)

            fig = px.bar(
                slice_summary,
                x="batting_team",
                y="total_runs",
                title=f"Team Runs in {slice_season}"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "OLAP requires batting_team, season, and total_runs columns."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "IPL StratX | MongoDB Atlas | Data Mining | Machine Learning | OLAP"
)
