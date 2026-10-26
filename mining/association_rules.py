
import os
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

os.makedirs("outputs", exist_ok=True)

# Load classification data prepared in Step 2
data = pd.read_csv("outputs/IPL_StratX_classification_data.csv")

# Create descriptive categories using the dataset's median
data["High_Score"] = (
    data["first_runs"] >= data["first_runs"].median()
)

data["High_Run_Rate"] = (
    data["first_run_rate"] >= data["first_run_rate"].median()
)

data["Many_Boundaries"] = (
    data["first_boundaries"] >= data["first_boundaries"].median()
)

data["Many_Wickets"] = (
    data["first_wickets"] >= data["first_wickets"].median()
)

data["High_Extras"] = (
    data["first_extras"] >= data["first_extras"].median()
)

data["First_Team_Win"] = data["first_team_wins"] == 1
data["First_Team_Loss"] = data["first_team_wins"] == 0

# Boolean transaction data
items = [
    "High_Score",
    "High_Run_Rate",
    "Many_Boundaries",
    "Many_Wickets",
    "High_Extras",
    "First_Team_Win",
    "First_Team_Loss"
]

transactions = data[items].astype(bool)

# Discover frequent itemsets
frequent_items = apriori(
    transactions,
    min_support=0.15,
    use_colnames=True
)

print("Frequent Itemsets:")
print(frequent_items.sort_values(
    "support", ascending=False
).to_string(index=False))

# Generate association rules
if len(frequent_items) >= 2:
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.5
    )

    if not rules.empty:
        rules = rules.sort_values(
            ["lift", "confidence"],
            ascending=False
        )

        print("\nAssociation Rules:")
        print(rules[
            ["antecedents", "consequents", "support",
             "confidence", "lift"]
        ].head(20).to_string(index=False))

        rules.to_csv(
            "outputs/IPL_StratX_association_rules.csv",
            index=False
        )
        print("\nRules saved successfully.")
    else:
        print("\nNo rules met the confidence threshold.")
else:
    print("\nNo frequent itemsets found. Try lowering min_support.")
    
frequent_items.to_csv(
    "outputs/IPL_StratX_frequent_itemsets.csv",
    index=False
)
