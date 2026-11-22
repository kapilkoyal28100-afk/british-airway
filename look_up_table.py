
import pandas as pd
import numpy as np


file_path = "British Airways Summer Schedule Dataset - Forage Data Science Task 1.xlsx"

df = pd.read_excel(file_path)


df["TOTAL_SEATS"] = (
    df["FIRST_CLASS_SEATS"]
    + df["BUSINESS_CLASS_SEATS"]
    + df["ECONOMY_SEATS"]
)

df["Flight group"] = (
    df["HAUL"] + " – " + df["TIME_OF_DAY"]
)

grouped = df.groupby("Flight group").agg({
    "TOTAL_SEATS": "sum",
    "TIER1_ELIGIBLE_PAX": "sum",
    "TIER2_ELIGIBLE_PAX": "sum",
    "TIER3_ELIGIBLE_PAX": "sum"
})

grouped["Tier 1"] = (
    grouped["TIER1_ELIGIBLE_PAX"] /
    grouped["TOTAL_SEATS"]
)

grouped["Tier 2"] = (
    grouped["TIER2_ELIGIBLE_PAX"] /
    grouped["TOTAL_SEATS"]
)

grouped["Tier 3"] = (
    grouped["TIER3_ELIGIBLE_PAX"] /
    grouped["TOTAL_SEATS"]
)

result = grouped[["Tier 1", "Tier 2", "Tier 3"]].reset_index()

result["Justification"] = result.apply(
    lambda row:
    "Estimated using total eligible passengers "
    "divided by total seat capacity for this "
    "flight group. Tier 1 represents the smallest "
    "eligible segment, followed by Tier 2 and Tier 3.",
    axis=1
)

output_file = "Lounge_Eligibility_Lookup_Completed.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    result.to_excel(
        writer,
        sheet_name="Eligibility Lookup",
        index=False
    )

print(result.to_string(index=False))

print("\nExcel file created successfully!")
print(output_file)
