# -*- coding: utf-8 -*-
"""
Code to calculate stat-RISMED score and assign priority to each case
Factors considered:
1. MedDRA-LLT severity (obtained from CISTERM) -- high vs non-high
2. Overseas action (obtained from BERT classifier) -- Yes/Unknown vs No
3. Broad therapeutic indication (rule-based keyword matching for products) -- Lifesaving vs Therapeutic/Vaccine/Symptomatic/Others

Expected output:
1. Numerical calculation based on odd ratios from above factors
2. Priority (High/Low) based on numerical score

Rollout strategy:
1. Current consolidation steps: Crawl all websites > Consolidated daily report > Substandard med classifier > LLT classifier
2. To add in an additional step for Overseas classifier + Indication keyword matching > "live_overseas_ind_prediction.py"
3. stat-RISMED score and priority to be calculated on final file
4. Hide all intermediate calculation columns and only show stat-RISMED priority (along with webpage crawled columns)
5. Randomisation: 50% of daily reports will be given random priority status (distribution of high-low depending on actual distribution)
6. To trial for 1 month (15 days actual 15 days randomised priority), with possible extension
"""
import random
import numpy as np
import pandas as pd
import os
from datetime import date, timedelta

# coefficients from multivariable logistic regression model, to update where necessary
or_meddra = 2.668993
or_overseas = 0.8223855
or_indication = 1.460011
intercept = -2.586555

high_severity_llt = ["Contamination with body fluid",
                     "Contamination with glass and/or metal particle",
                     "Contamination with microbes",
                     "Product mix up",
                     "Product inner label issue impacting strength, dose and/or safety",
                     "Product outer label issue impacting strength, dose and/or safety",
                     "Product counterfeit",
                     "Product formulation issue"]

# trial for stat-RISMED priority to be displayed in webcrawler report
# trial period: 2 - 31 Mar 2024 (inclusive, total 30 days)
# additional trial period: 1 - 30 Apr 2024 (inclusive, total 30 days)
# 50% (15 days) given actual model priority, remaining given random priority based on pre-defined distribution
"""Comment out below code if not using"""
# lower = 2
# upper = 31
# percent = 0.5
# num_samples = int(percent * (upper - lower + 1))
# random_dates = random.sample(range(lower, upper + 1), num_samples)
# print(sorted(random_dates)) #[2, 4, 5, 6, 8, 9, 10, 11, 14, 17, 20, 21, 22, 25, 30]
""""""

# GLOBAL VARIABLES
#random_dates = [2, 3, 4, 6, 8, 9, 10, 11, 14, 19, 20, 21, 22, 26, 28]  # for March: manipulated slightly to group Sat/Sun/Mon tgt as one entity
random_dates = [2, 3, 4, 6, 7, 8, 10, 11, 16, 19, 20, 21, 22, 26, 30]  # for April: manipulated slightly to group Sat/Sun/Mon tgt as one entity
date_today = date.today() #- timedelta(days=2)


def randomiser(current_day):
    return True if current_day in random_dates else False


def check_value(var, check_list, value_if_true, value_if_false):
    return value_if_true if var in check_list else value_if_false


def main():

    """Function to find the daily consolidated report"""
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

    csv_path = "./daily_reports"
    csv_filename = str(date_today.year) + "_" + str('{:02d}'.format(date_today.month)) + "_" + str(
        '{:02d}'.format(date_today.day)) + "_consolidated_report.csv"
    found = False

    for root, dirs, files in os.walk(csv_path):
        if csv_filename in files:
            csv_path = os.path.join(root, csv_filename)
            found = True

    if not found:
        print(f"No matching file found for {csv_filename}. stat-RISMED calculation not carried out.")

    """Load dataset and run stat-RISMED calculations"""
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        print(df.shape)

        """Perform calculations for every row in df"""
        df['Probability'] = np.nan  # empty numerical col
        df['Priority'] = None  # empty string col

        if randomiser(date_today.day):  # if today's date falls within the randomly selected TRUE dates
            print("Actual algo day")
            for index in range(len(df)):
                row = df.iloc[index]

                # get meddra score
                meddra_llt = row['LLT Step 2 prediction']
                meddra_score = check_value(meddra_llt, high_severity_llt, or_meddra, 0)

                # get overseas score
                overseas = row['Overseas']
                overseas_score = check_value(overseas, ["Yes"], or_overseas, 0)

                # get indication score
                indication = row['Indication']
                indication_score = check_value(indication, ["Lifesaving"], or_indication, 0)

                # calculate probability of outcome i.e. high impact, given predictors; and estimate priority
                if row['Label'] == "Substandard":
                    linear_combi = intercept + meddra_score + overseas_score + indication_score
                else:  # give 0 for meddra_score if non-substandard medicine
                    linear_combi = intercept + 0 + overseas_score + indication_score
                odds = np.exp(linear_combi)
                impact_probability = odds / (1 + odds)
                impact_probability = np.round(impact_probability, 4)

                threshold = 0.2  # adjust as needed
                priority = "HIGH" if impact_probability > threshold else "LOW"

                # add calculations to df
                df.at[index, 'Probability'] = impact_probability
                df.at[index, 'Priority'] = priority

        else:  # if not TRUE date, then randomly assign "HIGH" and "LOW" priority, and random probability to every row
            # Randomly assign 'Priority' column with "HIGH" and "LOW"
            print("Random algo day")
            proportion_high = random.uniform(0.05, 0.1)  # assume proportion of highs between 5-10% (can be adjusted)
            df['Priority'] = np.random.choice(['HIGH', 'LOW'], size=len(df), p=[proportion_high, 1-proportion_high])

            # Randomly assign 'Probability' score based on 'Priority'
            df['Probability'] = np.where(df['Priority'] == 'HIGH', np.round(np.random.uniform(0.2, 1.0, len(df)), 4),
                                         np.round(np.random.uniform(0.0, 0.2, len(df)), 4) # to match threshold for TRUE cases
                                         )

        consolidated_df = df[["Agency", "Country", "Webpage", "Date", "Title", "Issue / Background (combined)",
                              'Label', 'Priority', 'Probability',
                              "Url", "Products affected", "Affected Company Name", "Date scraped"]]
        consolidated_df.insert(6, 'Officer', "")
        consolidated_df.insert(7, 'Case creation', "")
        consolidated_df.insert(8, 'Related cases', "")

        """Overwrite current daily consolidated report with new stat-RISMED columns"""
        if not consolidated_df.empty:
            consolidated_df.to_csv(csv_path)
        else:
            print("No alerts for stat-RISMED calculation today!!!")
            consolidated_df = pd.DataFrame(columns=["Agency", "Country", "Webpage", "Date", "Title", "Issue / Background (combined)",
                                                    'Officer', 'Case creation', 'Related cases',
                                                    'Label', 'Priority', 'Probability',  # remove all intermediate calculations and only present final stat-RISMED priority to officer
                                                    "Url", "Products affected", "Affected Company Name", "Date scraped"])
            consolidated_df.to_csv(csv_path, index=False)

        return consolidated_df.shape[0]

    else:
        return "Failed to perform stat-RISMED calculation. Please check daily consolidated report."


if __name__ == '__main__':
    main()






