import pandas as pd

def load_data():
    # Load both datasets
    dataset1 = pd.read_csv("data/test_dataset1.csv")
    dataset2 = pd.read_csv("data/test_dataset2.csv")
    return dataset1, dataset2
    
def filter_data(dataset1, dataset2, period, sites, pathways, boroughs):
    # Filter Dataset 1
    if period:
        start_date, end_date = period
        dataset1 = dataset1[(dataset1['date'] >= str(start_date)) & (dataset1['date'] <= str(end_date))]

    if sites:
        dataset1 = dataset1[dataset1['site'].isin(sites)]

    # Filter Dataset 2
    if pathways:
        dataset2 = dataset2[dataset2['pathway'].isin(pathways)]

    if boroughs:
        dataset2 = dataset2[dataset2['borough'].isin(boroughs)]

    return dataset1, dataset2

def calculate_metrics(dataset1, dataset2):
    avg_los_7 = dataset1['patients LoS 7+ days'].mean()
    avg_discharge_passport = dataset2['discharge passport completion %'].mean()
    return {"avg_los_7": avg_los_7, "avg_discharge_passport": avg_discharge_passport}

def compare_weeks(dataset1, dataset2, week1, week2):
    week1_data = dataset1[pd.to_datetime(dataset1['date']).dt.to_period('W') == week1]
    week2_data = dataset1[pd.to_datetime(dataset1['date']).dt.to_period('W') == week2]
    
    comparison = {
        "Metric": ["Admissions", "Discharges", "Boarded Beds"],
        "Week 1": [
            week1_data['admissions'].sum(),
            week1_data['discharges'].sum(),
            week1_data['boarded beds'].sum(),
        ],
        "Week 2": [
            week2_data['admissions'].sum(),
            week2_data['discharges'].sum(),
            week2_data['boarded beds'].sum(),
        ],
    }
    return pd.DataFrame(comparison)

