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
