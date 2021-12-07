
#move to new file
if __name__ == "__main__":
    # stars = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/stdata_2021-11-29.p")
    # print(stars)
    # planets = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/plandata_2021-11-29.p")
    # print(planets)
    # orbitfits = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/orbfits_2021-11-29.p")
    # orbits = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/orbdata_2021-11-29.p")
    # pdfs = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/comps_2021-11-29.p")

    comp_data = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/comps_data_2021-11-29.p")
    col_names = comp_data.columns.values.tolist()
    scenario_names = []
    for x in col_names:
        if x[:8] == 'complete':
            scenario_names.append(x[13:])

    print(scenario_names)
    #drop contr_curve col
    completeness = pd.DataFrame([], columns = ['pl_id', 'contr_curve', 'completeness',  'scenario_name'])
    for i, row in tqdm(comp_data.iterrows()):
        # print(type(row))
        newRows = []
        for scenario_name in scenario_names:
            if pd.notna(row[('completeness_' + scenario_name)]):
                newRows.append([row['pl_id'], row[('contr_curve_' + scenario_name)], row['completeness_' + scenario_name], scenario_name])
                singleRow = pd.DataFrame(newRows, columns = ['pl_id', 'contr_curve', 'completeness', 'scenario_name'])
                completeness = completeness.append(singleRow, ignore_index=True)   
    print(completeness)

    # scenarios = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache")
    # contrastCurves = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/contr_data_2021-11-29.p")
    # print(contrastCurves)
