from csv import reader

gershman_data1_file = 'human_data/data1.csv'
gershman_data2_file = 'human_data/data2.csv'
stojic_data_file = 'human_data/exp1_banditData.csv'

## Reading csvs:
def file_to_rows(file):
    with open(file) as f:
        lines = reader(f)
        rows = []
        for r in lines:
            rows.append(r)
    return rows