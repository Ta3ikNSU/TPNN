def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    import csv

    file = open('ID_data_mass_18122012_test.csv')
    type(file)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    next(csvreader)
    next(csvreader)
    for row in csvreader:
        length = row.__len__()
        if isfloat(row[length - 3]) and (isfloat(row[length - 2]) or isfloat(row[length - 1])):
            rows.append([row[length - 3], row[length - 2], row[length - 1] * 1000])
    print(rows)
    # get good values
