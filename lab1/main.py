from matplotlib import pyplot as plt


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    import csv
    import numpy as np

    file = open('ID_data_mass_18122012_test.csv')
    type(file)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = np.empty(0)
    next(csvreader)
    next(csvreader)
    # for row in csvreader:
    #     length = row.__len__()
    #     if isfloat(row[length - 3]) and (isfloat(row[length - 2]) or isfloat(row[length - 1])):
    #         rows.append([row[length - 3], row[length - 2], row[length - 1] * 1000])
    # print(rows)
    countRows = 0
    length = 0
    r = 0
    for row in csvreader:
        row.pop(0)
        row.pop(0)
        rows = np.append(rows, np.array(row))
        length = len(row)
        countRows += 1
    matrix = np.zeros((32, 32))
    for i in range(0, length):
        for j in range(i, length):
            left = np.empty(0)
            right = np.empty(0)
            for index in range(0, countRows):
                if isfloat(rows[index * 32 + i]) and isfloat(rows[index * 32 + j]):
                    left = np.append(left, float(rows[index * 32 + i]))
                    right = np.append(right, float(rows[index * 32 + j]))
            try:
                r = np.corrcoef(left, right)[0][1]
            except:
                r = 0
            matrix[i][j] = r
            matrix[j][i] = r
    # a = np.random.random((16, 16))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()
