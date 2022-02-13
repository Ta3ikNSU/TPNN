from matplotlib import pyplot as plt

import csv
import numpy as np


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def feelNaN(input_array: []):
    arr = [input_array[0], input_array[1]]
    for el in range(2, input_array.__len__()):
        if isfloat(np.array(input_array)[el]):
            arr.append(float(np.array(input_array)[el]))
        else:
            arr.append('NaN')
    return arr


if __name__ == '__main__':

    input = open('ID_data_mass_18122012_test.csv')
    csvreader = csv.reader(input)
    rows = np.array([])

    table = []
    for row in csvreader:
        table.append(row)
    input.close()
    output = open('out.csv', 'w')
    mywriter = csv.writer(output, delimiter=',')
    mywriter.writerow(table[0])
    mywriter.writerow(table[1])
    mywriter.writerow(table[2])
for index in range(3, table.__len__()):
    string = table[index]
    length = string.__len__()
    if isfloat(string[length - 3]) and (isfloat(string[length - 2]) or isfloat(string[length - 1])):
        mywriter.writerow(feelNaN(table[index]))
output.close()


# теплограмма
countRows = 0
length = 0
r = 0
for row in table:
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
plt.imshow(matrix, cmap='hot', interpolation='nearest')
plt.show()
print(matrix)

