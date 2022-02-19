from matplotlib import pyplot as plt, pyplot

import csv
import numpy as np
import codecs
import seaborn as sns
from matplotlib.pyplot import subplot


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

    input = codecs.open("ID_data_mass_18122012_test.csv", "r", encoding="utf-8")
    csvreader = csv.reader(input)

    output = codecs.open("out.csv", "w")
    mywriter = csv.writer(output, delimiter=',')

    rows = np.array([])

    table = []
    count = 0
    for row in csvreader:
        table.append(row)
        if count < 3:
            mywriter.writerow(row)
            count += 1
        else:
            string = row
            length = string.__len__()
            if isfloat(string[length - 3]) or isfloat(string[length - 2]) or isfloat(string[length - 1]):
                mywriter.writerow(feelNaN(string))
    output.close()
    temp = table[1].copy()
    temp.pop(0)
    temp.pop(0)
    x_axis_labels = np.array(temp)
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
            r = np.abs(np.corrcoef(left, right)[0][1])
        except:
            r = 0
        matrix[i][j] = r
        matrix[j][i] = r
# plt.imshow(matrix, cmap='hot', interpolation='nearest')
# plt.show()
# plt.set
sns.set(rc={'figure.figsize': (20, 20)})
sns.heatmap(matrix, annot=True, fmt=".1g", vmin=0, vmax=1, center=0, cmap='coolwarm', square=True,
            xticklabels=x_axis_labels,
            yticklabels=x_axis_labels)
plt.savefig('saving-a-seaborn-plot-as-png-file-transparent.png', transparent=True)

plt.show()
# print(matrix)
