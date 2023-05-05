import numpy as np


def parse_row(line, len_row):
    # line = line.replace('{', '').replace('}', '')
    line = line.replace('\n', '')
    row = np.zeros(len_row)
    for idx, data in enumerate(line.split(',')):
        row[idx] = float(data)
    return row


def read_data_arff(filename):
    # Step 1. Read data by row.
    with open(filename, 'r') as fp:
        file_content = fp.readlines()

    # Step 2. Get the columns.
    columns = []
    len_attr = len('@attribute')

    for line in file_content:
        if line.startswith('@attribute '):
            col_name = line[len_attr:].split()[0]
            columns.append(col_name)

    # Step 3. Get the rows.
    rows = None
    len_row = len(columns)
    flag = False
    for line in file_content:
        if line.startswith('@data'):
            flag = True
            continue
        if flag:
            row = parse_row(line, len_row)
            if rows is None:
                rows = row
            else:
                rows = np.vstack((rows, row))

    # Step 4. Return the results.
    # df = pd.DataFrame(data=rows, columns=columns)
    return rows


if __name__ == '__main__':
    filename = "../data/abalone.arff"
    rows = read_data_arff(filename)
