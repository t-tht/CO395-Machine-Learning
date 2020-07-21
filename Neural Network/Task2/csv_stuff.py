import csv
import json
import operator

def get_permutation_csv(filename):
    f = open(filename)
    csv_f = csv.reader(f)

    permu = []

    for item in csv_f:
        train_loss = item[0]
        train_metric = item[1]
        test_loss = item[2]
        test_metric = item[3]
        temp_param = item[4]
        param = json.loads(temp_param)

        perm = {
            'train_loss': train_loss,
            'train_metric': train_metric,
            'test_loss': test_loss,
            'test_metric': test_metric,
            'param': param
        }
        permu.append(perm)

    return permu

def write_csv(filename, row):
    with open(filename , 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

def sort_csv(filename, row_no = 3):
    reader = csv.reader(open(filename))
    # sortedlist = sorted(reader, key=operator.itemgetter(3))
    sortedlist = sorted(reader, key=lambda row : row[3])

    with open(filename, "w") as csvFile:
        writer = csv.writer(csvFile)
        for row in sortedlist:
            writer.writerow(row)

    csvFile.close()

def get_model(row, filename = 'result/round_2.csv'):
    with open(filename, 'r') as csvFile:
        csv_f = csv.reader(csvFile)
        item = csv_f[row]
        val_average = item[3]
        params = json.loads(item[4])
        model = json.loads(item[5])
        weight = json.loads(item[6])

    return [val_average, params, model, weight]
