import csv
import os
import argparse
import pickle

class Fanfic:
    def __init__(self, work_id, title, body):
        self.title = title
        self.body = body
        self.work_id = work_id

def main():
    csv.field_size_limit(1000000000)  # up the field size because stories are long

    parser = argparse.ArgumentParser(description='convert fic text in a csv into txt files')
    parser.add_argument(
    'csv', metavar='csv',
    help='the name of the csv with the original data')

    args = parser.parse_args()
    csv_name = args.csv

    fics = []

    # clean extension
    if ".csv" not in csv_name:
        csv_name = csv_name + ".csv"

    with open(csv_name, 'r') as csvfile:
        rd = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(rd)  # skip the header row
        for row in rd:
            fics.append(Fanfic(row[0], row[1], row[-1]))
        pickle.dump(fics, open("./fics.pkl", 'wb'))


   
#    with open(folder_name + "/" + row[0] + ".txt", "w") as text_file:
#     text_file.write(row[-1])

if __name__ == "__main__":
    main()