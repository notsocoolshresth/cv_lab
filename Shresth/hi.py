import csv

input_csv = "C:\\Users\\Shresth\\Downloads\\submission (5).csv"
output_csv = "output111.csv"

with open(input_csv, newline="", encoding="utf-8") as infile, \
     open(output_csv, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        row[0] = row[0] + ".tif"
        writer.writerow(row)
