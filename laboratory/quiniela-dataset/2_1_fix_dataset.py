import csv

# Open the original csv file in read mode
with open("datasets/dataset.csv", "r") as infile:
    # Open the fixed csv file in write mode
    with open("datasets/fixed.csv", "w") as outfile:
        # Create a csv reader and writer
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Initialize the previous date
        previous_date = ""

        # Iterate over the rows in the original csv file
        for row in reader:
            # If the date is missing, use the previous date
            if row[3] == "":
                row[3] = previous_date
            else:
                # Update the previous date
                previous_date = row[3]

            # Write the fixed row to the output file
            writer.writerow(row)

# Manually replacement ,020 to ,201. And ,02 to ,20 in datasets/fixed2.csv
