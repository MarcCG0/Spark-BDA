
import os

data_dir = '../data'
entries = os.listdir(data_dir)

file_names = []
for file_name in entries:
    if file_name:  # Check if the file name is not empty
        file_names.append(file_name)

days = []
aircraftids = []

for f in file_names: 
    f_splitted = f.split("-")
    days.append(f_splitted[0])
    aircraftids.append(f_splitted[4] + "-" + f_splitted[5][:-4])


print(days)
print(aircraftids)
