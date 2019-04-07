import json, csv

csvfile = open('dataset/final_dataset.csv', 'r')
jsonfile = open('final_dataset.json', 'w')

# fieldnames = ("user_id","user_name","prod_id","age","reviews","rating")
reader = csv.DictReader( csvfile)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')