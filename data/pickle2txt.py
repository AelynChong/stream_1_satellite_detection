import pickle
import pandas as pd
import json
import pprint
import csv

path = "D:/Users/kirar/Documents/[Lux]_ISM/[ISM]_3rd_Semester/[3rd_Semester]_Computer Vision and Image Analysis/stream_1"

# Read .pickle file 
# -------------------------
obj = pickle.load(open(path + "/all_data.pickle", "rb"))

# Convert .pickle file to .txt file in .json format
# -------------------------
with open(path + "/pickle_all_data.txt", "a") as f:
    json.dump(obj, f, indent=2)


# Convert .csv file to .pickle file
# -------------------------    
data = []
data.append(list())
with open(path + "/train_cvia.csv", "r") as f:
    data_raw = csv.reader(f)
    first_row = True
    dict_classes = {
        "cheops": 0, 
        "debris": 0, 
        "double_star": 0, 
        "earth_observation_sat_1": 0, 
        "lisa_pathfinder": 0, 
        "proba_2": 0, 
        "proba_3_csc": 0, 
        "proba_3_ocs": 0, 
        "smart_1": 0, 
        "soho": 0, 
        "xmm_newton": 0,
        "bg": 0
    }
    for row in data_raw:
        if first_row:
            first_row = False
            continue

        data[0].append({
            "filepath": row[0] + row[1],
            "width": 1024,
            "height": 1024,
            "bboxes": [{
                "class": row[2],
                "x1": int(row[5]),
                "y1": int(row[4]),
                "x2": int(row[7]),
                "y2": int(row[6])
            }]
        })
        dict_classes[row[2]] += 1
    data.append(dict_classes)
    data.append({
        "cheops": 0,
        "debris": 1,
        "double_star": 2,
        "earth_observation_sat_1": 3,
        "lisa_pathfinder": 4,
        "proba_2": 5,
        "proba_3_csc": 6,
        "proba_3_ocs": 7,
        "smart_1": 8,
        "soho": 9,
        "xmm_newton": 10,
        "bg": 11
    })
        #print(data)
    pickle.dump(data, open(path + "/stream1_cvia.pickle", "wb"))

# Read new .pickle file in .txt file (Verify)
# -------------------------    
obj = pickle.load(open(path + "/stream1_cvia.pickle", "rb"))
with open(path + "/stream1_cvia.txt", "a") as f:
    json.dump(obj, f, indent=2)

