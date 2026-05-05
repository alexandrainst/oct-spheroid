import glob
import random

import pandas as pd

random.seed(10)

train_old = pd.read_csv("./data/train_split.csv")
val_old = pd.read_csv("./data/val_split.csv")
test_old = pd.read_csv("./data/test_split.csv")

all_data = glob.glob("./data/*/*/*.mat")
print(f"Length all train {len(train_old)}")
print(f"Length all val {len(val_old)}")
print(f"Length all test {len(test_old)} \n")

old = [list(train_old["file name"]) + list(val_old["file name"]) + list(test_old["file name"])]
old = [i for x in old for i in x]
new = [i for i in all_data if i not in old]

print(f"Length old {len(old)}")
print(f"Length new {len(new)} \n")

new_train = random.sample(new, int(0.8 * len(new)))
temp = [i for i in new if i not in new_train]
new_val = random.sample(temp, int(0.7 * (len(temp))))
new_test = [i for i in temp if i not in new_val]

new_train_l = [i.split("/")[3] for i in new_train]
new_val_l = [i.split("/")[3] for i in new_val]
new_test_l = [i.split("/")[3] for i in new_test]
print(f"Length new train {len(new_train)}")
print(f"Length new val {len(new_val)}")
print(f"Length new test {len(new_test)} \n")

new_train_rows = [{"file name": new_train[i], "labels": new_train_l[i]} for i in range(len(new_train))]
new_train_rows = pd.DataFrame(new_train_rows)
train_old = pd.concat([train_old, new_train_rows])

new_val_rows = [{"file name": new_val[i], "labels": new_val_l[i]} for i in range(len(new_val))]
new_val_rows = pd.DataFrame(new_val_rows)
val_old = pd.concat([val_old, new_val_rows])

new_test_rows = [{"file name": new_test[i], "labels": new_test_l[i]} for i in range(len(new_test))]
new_test_rows = pd.DataFrame(new_test_rows)
test_old = pd.concat([test_old, new_test_rows])

print(f"Length all train {len(train_old)}")
print(f"Length all val {len(val_old)}")
print(f"Length all test {len(test_old)} \n")

idx = []
for i in range(len(val_old)):
    fn = val_old["file name"].iloc[i]
    fn = fn.split("/")[-1].split("_")
    if len(fn) == 1:
        idx.append(i)
filtered_val_old = val_old.iloc[idx]

idx = []
for i in range(len(train_old)):
    fn = train_old["file name"].iloc[i]
    fn = fn.split("/")[-1].split("_")
    if len(fn) == 1:
        idx.append(i)
filtered_train_old = train_old.iloc[idx]

idx = []
for i in range(len(test_old)):
    fn = test_old["file name"].iloc[i]
    fn = fn.split("/")[-1].split("_")
    if len(fn) == 1:
        idx.append(i)
filtered_test_old = test_old.iloc[idx]

filtered_train_old.to_csv("./data/filtered_new_train_split.csv")
filtered_val_old.to_csv("./data/filtered_new_val_split.csv")
filtered_test_old.to_csv("./data/filtered_new_test_split.csv")

train_old.to_csv("./data/new_train_split.csv")
val_old.to_csv("./data/new_val_split.csv")
test_old.to_csv("./data/new_test_split.csv")
