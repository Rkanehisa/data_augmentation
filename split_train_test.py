import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
SEED = 42
random.seed(SEED)

negative_gun_dir = os.listdir("negative_gun_aug_Labels")
negative_noise_dir = os.listdir("negative_aug_Labels")
noise_dir = os.listdir("Labels")


positive = []
negative_gun = []
negative = []

for i in negative_gun_dir:
	print(i)
	if(i[-5:] == "a.txt"):
		negative_gun.append(i)
		print(i)

for i in negative_noise_dir:
	if(i[-5:] == "a.txt"):
		negative.append(i)

for i in noise_dir:
	#if(i[-5:] == "a.txt"):
	positive.append(i)

print(len(positive),len(positive)-len(negative_gun))
print(len(negative))

negative_balanced = []
for i in np.random.choice(negative, len(positive)-len(negative_gun), replace=False):
	negative_balanced.append(i)
for i in negative_gun:
	negative_balanced.append(i)


print(len(negative_balanced))
random.shuffle(negative_balanced)

positive_train,positive_test = train_test_split(positive, shuffle=False,test_size=0.2, random_state=42)
negative_train,negative_test = train_test_split(negative_balanced,shuffle=False, test_size=0.2,random_state=42)


train = positive_train+negative_train
test = positive_test+negative_test

print(len(train),len(test))


random.shuffle(train)
random.shuffle(test)

variants = ["a","b","c","d","e","f","g","h",]
#variants = ["a","b","c","d"]


arq = open("train.txt","w")
for i in train:
	for j in variants:
		arq.write(str("darknet/data/obj//obj/"+i[:-5]+j+".JPEG"))
		arq.write("\n")
arq.close()

arq = open("validation.txt","w")
for i in test:
	for j in variants:
		arq.write(str("darknet/data/obj/"+i[:-5]+j+".JPEG"))
		arq.write("\n")
arq.close()
