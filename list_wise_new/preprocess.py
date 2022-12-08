import itertools
import pandas as pd
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf

import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout
import re

print('run preprocess.py')

# filter the useless
txt_reader  = open('MIND_small/behaviors.tsv',mode='r')
csv_file = 'Preprocess_MIND_small/txt2csv'
csv_file_trans = 'Preprocess_MIND_small/csv_trans'
with open(csv_file, 'w+', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')
    count = 0
    while True:
        new_line = txt_reader.readline()
        count += 1
        # if count == 1200:
        #     break
        if new_line == '':
            break
        new_line = re.split('[\t, ,\n]',new_line)
        num_Of_histo = 0
        for j in range(len(new_line)):
            if (new_line[j].find('N')==0) & (new_line[j].find('-') == -1):
                num_Of_histo += 1
        # if num_Of_histo<12:
        #     continue
        pointer = 0
        num2pop = [4,3,2,0]
        for j in num2pop:
            new_line.pop(j)
        new_line.pop(len(new_line)-1)
        for j in range(len(new_line)):
            if new_line[j].find('-') != -1:
                for k in range(j,len(new_line)):
                    new_line.pop(j)
                break
            new_line[j] = new_line[j].strip('U')
            new_line[j] = new_line[j].strip('N')
            if j == len(new_line)-2:
                pass
        writer.writerow(new_line)

with open(csv_file_trans, 'w+', newline='', encoding='utf-8') as csvfile2:
    writer = csv.writer(csvfile2, dialect='excel',delimiter=' ')
    csv_reader = open('Preprocess_MIND_small/txt2csv', mode='r')
    user_ID=0
    while True:
        user_ID += 1
        new_line = csv_reader.readline()
        if new_line == '':
            break
        new_line = re.split('[, \n]',new_line)
        new_line.pop(len(new_line)-1)
        for i in range(1,len(new_line)-1):
            line_divide = [user_ID, new_line[i],5,i]
            writer.writerow(line_divide)

item_reader = open('MIND_small/news.tsv', encoding='utf-8')
csv_file_item = 'Preprocess_MIND_small/item_csv'
with open(csv_file_item, 'w+', newline='', encoding='utf-8') as csvfile_item:
    writer_item = csv.writer(csvfile_item, dialect='excel')
    count = 0
    while True:
        count += 1
        # if count == 1200:
        #     break
        new_line = item_reader.readline()
        if new_line == '':
            break
        new_line = re.split('\t',new_line)
        new_line = [new_line[0],new_line[3]]
        new_line[0] = new_line[0].strip('N')
        writer_item.writerow(new_line)


