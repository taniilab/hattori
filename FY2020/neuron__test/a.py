from tqdm import tqdm
import time
import itertools

for j in tqdm(range(10)):
    for i in range(100):
        time.sleep(0.1)#ループしたい処理を書く

