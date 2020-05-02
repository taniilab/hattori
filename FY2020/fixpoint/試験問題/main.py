import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import datetime

### Search scope ###
year_start = 2005
month_start = 4
day_start = 10
year_end = 2017
month_end = 4
day_end = 30
start_date = datetime.datetime(year_start, month_start, day_start)
end_date = datetime.datetime(year_end, month_end, day_end)
### Parameters ###
read_path = "./var/log/httpd/*_log"
lump_size = 5
##################
timezone_label = np.arange(0, 24, 1)
timezone_list =[]
for i in range(24):
    timezone_list.append(0)
remotehost_list = pd.DataFrame()

def main():
    lump_init_line = 0
    files = glob.glob(read_path)

    # process files every lump size
    for j in range(len(files)):
        df = pd.DataFrame(np.zeros((lump_size, 1)))
        while len(df) == lump_size:
            try:
                line_tmp = range(lump_init_line, lump_init_line+lump_size)
                df = pd.read_csv(files[j], delimiter=" ", header=None, skiprows=lambda x: x not in line_tmp)
                pd.set_option('display.max_columns', 500)
                process(df)
                lump_init_line += lump_size
            except pd.io.common.EmptyDataError:
                break
        lump_init_line = 0

    # save
    remotehost_list_T = remotehost_list.transpose()
    remotehost_list_T = remotehost_list_T.rename(columns={'0': 'traffic'})
    remotehost_list_T.to_csv('./remotehost.csv', header=None)
    df = pd.DataFrame()
    df['timezone'] = timezone_label
    df['traffic'] = timezone_list
    df.to_csv('./timezone.csv')

    # visualize
    plt.figure(figsize=(10, 5))
    plt.bar(timezone_label, timezone_list)
    plt.xlabel('remotehost')
    plt.ylabel('traffic')
    plt.tight_layout()
    remotehost_list_T.plot.bar(figsize=(10, 5), legend=False)
    plt.xlabel('timezone')
    plt.ylabel('traffic')
    plt.tight_layout()
    plt.show()

def process(data):
    #extract access log in search scope
    drop_list = []
    for i in range(len(data)):
        date_text = data.iat[i, 3].split(':')[0].strip('[')
        date = datetime.datetime.strptime(date_text, '%d/%b/%Y')
        if date < start_date or date > end_date:
            drop_list.append(i)
    data = data.drop(drop_list)

    # count the number of accesses by time zone
    for i in range(len(data)):
        hour = data.iat[i, 3].split(':')[1]
        timezone_list[int(hour)] += 1

    # count the number of accesses by remote host
    for i in range(len(data)):
        if not data.iat[i, 0] in remotehost_list.columns:
            remotehost_list[data.iat[i, 0]] = [0]
        remotehost_list[data.iat[i, 0]] += 1

    print(remotehost_list)
    print("****************************")

if __name__ == '__main__':
     main()