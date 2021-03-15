import time
path = "C:/Users/Neuron/Documents/GitHub/hattori/stimulation/"
x = ""
counter = 0

"""
#theta burst stimulation 1set
for i in range(20):
    x += "1\n"
    counter += 1
for i in range(389):
    x += "0\n"
    counter += 1
for i in range(20):
    x += "1\n"
    counter += 1
for i in range(389):
    x += "0\n"
    counter += 1
for i in range(20):
    x += "1\n"
for i in range(389):
    x += "0\n"
    counter += 1
for i in range(20):
    x += "1\n"
    counter += 1
for i in range(389):
    x += "0\n"
    counter += 1
for i in range(20):
    x += "1\n"
    counter += 1
for i in range(389):
    x += "0\n"
    counter += 1
for i in range(6167):
    x += "0\n"
    counter += 1
"""

#テタヌス
for j in range(100):
    for i in range(4):
        x += "1\n"
        counter += 1
    for i in range(78):
        x += "0\n"
        counter += 1

#theta burst stimulation 10set→分解能不足
"""
for j in range(10):
    for i in range(2):
        x += "1\n"
        counter += 1
    for i in range(39):
        x += "0\n"
        counter += 1
    for i in range(2):
        x += "1\n"
        counter += 1
    for i in range(39):
        x += "0\n"
        counter += 1
    for i in range(2):
        x += "1\n"
        counter += 1
    for i in range(39):
        x += "0\n"
        counter += 1
    for i in range(2):
        x += "1\n"
        counter += 1
    for i in range(39):
        x += "0\n"
        counter += 1
    for i in range(2):
        x += "1\n"
        counter += 1
    for i in range(39):
        x += "0\n"
        counter += 1
    for i in range(614):
        x += "0\n"
        counter += 1
x += "0\n"
counter += 1
x += "0\n"
counter += 1
"""
print(counter)
#print(x)
#f = open(path+'TBS_0.5m_9.5m_5cycle_200mhz_.txt', 'w')
f = open(path+'tetanus_0.5m_9.5m_100hz_1s.txt', 'w')
f.write(x)
f.close()