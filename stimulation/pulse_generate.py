import time
path = "C:/Users/Neuron/Documents/GitHub/hattori/stimulation/"
savename = '24_monopolar_100mL_200mhz.fy'
x = ""
counter = 0

#バイポーラ
"""
for i in range(164):
    x += "-1\n"
    counter += 1
for i in range(164):
    x += "1\n"
    counter += 1
for i in range(7864):
    x += "0\n"
    counter += 1
"""

#モノポーラ
for i in range(164):
    x += "-1\n"
    counter += 1
for i in range(1):
    x += "1\n"
    counter += 1
for i in range(8027):
    x += "0\n"
    counter += 1

#テタヌス
"""
for j in range(100):
    for i in range(10):
        x += "-1\n"
        counter += 1
    for i in range(10):
        x += "1\n"
        counter += 1
    for i in range(62):
        x += "0\n"
        counter += 1
"""
#シータバースト
"""
for j in range(5):
    for i in range(10):
        x += "-1\n"
        counter += 1
    for i in range(10):
        x += "1\n"
        counter += 1
    for i in range(62):
        x += "0\n"
        counter += 1
    for i in range(10):
        x += "-1\n"
        counter += 1
    for i in range(10):
        x += "1\n"
        counter += 1
    for i in range(62):
        x += "0\n"
        counter += 1
    for i in range(10):
        x += "-1\n"
        counter += 1
    for i in range(10):
        x += "1\n"
        counter += 1
    for i in range(62):
        x += "0\n"
        counter += 1
    for i in range(10):
        x += "-1\n"
        counter += 1
    for i in range(10):
        x += "1\n"
        counter += 1
    for i in range(62):
        x += "0\n"
        counter += 1
    for i in range(10):
        x += "-1\n"
        counter += 1
    for i in range(10):
        x += "1\n"
        counter += 1
    for i in range(62):
        x += "0\n"
        counter += 1
    for i in range(1228):
        x += "0\n"
        counter += 1
"""

print(counter)
f = open(path+savename, 'w')
f.write(x)
f.close()
