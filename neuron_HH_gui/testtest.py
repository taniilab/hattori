i=0
j=0
k=0
i_limit = round(1.00, 3)
j_limit = round(1.00, 3)
k_limit = round(4.00, 3)

for l in range(21*21*5):
    print(str(i) + "  :  " + str(j) + "  :  " + str(k))

    if i < i_limit:
        i = round(i+0.05, 4)
    elif i == i_limit:
        i = 0
        j = round(j+0.05, 4)

    if j == j_limit+0.05:
        j = 0
        k = round(k+0.1, 4)

