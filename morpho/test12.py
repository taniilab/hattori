import sys


def main(lines):
    num_people = lines[0].split()[0]
    num_call =lines[0].split()[1]
    seats = list(lines[1])
    tri_seats = []
    res_counter = 0

    list_triple = []
    for i in range(len(seats)-2):

        if seats[i] == seats[i+1] == seats[i+2]:
            tri_seats.append(seats.copy())
            tri_seats[res_counter][i] = "."
            tri_seats[res_counter][i+1] = "."
            tri_seats[res_counter][i+2] = "."
            print(i)
            print(tri_seats[res_counter])
            print("")
            print(tri_seats[res_counter].count('S'))
            res_counter+=1


if __name__ == '__main__':
    main(['6 2', 'S.SSSS'])