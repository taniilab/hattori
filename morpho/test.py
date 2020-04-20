import sys


def main(argv):
    # このコードは引数と標準出力を用いたサンプルコードです。
    # このコードは好きなように編集・削除してもらって構いません。
    # ---
    # This is a sample code to use arguments and outputs.
    # Edit and remove this code as you like.
    #num_list_decimal = list(range(4 ** int(argv[0])))
    len_num_list_decimal = 4 ** int(argv[0])
    print(len_num_list_decimal)
    num_three = 0

    #print(num_list_decimal)

    for i in range(len_num_list_decimal):
        #num_three = 0  # 0, 1以外ならなんでも良い
        tmp = i
        #print(i)
        while tmp //4 != 0:
            tmp, mod = divmod(tmp, 4)
            if mod == 3:
                num_three += 1
        if tmp == 3:
            num_three += 1
        #print("number of three: {0}", num_three)
        #print("")
    print(num_three)
    return num_three

if __name__ == '__main__':
    main(['30'])