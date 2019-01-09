import serial as s
from time import sleep
import pandas as pd

#save_path = "C:/Users/Hattori/Desktop/"
save_path = "//192.168.13.10/Public/hattori/sens/"

class Main():
    def __init__(self):
        self.ser = s.Serial("COM18", 115200)
        print(self.ser.name)
        self.ser.flushInput()
        self.read_data_length = 5


def main():
    m = Main()

    m.ser.write(b"sf,1\r\n")
    data_b = m.ser.read(5)
    print(data_b)

    m.ser.write(b"ss,40000001\r\n")
    data_b = m.ser.read(5)
    print(data_b)

    m.ser.write(b"sr,90000000\r\n")
    data_b = m.ser.read(5)
    print(data_b)

    m.ser.write(b"pz\r\n")
    data_b = m.ser.read(5)
    print(data_b)

    m.ser.write(b"ps,FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\r\n")
    data_b = m.ser.read(5)
    print(data_b)

    m.ser.write(b"pc,EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE,1A,20\r\n")
    data_b = m.ser.read(5)
    print(data_b)

    m.ser.write(b"r,1\r\n")
    data_b = m.ser.read(8)
    print(data_b)
    sleep(5)

    m.ser.flushInput()
    m.ser.write(b"e,0,001ec0550fd1\r\n")
    data_b = m.ser.read(16)
    print(data_b)

    m.ser.write(b"cuwc,123456789012345678901234567890ff,1\r\n")
    data_b = m.ser.read(20)
    print(data_b)
    sleep(3)
    m.ser.flushInput()
    """
    while b"\n" != m.ser.read(1):
        pass
    """

    #junk data
    """
    for i in range(20):
        print(m.ser.read(55))
    """

    df_init = pd.DataFrame({'accelX': [0], 'accelY': [0], 'accelZ': [0],
                            'gyloX': [0], 'gyloY': [0], 'gyloZ': [0],
                            'psdX1': [0], 'psdX2': [0], 'psdY1': [0], 'psdY2': [0]})
    df_init.to_csv(save_path + "wireless_psd.csv", mode="a")

    acx = 0
    acy = 0
    acz = 0
    gyx = 0
    gyy = 0
    gyz = 0
    px1 = 0
    px2 = 0
    py1 = 0
    py2 = 0

    while 1:
        data_b = m.ser.read(55)

        if "F" == chr(data_b[12]) and "A" == chr(data_b[49]) and "A" == chr(data_b[50]) and "A" == chr(data_b[51]):
            acxh = 16 *int(chr(data_b[16]), 16) + int(chr(data_b[17]), 16)
            acxl = 16 *int(chr(data_b[18]), 16) + int(chr(data_b[19]), 16)
            acx = acxh*256 + acxl
            acyh = 16 *int(chr(data_b[20]), 16) + int(chr(data_b[21]), 16)
            acyl = 16 *int(chr(data_b[22]), 16) + int(chr(data_b[23]), 16)
            acy = acyh*256 + acyl
            aczh = 16 *int(chr(data_b[24]), 16) + int(chr(data_b[25]), 16)
            aczl = 16 *int(chr(data_b[26]), 16) + int(chr(data_b[27]), 16)
            acz = aczh*256 + aczl
            gyxh = 16 *int(chr(data_b[28]), 16) + int(chr(data_b[29]), 16)
            gyxl = 16 *int(chr(data_b[30]), 16) + int(chr(data_b[31]), 16)
            gyx = gyxh*256 + gyxl
            gyyh = 16 * int(chr(data_b[28]), 16) + int(chr(data_b[29]), 16)
            gyyl = 16 * int(chr(data_b[30]), 16) + int(chr(data_b[31]), 16)
            gyy = gyyh * 256 + gyyl
            gyzh = 16 * int(chr(data_b[28]), 16) + int(chr(data_b[29]), 16)
            gyzl = 16 * int(chr(data_b[30]), 16) + int(chr(data_b[31]), 16)
            gyz = gyzh * 256 + gyzl
            """
            df = pd.DataFrame(columns = [acx, acy, acz, gyz, gyy, gyz, px1, px2, py1, py2])
            df.to_csv(save_path + "wireless_psd.csv", mode= "a")
            """
            print("imu")

        elif "0" == chr(data_b[12]) and "A" == chr(data_b[49]) and "A" == chr(data_b[50]) and "A" == chr(data_b[51]):
            px1h = 16 *int(chr(data_b[16]), 16) + int(chr(data_b[17]), 16)
            px1l = 16 *int(chr(data_b[18]), 16) + int(chr(data_b[19]), 16)
            px1 = 5.0 * (px1h*256 + px1l)/4096
            px2h = 16 *int(chr(data_b[20]), 16) + int(chr(data_b[21]), 16)
            px2l = 16 *int(chr(data_b[22]), 16) + int(chr(data_b[23]), 16)
            px2 = 5.0 * (px2h*256 + px2l)/4096
            py1h = 16 *int(chr(data_b[24]), 16) + int(chr(data_b[25]), 16)
            py1l = 16 *int(chr(data_b[26]), 16) + int(chr(data_b[27]), 16)
            py1 = 5.0 * (py1h*256 + py1l)/4096
            py2h = 16 *int(chr(data_b[28]), 16) + int(chr(data_b[29]), 16)
            py2l = 16 *int(chr(data_b[30]), 16) + int(chr(data_b[31]), 16)
            py2 = 5.0 * (py2h*256 + py2l)/4096

            df = pd.DataFrame(columns = [acx, acy, acz, gyz, gyy, gyz, px1, px2, py1, py2])
            df.to_csv(save_path + "wireless_psd.csv", mode= "a")
            print("psd")
            print(df)

        else:
            print("missed")
            df = pd.DataFrame(columns = [acx, acy, acz, gyz, gyy, gyz, px1, px2, py1, py2])
            df.to_csv(save_path + "wireless_psd.csv", mode= "a")
            while b"\n" != m.ser.read(1):
                pass


if __name__ == '__main__':
    main()