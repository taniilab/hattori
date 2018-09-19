@@ -1,28 +0,0 @@
from multiprocessing import Pool

class Main():
    def __init__(self):
        self.value = 0
        self.mp_counter = 0

    def run(self, process):
        self.value = process +self.mp_counter

        return self.value


def main():
    main = Main()
    process = 6
    mp_cycle = 3

    for i in range(0, mp_cycle):
        pool = Pool(process)
        res = pool.map(main.run, range(process))
        for j in range(len(res)):
            print(res[j])

        main.mp_counter += 100

if __name__ == '__main__':
     main()