import pandas as pd

df = pd.DataFrame(columns= ["kashikoma"])

df2 = pd.DataFrame({'kanopero': [0, 1, 2],
                    'pippi': [3, 4, 5],
                    'ju': [6, 7, 8]},
                   columns=['kanopero', 'pippi', 'ju'])
df3 = pd.concat([df, df2])

df.to_csv("C:/Users/Hattori/Desktop/test.csv")
df2.to_csv("C:/Users/Hattori/Desktop/test.csv", mode="a")

#df3.to_csv("C:/Users/Hattori/Desktop/test.csv")