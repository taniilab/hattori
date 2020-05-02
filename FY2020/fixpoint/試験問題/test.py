import datetime

now = datetime.datetime.now()
str_now = now.strftime('%d/%b/%Y')
print(str_now)

tetext_d = datetime.datetime.strptime(text, '%d/%b/%Y')
xt = '22/Apr/2017'
print(text_d)

d = datetime.datetime(2017, 4, 23)
print(d)

if d > text_d:
    print("kanopero")