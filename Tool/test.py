import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot([1, 2, 3, 4])
ax.set_xlabel('日本語を簡単に使える喜び')
#plt.xlabel('日本語を簡単に使える喜び')
plt.show()