x5 = [195, 200, 205, 210, 225, 230, 235, 240]
y5 = [6.7, 6.6, 6.7, 6.7, 6.3, 6.0, 5.8, 6.0]
x10 = [165, 195, 205, 220, 225, 230, 240, 250]
y10 = [8.5, 8.2, 7.4, 7.1, 6.5, 6.6, 6.6, 7.5]
x15 = [200, 210, 225, 235, 245]
y15 = [8.3, 7.4, 7.3, 7.7, 7.1]


import matplotlib.pyplot as plt

plt.xlabel(r'$presicion$')
plt.ylabel(r'$mistake$')
plt.title('')
label = 'k = 5'
plt.plot(x5, y5, linestyle='-', label=label)
label = 'k = 10'
plt.plot(x10, y10, linestyle='-', label=label)
label = 'k = 15'
plt.plot(x15, y15, linestyle='-', label=label)
plt.legend()
plt.grid(True)
plt.show()