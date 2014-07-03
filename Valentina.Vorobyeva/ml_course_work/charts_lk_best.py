x1 = [214.5, 215, 215.5]
y1 = [6.2, 6.2, 6.2]
x5 = [230, 235, 240]
y5 = [6.0, 5.9, 5.9]
x6 = [239.5, 240, 240.5]
y6 = [6.7, 6.7, 6.7]
x7 = [180, 215, 220, 225, 230, 240]
y7 = [7.5, 6.4, 6.1, 6.3, 6.0, 5.9]
x8 = [205, 220]
y8 = [7.2, 6.7]
x9 = [215, 220, 225, 240]
y9 = [6.7, 6.7, 6.3, 6.4]
x10 = [215, 220]
y10 = [6.7, 6.4]

import matplotlib.pyplot as plt

plt.xlabel(r'$presicion$')
plt.ylabel(r'$mistake$')
plt.title('')
label = 'k = 1'
plt.plot(x1, y1, linestyle='-', label=label)
label = 'k = 5'
plt.plot(x5, y5, linestyle='-', label=label)
label = 'k = 6'
plt.plot(x6, y6, linestyle='-', label=label)
label = 'k = 7'
plt.plot(x7, y7, linestyle='-', label=label)
label = 'k = 8'
plt.plot(x8, y8, linestyle='-', label=label)
label = 'k = 9'
plt.plot(x9, y9, linestyle='-', label=label)
label = 'k = 10'
plt.plot(x10, y10, linestyle='-', label=label)
plt.legend()
plt.grid(True)
plt.show()