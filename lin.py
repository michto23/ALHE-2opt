import numpy as np
import scipy.spatial as ssp
import numpy.random as nprnd
import matplotlib.pyplot as plt
from matplotlib import animation

numLow = 1
numHigh = 10000
numCities = 90
m = 3
Dist = np.zeros((numCities, numCities))


def generatecities(n):
    # Generate the coordinates of n random cities
    xcities = []
    ycities = []
    nprnd.seed(14)
    for x in range(0, n):
        xcities.append(nprnd.randint(numLow, numHigh))
        ycities.append(nprnd.randint(numLow, numHigh))
    xcities[0] = numHigh/2
    ycities[0] = numHigh/2
    return xcities, ycities


def plotcities(opttour, xys):
    # Plots the cities on a square
    # Input: list of 2 lists with x,y coordinates on each list
    xy1 = xys[0][:]
    xy2 = xys[1][:]
    # Sort according to latest tour optimization
    xy1 = [xy1[i] for i in opttour]
    xy2 = [xy2[i] for i in opttour]
    # Make it a cycle
    xy1.append(xy1[0])
    xy2.append(xy2[0])
    ims.append(plt.plot(xy1, xy2, linestyle='-', marker='o', color='b', markerfacecolor='red'))
    plt.ylabel('original path')


def genDistanceMat(x, y):
    X = np.array([x, y])
    distMat = ssp.distance.pdist(X.T)
    sqdist = ssp.distance.squareform(distMat)
    return sqdist


def calcTourLength(hamPath):
    tourLength = sum(Dist[hamPath[0:-1], hamPath[1:len(hamPath)]])
    tourLength += Dist[hamPath[-1], hamPath[0]]
    return tourLength

def calcTourValue(optlist, distMat):
    sum = 0
    for i in range(0, numCities - 1):
        distance = Dist[optlist[i]][optlist[i+1]]
        sum += distance
    return sum


# Generate cities
x, y = generatecities(numCities)

Dist = genDistanceMat(x, y)
# Generate initial tour
optlist = list(range(0, numCities))
improvement = 1
ims = []
fig1 = plt.figure()

total = 0
while True:
    plotcities(optlist, [x, y])
    count = 0
    for i in xrange(numCities - 2):
        # plotcities(optlist, [x, y])
        i1 = i + 1
        for j in xrange(i + 2, numCities):
            if j == numCities - 1:
                j1 = 0
            else:
                j1 = j + 1
            if i != 0 or j1 != 0:
                l1 = Dist[optlist[i]][optlist[i1]]
                l2 = Dist[optlist[j]][optlist[j1]]
                l3 = Dist[optlist[i]][optlist[j]]
                l4 = Dist[optlist[i1]][optlist[j1]]
                old = list(optlist)
                new_path = optlist[i1:j + 1]
                optlist[i1:j + 1] = new_path[::-1]
                if (calcTourValue(optlist, Dist) < calcTourValue(old, Dist)):
                    count += 1
                    print("FOUND")
                    print(calcTourLength(optlist))
                    print(calcTourValue(optlist, Dist))
                    # print(calcTourLength(optlist))
                else:
                    optlist = list(old)

                # if l1 + l2 > l3 + l4:
                #     new_path = optlist[i1:j + 1]
                #     optlist[i1:j + 1] = new_path[::-1]
                #     count += 1
                #     print(calcTourLength(optlist))
    total += count
    if count == 0: break

print('final optlist: ', optlist)
for i in optlist:
    print(i, x[i], y[i])
plotcities(optlist, [x, y])
im_ani = animation.ArtistAnimation(fig1, ims, interval=20, repeat_delay=300000000000000, blit=True)
plt.show()



# def opt2(path):
#     size = 20
#     global distance_table
#     total = 0
#     while True:
#         count = 0
#         for i in xrange(size - 2):
#             i1 = i + 1
#             for j in xrange(i + 2, size):
#                 if j == size - 1:
#                     j1 = 0
#                 else:
#                     j1 = j + 1
#                 if i != 0 or j1 != 0:
#                     l1 = Dist[optlist[i]][optlist[i1]]
#                     l2 = Dist[optlist[j]][optlist[j1]]
#                     l3 = Dist[optlist[i]][optlist[j]]
#                     l4 = Dist[optlist[i1]][optlist[j1]]
#                     if l1 + l2 > l3 + l4:
#                         new_path = optlist[i1:j+1]
#                         optlist[i1:j+1] = new_path[::-1]
#                         count += 1
#         total += count
#         if count == 0: break