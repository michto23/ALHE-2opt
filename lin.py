import numpy as np
import scipy.spatial as ssp
import numpy.random as nprnd

import matplotlib.pyplot as plt
from matplotlib import animation

numLow = 1
numHigh = 10000
m = 3
numCities = 50


def generatecities(n):
    # Generate the coordinates of n random cities
    xcities = []
    ycities = []
    patience = []
    nprnd.seed(1103)
    for x in range(0, n):
        xcities.append(nprnd.randint(numLow, numHigh))
        ycities.append(nprnd.randint(numLow, numHigh))
        patience.append(nprnd.randint(1, 4))

    xcities[0] = numHigh/2
    ycities[0] = numHigh/2
    return xcities, ycities, patience

def readLocations():
    file = open('input.txt', 'r')
    x = []
    y = []
    p = []
    num = 0
    for line in file:
        tab = line.split()
        x.append(int(tab[0]))
        y.append(int(tab[1]))
        p.append(int(tab[2]))
        num = num + 1
    file.close()
    return x, y, p, num

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
    # ims.append(plt.plot(xy1[0], xy2[0], linestyle='-', marker='o', color='b', markerfacecolor='blue'))
    plt.ylabel('original path')


def genDistanceMat(x, y):
    X = np.array([x, y])
    distMat = ssp.distance.pdist(X.T)
    sqdist = ssp.distance.squareform(distMat)
    return sqdist


# def calcTourLength(hamPath):
#     tourLength = sum(Dist[hamPath[0:-1], hamPath[1:len(hamPath)]])
#     tourLength += Dist[hamPath[-1], hamPath[0]]
#     return tourLength

def calcTourValue(optlist, distMat):
    sum = 0
    unhappy = 0
    for i in range(0, numCities - 1):
        distance = Dist[optlist[i]][optlist[i+1]]
        sum += distance  #* trafficJamFunc(sum)
        unhappy += unhappyFunc(sum) #* patience[i]
        # sum += distance * trafficJamFunc(sum)
    return sum

def unhappyFunc(sum):
    return sum * np.math.log(sum)

def trafficJamFunc(sum):
    if(sum < numHigh):
        return 1
    elif(sum < 3 * numHigh):
        return 2
    else:
        return 3


# Generate cities
# x, y, patience = generatecities(numCities)

x, y, patience, numCities = readLocations()

Dist = np.zeros((numCities, numCities))



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
                    print("FOUND old:", calcTourValue(old, Dist), " new ", calcTourValue(optlist, Dist))
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
calcTourValue(optlist, Dist)
plotcities(optlist, [x, y])
im_ani = animation.ArtistAnimation(fig1, ims, interval=20, repeat_delay=300000000000000, blit=True)
plt.show()