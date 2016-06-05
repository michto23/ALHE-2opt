import numpy as np
import scipy.spatial as ssp
import numpy.random as nprnd

import matplotlib.pyplot as plt
from matplotlib import animation

minCoordinate = 1
maxCoordinate = 10000
numPoints = 20
Dist = np.zeros((numPoints, numPoints))
isTrafficOnstart = 0

def createPoints(n):
    x = []
    y = []
    p = []
    nprnd.seed(287)
    for i in range(0, n):
        x.append(nprnd.randint(minCoordinate, maxCoordinate))
        y.append(nprnd.randint(minCoordinate, maxCoordinate))
        p.append(nprnd.randint(1, 4))

    x[0] = maxCoordinate / 2
    y[0] = maxCoordinate / 2
    p[0] = 0;
    return x, y, p

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

def generatePointsPlot(opttour, xys):
    xy1 = xys[0][:]
    xy2 = xys[1][:]
    xy1 = [xy1[i] for i in opttour]
    xy2 = [xy2[i] for i in opttour]
    xy1.append(xy1[0])
    xy2.append(xy2[0])
    ims.append(plt.plot(xy1, xy2, linestyle='-', marker='o', color='b', markerfacecolor='red'))
    # ims.append(plt.plot(xy1[0], xy2[0], linestyle='-', marker='o', color='b', markerfacecolor='blue'))
    plt.ylabel('Trasa dostawcy pizzy')


def generateDistanceMatrix(x, y):
    X = np.array([x, y])
    distMat = ssp.distance.pdist(X.T)
    sqdist = ssp.distance.squareform(distMat)
    return sqdist

def calcTourValue(optlist, distMat):
    sum = 0
    unhappy = 0
    for i in range(0, numPoints - 1):
        distance = Dist[optlist[i]][optlist[i+1]]
        sum += distance * trafficJamFunc(sum)
        unhappy += unhappyFunc(distance, sum) * patience[i]
        # sum += distance * trafficJamFunc(sum)
    return unhappy

def unhappyFunc(dist, sum):
    return (sum * sum * np.math.log(sum)) / maxCoordinate

def trafficJamFunc(sum):
    if (isTrafficOnstart == 0):
        if(sum < maxCoordinate):
            return 1
        elif(sum < 3 * maxCoordinate):
            return 2
        else:
            return 3
    else:
        if (sum < maxCoordinate):
            return 3
        elif (sum < 3 * maxCoordinate):
            return 2
        else:
            return 1

#x, y, patience = createPoints(numPoints)

x, y, patience, numPoints = readLocations()

Dist = np.zeros((numPoints, numPoints))



Dist = generateDistanceMatrix(x, y)
currentBest = list(range(0, numPoints))
ims = []
fig1 = plt.figure(1)
resultList = []
resultListImprove = []
isTrafficOnstart = patience[0]

total = 0
while True:
    generatePointsPlot(currentBest, [x, y])
    count = 0
    for i in xrange(numPoints - 2):
        i1 = i + 1
        for j in xrange(i + 2, numPoints):
            if j == numPoints - 1:
                j1 = 0
            else:
                j1 = j + 1
            if i != 0 or j1 != 0:
                l1 = Dist[currentBest[i]][currentBest[i1]]
                l2 = Dist[currentBest[j]][currentBest[j1]]
                l3 = Dist[currentBest[i]][currentBest[j]]
                l4 = Dist[currentBest[i1]][currentBest[j1]]
                resultList.append(calcTourValue(currentBest, Dist))
                old = list(currentBest)
                new_path = currentBest[i1:j + 1]
                currentBest[i1:j + 1] = new_path[::-1]
                if (calcTourValue(currentBest, Dist) < calcTourValue(old, Dist)):
                    count += 1
                    resultListImprove.append(calcTourValue(currentBest, Dist))
                    print(calcTourValue(currentBest, Dist))
                else:
                    currentBest = list(old)

    total += count
    if count == 0: break

calcTourValue(currentBest, Dist)
generatePointsPlot(currentBest, [x, y])

plt.figure(2)
plt.subplot(211)
plt.plot(range(len(resultListImprove)), resultListImprove, 'bo', range(len(resultListImprove)), resultListImprove, 'k') #x ,y, reprezentacja

plt.subplot(212)
plt.plot(range(len(resultList)), resultList, 'k') #x ,y, reprezentacja

im_ani = animation.ArtistAnimation(fig1, ims, interval=20, repeat_delay=300000000000000, blit=True)

plt.show()