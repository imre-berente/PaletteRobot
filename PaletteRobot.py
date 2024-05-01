#f=open("adatok.txt","w")
#for i in range(15):
#    f.write(str(i)+" "+str(i*i)+"\n")
#f.close()
import turtle
from math import cos
from math import sin
from math import pi
from math import sqrt
from os import system

def drawline(wide,linecolor,x1,y1,x2,y2):
    turtle.width(wide)
    turtle.color(linecolor)
    turtle.up()
    turtle.goto(x1, y1)
    turtle.down()
    turtle.goto(x2, y2)

def paintrobot(midjointxyz, xyzfi):
    global paintdone
    if not paintdone:
        return
    paintdone = False

    turtle.clear()
    offset = 150
    if (linepos <= maxlinestep):
        # X-Z (side) view
        drawline(1,"darkgrey",linedat[0][0] * sc, linedat[0][2] * sc + offset,linedat[maxlinestep][0] * sc, linedat[maxlinestep][2] * sc + offset)
        # X-Y (top)view
        drawline(1,"darkgreen",linedat[0][0] * sc, linedat[0][1] * sc - offset,linedat[maxlinestep][0] * sc, linedat[maxlinestep][1] * sc - offset)

    # X-Z (side) view
    drawline(1,"pink",-4 * sc, offset - 1,4 * sc, offset - 1)
    xplace = [-4 * sc, 3.8 * sc]
    for i in range(2):
        for j in range(2):
            drawline(1, "pink",xplace[i] + j * sc / 5, offset - 1,xplace[i] + j * sc / 5, offset - 1 - sc / 2)

    drawline(3,"white",0, offset,midjointxyz[0] * sc, midjointxyz[2] * sc + offset)
    turtle.goto(xyzfi[0] * sc, xyzfi[2] * sc + offset) # continue line from midjoint
    turtle.dot(10) #dot at the TCP

    # X-Y (top)view
    turtle.width(1)
    turtle.color("pink")
    turtle.up()
    turtle.goto(2 * sc, -offset)
    turtle.down()
    turtle.setheading(90)
    turtle.circle(2 * sc)

    drawline(3,"green",0, -offset,midjointxyz[0] * sc, midjointxyz[1] * sc - offset)
    turtle.goto(xyzfi[0] * sc, xyzfi[1] * sc - offset)
    turtle.dot(10)

    turtle.setheading(deg(xyzfi[3]))
    turtle.forward(20)
    paintdone = True


def vecchk(vec, size=None):
    if (str(type(vec)) != "<class 'list'>"):
        printerr(vec, "not a list and vectors must be")

    if (size != None):
        if (len(vec) != size):
            printerr(vec, "size isn't" + str(size))

    for x in vec:
        if (str(type(x)) != "<class 'int'>") and (str(type(x)) != "<class 'float'>"):
            printerr(vec, "element has the type" + str(type(x)))


def vecmul(a, b):  # multiplies 2 vectors to get scalar
    vecchk(a)
    vecchk(b, len(a))
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return (res)


def deg(alfa):
    return (alfa / pi * 180)


def printerr(obj, message):
    print("The following object produced an error: ", message)
    print(obj)
    quit()


def matchk(mat, rowsize=None, colsize=None):
    if (str(type(mat)) != "<class 'list'>"):
        printerr(mat, "not a list, and matrices must be")

    if (rowsize == None):
        rsize = len(mat)
    else:
        rsize = rowsize
        if (len(mat) != rsize):
            printerr(mat, "size isn't" + str(rsize))

    for i in range(rsize):
        if (str(type(mat[i])) != "<class 'list'>"):
            printerr(mat, str(i) + ".th element is not a list, and for matrices it must be")

        if (i == 0):
            if (colsize == None):
                csize = len(mat[0])
            else:
                csize = colsize

        if (len(mat[i]) != csize):
            printerr(mat, str(i) + ".th element's size isn't" + str(csize))

        for j in range(csize):
            if (str(type(mat[i][j])) != "<class 'int'>") and (str(type(mat[i][j])) != "<class 'float'>"):
                printerr(mat, str(i) + "," + str(j) + "element has the type" + str(type(mat[i][j])))


def absolute(x):
    if (x < 0):
        return -x
    else:
        return x


def printmat(mat, rowheads=None):
    matchk(mat)
    rsize = len(mat)
    csize = len(mat[0])
    if (rowheads != None):
        if (str(type(rowheads)) != "<class 'list'>"):
            printerr(rowheads, "not a list and vectors must be")

        if (len(rowheads) != rsize):
            printerr(rowheads, "size isn't" + str(rsize))

        for i in range(rsize):
            if (str(type(rowheads[i])) != "<class 'str'>"):
                printerr(rowheads, str(i) + "element has the type" + str(type(rowheads[i])))

    for i in range(rsize):
        if (rowheads != None):
            print(rowheads[i], end="")
        for j in range(csize):
            if (round(mat[i][j]) == mat[i][j]):
                print("{:9d}".format(round(mat[i][j])), end="")
            else:
                print("{:9.4f}".format(mat[i][j]), end="")
        print("")


def matmul(mat_1, mat_2):  # multiplies the matrices, returning the result:
    matchk(mat_1)
    rsize = len(mat_1)
    csize = len(mat_1[0])
    matchk(mat_2, csize, rsize)
    res = [[0 for i in range(rsize)] for j in range(rsize)]
    for i in range(rsize):
        for j in range(rsize):
            for k in range(csize):
                res[i][j] += mat_1[i][k] * mat_2[k][j]
    return res


def copymat(mat):
    matchk(mat)
    rsize = len(mat)
    csize = len(mat[0])
    cp = [[0 for i in range(csize)] for j in range(rsize)]
    for i in range(rsize):
        for j in range(csize):
            cp[i][j] = mat[i][j]
    return cp


def matzerolimit(mat, zerolimit):  # set elements that are between +- zerolimits to real zero
    matchk(mat)
    size = len(mat)
    for i in range(size):
        for j in range(size):
            if absolute(mat[i][j]) < zerolimit:
                mat[i][j] = 0


def submat(mat, row, col):  # creates a submatrix that lacks row and col from the original
    matchk(mat)
    rsize = len(mat)
    if (row >= rsize):
        printerr(mat, "doesn't have " + str(row) + " rows")
    csize = len(mat[0])
    if (col >= csize):
        printerr(mat, "doesn't have " + str(col) + " columns")
    submat = [[0 for i in range(rsize - 1)] for j in range(csize - 1)]
    ii = 0
    jj = 0
    for i in range(rsize):
        if (i != row):
            for j in range(csize):
                if (j != col):
                    submat[ii][jj] = mat[i][j]
                    jj += 1
            ii += 1
            jj = 0
    return submat


def matdet_laplace(mat):
    matchk(mat)
    size = len(mat)
    if (len(mat[0]) != size):
        printerr(mat, "is not a square matrix")
    if (size == 2):
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    else:
        det = 0
        sign = 1
        for i in range(size):
            det += sign * mat[0][i] * matdet_laplace(submat(mat, 0, i))
            sign = -sign
        return det


def gauss_elimination(mat, limit):  # returns determinant of mat and inverts the mat if possible
    matchk(mat)
    old = copymat(mat)
    size = len(mat)
    lapd = matdet_laplace(mat)
    matd = 1
    for i in range(size):  # appending matrix with identity
        for j in range(size):
            mat[i].append(0)
        mat[i][size + i] = 1

    for active in range(size):
        i = active  # switching lines>= active to put nonzero to diagonal
        while mat[i][active] == 0:
            i += 1
            if (i == size):
                break
        if (i > active) and (i < size):
            u = mat[i]
            mat[i] = mat[active]
            mat[active] = u
            matd *= -1

        actact = mat[active][active]  # active diagonal must be 1 or 0
        if (actact == 0):
            matd = 0
            break
        elif (actact != 1):
            for i in range(active + 1, size * 2):
                mat[active][i] /= actact
            matd *= actact
            mat[active][active] = 1

        for i in range(active + 1,
                       size):  # making all lines 0 in the active column below the active line by subtracting active line
            if (mat[i][active] != 0):
                for j in range(active + 1, size * 2):
                    mat[i][j] -= mat[active][j] * mat[i][active]
                mat[i][active] = 0

    if absolute(matd - lapd) > limit * absolute(matd):
        print("Gauss determinant is", matd, "Laplace determinant is", lapd, "they should be within", limit, "*", matd)
        quit()

    if absolute(matd) >= limit:
        for i in range(
                size - 1):  # making all elements 0 right of the diagonal by subtracting corresponding diagonal line
            for j in range(i + 1, size):
                if (mat[i][j] != 0):
                    for k in range(j + 1, size * 2):
                        mat[i][k] -= mat[j][k] * mat[i][j]
                    mat[i][j] = 0

        for i in range(size):  # removing identity matrix from the front
            for j in range(size):
                mat[i].pop(0)

        test = matmul(old, mat)
        test2 = matmul(mat, old)
        for i in range(size):
            for j in range(size):
                if (i == j):
                    correct = 1
                else:
                    correct = 0
                if (absolute(test[i][j] - correct) > limit * limit):
                    printerr(old, "inverse matrix failed A*A^-1 test")
                if (absolute(test2[i][j] - correct) > limit * limit):
                    printerr(old, "inverse matrix failed A^-1*A test")

    return matd


def strnum(num, limit=0.0000000001):
    if (absolute(num - round(num)) < limit):
        return "{:5d}   ".format(round(num))
    else:
        return "{:8.2f}".format(num)


def calcjakobi(jvec):
    vecchk(jvec, 4)
    jak = [[0 for i in range(4)] for j in range(4)]
    j3 = pi - jvec[1] - jvec[2]
    jak[0][0] = -a1 * sin(jvec[0]) * cos(jvec[1]) + a2 * sin(jvec[0]) * cos(j3)  # dx/djoint1
    jak[0][1] = -a1 * cos(jvec[0]) * sin(jvec[1]) - a2 * cos(jvec[0]) * sin(j3)  # dx/djoint2
    jak[0][2] = -a2 * cos(jvec[0]) * sin(j3)  # dx/djoint3
    jak[0][3] = 0  # dx/djoint4

    jak[1][0] = a1 * cos(jvec[0]) * cos(jvec[1]) - a2 * cos(jvec[0]) * cos(j3)  # dy/djoint1
    jak[1][1] = -a1 * sin(jvec[0]) * sin(jvec[1]) - a2 * sin(jvec[0]) * sin(j3)  # dy/djoint2
    jak[1][2] = -a2 * sin(jvec[0]) * sin(j3)  # dy/djoint3
    jak[1][3] = 0  # dy/djoint4

    jak[2][0] = 0  # dz/djoint1
    jak[2][1] = a1 * cos(jvec[1]) - a2 * cos(j3)  # dz/djoint2
    jak[2][2] = -a2 * cos(j3)  # dz/djoint3
    jak[2][3] = 0  # dz/djoint4

    jak[3][0] = 1  # dfi/djoint1
    jak[3][1] = 0  # dfi/djoint2
    jak[3][2] = 0  # dfi/djoint3
    jak[3][3] = 1  # dfi/djoint4

    return jak


def calcxyzfi(jvec,
              midjointxyz=None):  # calculate XYZfi coordinates of TCP from joint coordinates. Returns midjoint (vector 3 needed)
    if (midjointxyz == None):
        midjointxyz = [0, 0, 0]
    else:
        vecchk(midjointxyz, 3)

    midjointxyz[0] = a1 * cos(jvec[0]) * cos(jvec[1])
    midjointxyz[1] = a1 * sin(jvec[0]) * cos(jvec[1])
    midjointxyz[2] = a1 * sin(jvec[1])

    xyzfi = [0, 0, 0, 0]
    xyzfi[0] = midjointxyz[0] - a2 * cos(jvec[0]) * cos(pi - jvec[1] - jvec[2])
    xyzfi[1] = midjointxyz[1] - a2 * sin(jvec[0]) * cos(pi - jvec[1] - jvec[2])
    xyzfi[2] = midjointxyz[2] + a2 * sin(pi - jvec[1] - jvec[2])
    xyzfi[3] = jvec[0] + jvec[3]
    return xyzfi


def printjakobi(jakobi, jakobidet, invjakobi):  # Prints Jakobi matrix, its determinant and its inverse to the console
    print("Jakobi matrix:")
    print("      ../dj1   ../dj2   ../dj3   ../dj4")
    rowhead = [" dx", " dy", " dz", "dfi"]
    printmat(jakobi, rowhead)

    print("Jakobi matrix determinant: {:9.4f}".format(jakobidet))
    if (absolute(jakobidet) > 0.01):
        print("Inverse Jakobi matrix:")
        printmat(invjakobi)
    else:
        print("Jakobi is singular!")


def strjvec(jvec):
    return strnum(deg(jvec[0])) + strnum(deg(jvec[1])) + strnum(deg(jvec[2])) + strnum(deg(jvec[3]))


def strxyzfivec(xyzfi):
    return strnum(xyzfi[0] * sc) + strnum(xyzfi[1] * sc) + strnum(xyzfi[2] * sc) + strnum(deg(xyzfi[3]))


def printui(jvec, xyzfi):  # prints current joint and world coordinates, along with the user keys
    vecchk(jvec, 4)
    vecchk(xyzfi, 4)
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("Palette robot movement simulator:")
    print("You can move the robot in joint and world with the keys below:")
    print('  +1-q    +2-w    +3-e    +4-r   +5-t    +6-y    +7-u    +8-i', fok, "degrees/pixels keypress (+p-m)")
    print('   j1      j2      j3      j4    TCPx    TCPy    TCPz   TCPfi')
    print(strjvec(jvec) + strxyzfivec(xyzfi))
    if linepos < maxlinestep:
        print("\n in line mode, spacebar moves the robot forward. Currently in step", linepos, "/", maxlinestep,
              " Any other key exits the mode")
        print("           TCPx    TCPy    TCPz   TCPfi")
        print("Start:  ", strxyzfivec(linedat[0]))
        print("End:    ", strxyzfivec(linedat[maxlinestep]))
    elif linepos == maxlinestep:
        print("Line mode reached its end, you can continue with any function keys")
    else:
        print("For drawing a straight line from current location to a choosen XYZfi position press l")


def veclength(vec):
    length = 0
    for x in vec:
        length += x ** 2
    return sqrt(length)


def vecadd(v1, v2, v2mul=1):  # add v1+v2*v2mul (scalar)
    size = len(v1)
    if (len(v2) != size):
        printerr(v2, "has length different from" + str(size))

    vsum = [0 for i in range(size)]
    for i in range(size):
        vsum[i] = v1[i] + v2[i] * v2mul
    return vsum


def worldmoverobot(xyzfi, delta=True,
                   linemode=False):  # in delta mode, dxyzfi means change. In absolute mode it means the complete goal vector
    if not linemode:
        global linepos
        linepos = maxlinestep + 1

    global jvec

    system('cls')

    vecchk(xyzfi, 4)

    oldxyzfi = calcxyzfi(jvec)
    if delta:
        dxyzfi = xyzfi
        goalxyzfi = vecadd(oldxyzfi, dxyzfi)
    else:
        goalxyzfi = xyzfi
        dxyzfi = vecadd(goalxyzfi, oldxyzfi, -1)

    if (veclength(goalxyzfi[0:3]) > 1.99) or (goalxyzfi[2] < -0.001):
        print("Movement goal is out of work area")
        printui(jvec, oldxyzfi)
        return

    print("Performing multiple approximations: Joint_N+1 = Joint_N - Jakobi^-1*(World_goal-World_N), N=1: original")
    iterlimit = 0.01
    olderr = veclength(dxyzfi)
    midjoint = [0, 0, 0]

    for itnum in range(999):
        jakobi = calcjakobi(jvec)  # calculating jakobi, determinant, inverse
        matzerolimit(jakobi, 0.00000001)
        invjakobi = copymat(jakobi)
        jakobidet = gauss_elimination(invjakobi, 0.01)
        matzerolimit(invjakobi, 0.00000001)
        printjakobi(jakobi, jakobidet, invjakobi)

        if (absolute(jakobidet) <= 0.01):
            print("Jakobi is singular, only joint movements are allowed!")
            printui(jvec, oldxyzfi)
            return

        print("                           TCPx    TCPy    TCPz   TCPfi   error")
        print("XYZfi wanted:           " + strxyzfivec(goalxyzfi) + " := 0")
        print("old XYZfi:              " + strxyzfivec(oldxyzfi) + strnum(olderr * sc))

        djoint = [0, 0, 0, 0]  # matrix * vector = vector
        for i in range(4):
            djoint[i] = vecmul(invjakobi[i], dxyzfi)
        tryj = vecadd(jvec, djoint)
        tryxyzfi = calcxyzfi(tryj, midjoint)
        tryerr = veclength(vecadd(tryxyzfi, goalxyzfi, -1))
        print("try XYZfi:              " + strxyzfivec(tryxyzfi) + strnum(tryerr * sc))

        maxchange = absolute(dxyzfi[0])  # finding the dominant coordinate of the change for overjump check
        maxchangeindex = 0
        for i in range(1, 4):
            if maxchange < absolute(dxyzfi[i]):
                maxchange = absolute(dxyzfi[i])
                maxchangeindex = i
        if (maxchange < 0.001):
            print("Convergence reached")
            break
        tryd_xyzfi = vecadd(tryxyzfi, oldxyzfi, -1)
        overjump = tryd_xyzfi[maxchangeindex] / dxyzfi[maxchangeindex]
        if (overjump < 0):
            print("Iteration is diverging")
            printui(jvec, oldxyzfi)
            return
        elif (overjump > 1):
            tryj = vecadd(jvec, djoint, 1 / overjump)
            tryxyzfi = calcxyzfi(tryj, midjoint)
            tryerr = veclength(vecadd(tryxyzfi, goalxyzfi, -1))
            print("Corrected try XYZfi:    " + strxyzfivec(tryxyzfi) + strnum(tryerr * sc))

        if (midjoint[2] < -0.001):
            print("Midjoint out of work area")
            printui(jvec, oldxyzfi)
            return

        if (tryerr > olderr):
            print("Calculated position is worse than the original, rejected. Probable singularity or out of work area.")
            printui(jvec, oldxyzfi)
            return

        print("Iteration step", itnum + 1, "complete\n")
        jvec = tryj

        if (tryerr * sc <= iterlimit):
            print("Iteration converged to" + strnum(iterlimit))
            break

        oldxyzfi = tryxyzfi
        olderr = tryerr
        dxyzfi = vecadd(goalxyzfi, oldxyzfi, -1)

    printui(jvec, tryxyzfi)
    paintrobot(midjoint, tryxyzfi)


def jointmoverobot(djvec, linemode=False):
    if not linemode:
        global linepos
        linepos = maxlinestep + 1

    global jvec
    tryjvec = vecadd(jvec, djvec)

    midjointxyz = [0, 0, 0]
    xyzfi = calcxyzfi(tryjvec, midjointxyz)
    if (midjointxyz[2] < -0.001) or (xyzfi[2] < -0.001):
        print("Out of work area")
        return

    system('cls')
    jvec = tryjvec
    printui(jvec, xyzfi)
    paintrobot(midjointxyz, xyzfi)


def j1plus():
    jointmoverobot([fok / 180 * pi, 0, 0, 0])


def j1minus():
    jointmoverobot([-fok / 180 * pi, 0, 0, 0])


def j2plus():
    jointmoverobot([0, fok / 180 * pi, 0, 0])


def j2minus():
    jointmoverobot([0, -fok / 180 * pi, 0, 0])


def j3plus():
    jointmoverobot([0, 0, fok / 180 * pi, 0])


def j3minus():
    jointmoverobot([0, 0, -fok / 180 * pi, 0])


def j4plus():
    jointmoverobot([0, 0, 0, fok / 180 * pi])


def j4minus():
    jointmoverobot([0, 0, 0, -fok / 180 * pi])


def morefok():
    global fok
    if fok == 5:
        fok = 15
    elif fok == 2:
        fok = 5
    elif fok == 1:
        fok = 2
    jointmoverobot([0, 0, 0, 0])


def lessfok():
    global fok
    if fok == 2:
        fok = 1
    elif fok == 5:
        fok = 2
    elif fok == 15:
        fok = 5
    jointmoverobot([0, 0, 0, 0])


def xplus():
    worldmoverobot([fok / sc, 0, 0, 0])


def xminus():
    worldmoverobot([-fok / sc, 0, 0, 0])


def yplus():
    worldmoverobot([0, fok / sc, 0, 0])


def yminus():
    worldmoverobot([0, -fok / sc, 0, 0])


def zplus():
    worldmoverobot([0, 0, fok / sc, 0])


def zminus():
    worldmoverobot([0, 0, -fok / sc, 0])


def fiplus():
    worldmoverobot([0, 0, 0, fok / 180 * pi])


def fiminus():
    worldmoverobot([0, 0, 0, -fok / 180 * pi])


def stepline():
    global linepos
    if (linepos <= maxlinestep):
        if (linepos < maxlinestep):
            linepos += 1
        worldmoverobot(linedat[linepos], False, True)
    else:
        print("Only works in line mode, press l to set it up")


def getfloat(prompt):
    needinput = True
    while needinput:
        s = input(prompt)
        try:
            f = float(s)
        except ValueError:
            print("This isn't a number")
            continue
        needinput = False
    return f


def setline():
    print(
        "A line will be drawn from the current position to the given coordinates. The robot operates in a 200 radius sphere")

    global linepos
    global linedat
    linedat[maxlinestep][0] = getfloat("Input X:") / sc
    linedat[maxlinestep][1] = getfloat("Input Y:") / sc
    linedat[maxlinestep][2] = getfloat("Input Z:") / sc
    if veclength(linedat[maxlinestep]) >= 1.99 or (linedat[maxlinestep][2] < 0):
        print(
            "Target coordinates are out of work area of robot. Press 'l' again to set new coordinates or use any other key as usual")
        return
    linedat[maxlinestep][3] = getfloat("Input fi in degrees, 0 means looking at +X coordinate:") / 180 * pi

    linepos = 0
    oldxyzfi = calcxyzfi(jvec)
    for j in range(4):
        for i in range(maxlinestep):
            linedat[i][j] = (oldxyzfi[j] * (maxlinestep - i) + linedat[maxlinestep][j] * i) / maxlinestep

    jointmoverobot([0, 0, 0, 0], True)


# --------------------------------------------------------- main -------------------------------------------------------------
wn = turtle.Screen()
wn.bgcolor("black")
wn.listen()
turtle.speed(0)

fok = 15
a1 = 1
a2 = 1
sc = 100  # screen scale factor
paintdone = True

maxlinestep = 50
linepos = maxlinestep + 1
linedat = [[0 for i in range(4)] for j in range(maxlinestep + 1)]

jvec = [0, 45 / 180 * pi, -45 / 180 * pi, 0]
jointmoverobot([0, 0, 0, 0])

wn.onkeyrelease(j1plus, "1")
wn.onkeyrelease(j1minus, "q")
wn.onkeyrelease(j2plus, "2")
wn.onkeyrelease(j2minus, "w")
wn.onkeyrelease(j3plus, "3")
wn.onkeyrelease(j3minus, "e")
wn.onkeyrelease(j4plus, "4")
wn.onkeyrelease(j4minus, "r")
wn.onkeyrelease(xplus, "5")
wn.onkeyrelease(xminus, "t")
wn.onkeyrelease(yplus, "6")
wn.onkeyrelease(yminus, "y")
wn.onkeyrelease(zplus, "7")
wn.onkeyrelease(zminus, "u")
wn.onkeyrelease(fiplus, "8")
wn.onkeyrelease(fiminus, "i")
wn.onkeyrelease(morefok, "p")
wn.onkeyrelease(lessfok, "m")
wn.onkeyrelease(setline, "l")
wn.onkeyrelease(stepline, " ")

wn.mainloop()
