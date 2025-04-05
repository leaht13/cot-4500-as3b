# Leah Tomberg

import numpy


#       1 Gaussian Elimination & Backward Substitution
ma1=numpy.matrix([[2,-1,1,6],[1,3,1,0],[-1,5,4,-3]])

# Remember its by index so rows 1-3 correspond to index 0-2
ma1[2]=ma1[1]+ma1[2]

# Whichever row we multiply smth to cant be the one that becomes smth new

# Doing times (1/2) or .5 doesnt work 
ma1[0]=ma1[0]+(-2*ma1[1])


# another way of saying ma1[[0,1], :]
ma1[[0,1]]=ma1[[1,0]]


# next step
ma1[2]=ma1[2]+ma1[1]


# next
ma1[1]=ma1[1]+7*ma1[2]


# Swap (brings to echelon form)
ma1[[1,2]]=ma1[[2,1]]

# Tying lose ends and printing results
first=numpy.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
second=numpy.array([6, 0, -3])
answer1=numpy.linalg.solve(first, second)
answer1=answer1.astype(int)
print(answer1)





#       2 LU Factorization

ma2 = numpy.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

leng = ma2.shape[0]
L = numpy.zeros_like(ma2)
U = numpy.zeros_like(ma2)
    
for i in range(leng):
    # U
    for j in range(i, leng):
        U[i, j] = ma2[i, j] - numpy.dot(L[i, :i], U[:i, j])
        
    # L
    for j in range(i, leng):
        if i == j:
            # The diagonal for L is 1 in this case
             L[i, i] = 1  
        else:
             L[j, i] = (ma2[j, i] - numpy.dot(L[j, :i], U[:i, i])) / U[i, i]


# Finding the det & print results

det=numpy.linalg.det(ma2)

print("\n")
print(det)
print("\n")
print(L)
print("\n")
print(U)




#       3 Check for Diagonally Dom
ma3 = numpy.array([[9,0,5,2,1],
                   [3,9,1,2,1],
                   [0,1,7,2,3],
                   [4,2,3,12,2],
                   [3,2,4,0,8]])

leng2=ma3.shape[0]

# Checking & Comparing pivots with sums of rows not including pivots
for i in range(leng2):
    pivot=abs(ma3[i,i])

    nonPivSum=sum(abs(ma3[i, j]) for j in range(leng2) if j != i)
    
    if pivot < nonPivSum:
        diagDom = False  
        break  

# Row 5 was the one that caused it to not be diag dom

if diagDom:
    print("\nThe matrix is diagonally dominant")
else:
    print("\nThe matrix is not diagonally dominant")




#       4 Checking for pos def

ma4 = numpy.array([[2,2,1],[2,3,0],[1,0,2]])

# Check if symmetric
checkSymm=numpy.array_equal(ma4,ma4.T)

# Check eigenvals
eigen=numpy.linalg.eigvals(ma4)
if numpy.all(eigen>0):
    checkEigen=True

if checkSymm==True and checkEigen==True:
    print("\nThe matrix is positive definite")
else:
    print("\nThe matrix is not positive definite")



