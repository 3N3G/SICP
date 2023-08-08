#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:47:12 2023

@author: gene
"""

"""
Exercise 2.1
"""
# class version
class Fraction:
    def __init__(self, numer, denom):
        self.numer = numer
        self.denom = denom

def add_rat(x,y):
    return Fraction(x.numer*y.denom+y.numer*x.denom,x.denom*y.denom)
    
def sub_rat(x,y):
    return Fraction(x.numer*y.denom - x.denom*y.numer, x.denom*y.denom)

def mul_rat(x,y):
    return Fraction(x.numer*y.numer, x.denom*y.denom)

def div_rat(x,y):
    return Fraction(x.numer*y.denom, x.denom*y.numer)

def equal_rat(x,y):
    return x.numer*y.denom == x.denom*y.numer


# sicp version
import math

def make_rat(n, d):
    if n*d<0:
        return (-1*abs(int(n/math.gcd(n,d))),abs(int(d/math.gcd(n,d))))
    else:
        return (abs(int(n/math.gcd(n,d))),abs(int(d/math.gcd(n,d))))
def numer(x):
    return x[0]
def denom(x):
    return x[1]

def addRat(x,y):
    return make_rat(numer(x)*denom(y)+numer(y)*denom(x),denom(x)*denom(y))

def subRat(x,y):
    return make_rat(numer(x)*denom(y)-numer(y)*denom(x),denom(x)*denom(y))

def mulRat(x,y):
    return make_rat(numer(x)*numer(y),denom(x)*denom(y))

def divRat(x,y):
    return make_rat(numer(x)*denom(y),denom(x)*numer(y))

def printRat(x):
    print(numer(x),"/",denom(x),sep='') #sep is to remove the spaces in between the numbers and the /

third = make_rat(1,3)
nhalf = make_rat(1,-2)

# printRat(mulRat(third,nhalf))


"""
Exercise 1.2
"""

def make_point(x,y):
    return (x,y)

def x_point(p):
    return p[0]

def y_point(p):
    return p[1]

def dist(p1,p2):
    return math.sqrt((x_point(p1)-x_point(p2))**2+(y_point(p1)-y_point(p2))**2)

def make_segment(x,y):
    return (x,y)

def start_segment(s):
    return s[0]

def end_segment(s):
    return s[1]

def len_segment(s):
    return dist(s[0],s[1])

def midpoint_segment(s):
    return make_point((x_point(start_segment(s))+x_point(end_segment(s)))/2, (y_point(start_segment(s))+y_point(end_segment(s)))/2)

def print_point(p):
    print("(",x_point(p),",",y_point(p),")",sep='')
    
def add_point(p1,p2):
    return (p1[0]+p2[0],p1[1]+p2[1])

def scale_point(p,l):
    return (l*p[0], l*p[1])

def rotate(p):
    return (-p[1],p[0])

"""
Exercise 2.3
"""
def make_rect(p1,p2,p3,p4):
    return (p1,p2,p3,p4)

def p1(r):
    r[0]
def p2(r):
    r[1]
def p3(r):
    r[2]
def p4(r):
    r[3]

def perim(r):
    return dist(p1(r),p2(r))+dist(p2(r),p3(r))+dist(p3(r),p4(r))+dist(p4(r),p1(r))

def area(r):
    return dist(r[0],r[1])*dist(r[2],r[3])

r1 = make_point(0,0)
r2 = make_point(8,6)
r3 = make_point(5,10)
r4 = make_point(-3,4)
rect = make_rect(r1,r2,r3,r4)
# print(area(rect))

def make_rect2(p1,p2,l): # so that the rectangle has p1 then p2 in clockwise order
    return (p1,p2,l)
def po1(r):
    r[0]
def po2(r):
    r[1]
def po3(r):
    add_point(r[1], rotate(scale_point(add_point(scale_point(r[0],-1), r[1]),r[2]/dist(r[0],r[1]))))
def po4(r):
    add_point(r[0], rotate(scale_point(add_point(scale_point(r[0],-1), r[1]),r[2]/dist(r[0],r[1]))))

def perim2(r):
    return dist(po1(r),po2(r))+dist(po2(r),po3(r))+dist(po3(r),po4(r))+dist(po4(r),po1(r))

def area2(r):
    return dist(po1(r),po2(r))*dist(po2(r),po3(r))

"""
Exercise 2.4
"""
def cons(x,y):
    return lambda m: m(x,y)
def car(z):
    return z(lambda x,y: x)
def cdr(z):
    return z(lambda x,y: y)

# print(car(cons(1,2)))

"""
Exercise 2.5
"""

def cons_ex5(x,y):
    return 2**x*3**y

def powerof(p, n):
    if (n % p == 0):
        return powerof(p, n/p)
    else:
        return 0
def car_ex5(z):
    return powerof(2,z)
def cdr_ex5(z):
    return powerof(3,z)


"""
Exercise 2.6
"""

zero = lambda f: lambda x: x

def add_one(n):
    return lambda f: lambda x: f(n(f)(x))

#one = add_one(zero) = lambda f: lambda x: f(f(x))
# one = lambda f: lambda x: f(f(x))
# two = lambda f: lambda x: f(f(f(x)))

def add(m,n):
    return lambda f: lambda x: m(f)(n(f)(x))

"""
Exercise 2.7
"""
def make_interval(a,b):
    return (a,b)

def lower_bound(x):
    return x[0]
def upper_bound(x):
    return x[1]

"""
Exercise 2.8
"""

def sub_interval(x,y): # interval that is x-y
    return (lower_bound(x)-upper_bound(y), upper_bound(x)-lower_bound(y))


"""
Exercise 2.9

Width of a sum only depends on the width of the summands:
    width([a,b]+[c,d]) = width([a+c,b+d]) = ((b+d)-(a+c))/2 = width([a,b])+width([c,d])

However, this is not true for multiplication because there is an "or" function:
    width([a,b]*[c,d]) != width([a+10,b+10]*[c,d])
Because division uses multiplication it must not be true for division either
"""


"""
Exercise 2.10
"""
def div_interval(): #assume alyssas program
    return #alyssas program

def divide_interval(x,y):
    if (lower_bound(x)*upper_bound(x)<=0 or lower_bound(y)*upper_bound(y)>=0):
        raise Exception("No intervals cross zero")
    return div_interval(x,y)

"""
Exercise 2.17
"""
def last_pair(l):
    if (isinstance(l, int)):
        return l
    else:
        return last_pair(l[1])


# seq = (1,(2,(3,4)))
# print(last_pair(seq))


"""
Exercise 2.18
"""
def reverse(l):
    if isinstance(l[1], int):
        return (l[1],l[0])
    return (reverse(l[1]),l[0])
    
# print(reverse(seq))


"""
Exercise 2.20

Lisp specific
"""

"""
Exercise 2.21
"""
def maplist(procedure, thelist):
    if len(thelist)==0:
        return []
    return [procedure(thelist[0])] + (maplist(procedure, thelist[1:]))

# print(maplist(lambda x: x**2, [-1,0,1,2,3]))

"""
Exercise 2.22

He's iterating through the list in the wrong order in the first one. it takes things fron the end (cdr) and puts them at the front of answer (car)

the second approach cons a list and an element together, so it would be like {{1,2},{3}}
"""


"""
Exercise 2.23

define (foreach function list) (if (null? list) True (function(car(list)) foreach function cdr(list)))
"""

def for_each(function, listA):
    for i in listA:
        function(i)

"""
Exercise 2.24

{1,{2,{3,4}}}
(list 1 (list 2 (list 3 4)))
(1(2(3 4))
| \
1 (2(3 4)
    /  \
   2   (3 4)
        /  \
        3  4
        
"""
"""
Exercise 2.25
x1 = (1 3 (5 7) 9)
x2 = ((7))
x3 = (1 (2 (3 (4 (5 (6 7))))))

cdr(car(cdr(cdr(x1)))) = 7
car(car(x2)) = 7
cdr(cdr(cdr(cdr(cdr(cdr(x3))))))=7
"""

"""
Exercise 2.26

1)
    (1 2 3 4 5 6)
2)
    ((1 2 3) 4 5 6)
3)
    (1 2 3) (4 5 6)
"""

"""
Exercise 2.27
"""
def deep_reverse(l):
    l2 = []
    if isinstance(l, int):
            return l
        
    for i in range(len(l)):
        l2.append(deep_reverse(l[len(l)-1-i]))

    return l2


# testl = [1,2,[3,4]]
# print(deep_reverse(testl))

"""
Exercise 2.28
"""

def fringe(l):
    if (not l):
        return []
    if isinstance(l, int):
        return [l]
    first = l[0]
    rest = l[1:]
    return fringe(first) + fringe(rest)

# testlist = [1,2,[3,4]]
# print(fringe(testlist))

"""
Exercise 2.29
"""

def make_mobile(left,right):
    return (left,right)

def left_branch(mobile):
    return mobile[0]

def right_branch(mobile):
    return mobile[1]

def make_branch(length, structure):
    return (length, structure)

def branch_length(branch):
    return branch[0]

def branch_structure(branch):
    return branch[1]

def total_weight(mobile):
    if isinstance(mobile, int):
        return mobile
    if (not mobile):
        return 0
    return total_weight(branch_structure(left_branch(mobile))) + total_weight(branch_structure(right_branch(mobile)))

"""
b1 = make_branch(1,2)
b2 = make_branch(2,3)
b3 = make_branch(3,make_mobile(b2,b2))
m1 = make_mobile(b1,b3)
print(total_weight(m1))
"""

def naivelyBalanced(mobile):
    return branch_length(right_branch(mobile))*total_weight(branch_structure(right_branch(mobile))) == branch_length(left_branch(mobile))*total_weight(branch_structure(left_branch(mobile)))

def isBalanced(mobile):
    if isinstance(mobile, int):
        return True
    if (not mobile):
        return True
    return isBalanced(branch_structure(left_branch(mobile))) and isBalanced(branch_structure(right_branch(mobile))) and branch_length(right_branch(mobile))*total_weight(branch_structure(right_branch(mobile))) == branch_length(left_branch(mobile))*total_weight(branch_structure(left_branch(mobile)))

# mobiletest = make_mobile(make_branch(4,6),make_branch(2,make_mobile(make_branch(5,8),make_branch(10,4))))
# print(isBalanced(mobiletest))

"""
d)
Not much work at all. As long as the left_branch, right_branch, branch_length and branch_structure are changed, the other methods will still work.
"""

"""
Exercise 2.30, 2.31
"""
def mapTree(tree, map):
    if not tree:
        return []
    if isinstance(tree, int):
        return map(tree)
    return [mapTree(tree[0],map)] + [mapTree(tree[1],map)]

def squareTree(tree):
    return mapTree(tree, lambda x: x**2)

"""
Exercise 2.32
"""
def powerset(S):
    if not S:
        return [[]]
    rest = powerset(S[1:])
    return rest + maplist(lambda x: [S[0]]+x, rest)
# Two copies of the powerset of S without first element: one with it and one without it

"""
Exercise 2.33
"""
def accumulate(operation, initial, sequence):
    if not sequence:
        return initial
    return operation(sequence[0], accumulate(operation, initial, sequence[1:]))

def map(p, sequence):
    def op(x,y):
        return [p(x)]+y
    return accumulate(op, [], sequence)
def append(seq1, seq2):
    def op(x,y):
        return x+y
    return accumulate(op, seq1, seq2)
def length(sequence):
    def op(x,y):
        return y+1
    return accumulate(op, 0, sequence)
testlength = [1,2,3,[4,5]]
# print(length(testlength))


def testp(x):
    return 2*x
testseq = [1,2,3,4]

# print(map(testp, testseq))

"""
Exercise 2.34
"""
def hornerEval(x, coefficientSequence):
    return accumulate(lambda thisCoeff, higherTerms: thisCoeff + x*higherTerms, 0, coefficientSequence)

"""
Exercise 2.35
"""
def sum(a,b):
    return a+b
def countLeaves(tree):
    def func(x):
        if isinstance(x,int):
            return 1
        return countLeaves(x)
    return accumulate(sum,0,map(func,tree))



testcountleaves = [1,2,3,[1,2,3,4]]

# print(length(testcountleaves))
# print(countLeaves(testcountleaves))

"""
Exercise 2.36
"""

def carofseqs(sequences):
    list = []
    for i in sequences:
        list = list + [i[0]]
    return list
def cdrofseqs(sequences):
    list = []
    for i in sequences:
        list = list + [i[1:]]
    return list

def accumulate_n(operation, init, sequences):
    if not sequences[0]:
        return []
    return [accumulate(operation, init, carofseqs(sequences))] + accumulate_n(operation, init, cdrofseqs(sequences))

testaccumulaten = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
# print(accumulate_n(sum, 0, testaccumulaten))

"""
Exercise 2.37
"""
def dotproduct(v,w):
    list = []
    for i in range(len(v)):
        list = list + [v[i]*w[i]]
    return accumulate(lambda x,y: x+y, 0, list)

testdotproduct1 = [1,2,3]
testdotproduct2 = [4,5,6]
# print(dotproduct(testdotproduct1, testdotproduct2))

def matrixtimesvector(A,x):
    return map(lambda y: dotproduct(y,x), A)

def transpose(A):
    return accumulate_n(lambda x,y: [x]+y, [], A)

# testmatrix = [[1,2,3],[4,5,6],[7,8,9]]
# print(transpose(testmatrix))

def matrixmult(A,B):
    return map(lambda x: matrixtimesvector(B,x), transpose(A))

testmatrix1 = [[1,2,3],[4,5,6],[7,8,9]]
testmatrix2 = [[1,2,3],[4,5,6],[7,8,9]]
# print(matrixmult(testmatrix1, testmatrix2))

"""
Exercise 2.38

3/2
1/6
[1,[2,[3,[]]]
[[[],1],2],3]
associativity
"""
def fold_left(op,init,seq):
    if not seq:
        return init
    return fold_left(op, op(init, seq[0]), seq[1:])
"""
Exercise 2.39
"""
def reverse1(seq):
    return accumulate(lambda x,y: y+[x], [], seq)
def reverse2(seq):
    return fold_left(lambda x,y: [y]+x, [], seq)

testreverse = [1,2,3,4,5]
# print(reverse1(testreverse))
# print(reverse2(testreverse))

"""
Exercise 2.40
"""
def unique_pairs(n):
    if n < 0:
        return []
    
    sequence = []
    for i in range(n):
        for j in range(i):
            sequence = sequence + [[i,j]]
    return sequence

def filter(condition, sequence):
    seq2 = []
    for i in sequence:
        if condition(i):
            seq2 = seq2 + [i]
    return seq2
def prime_sum_pairs(n):
    sequence = unique_pairs(n)
    return filter(lambda x: isPrime(x[0]+x[1]), sequence)

"""
Exercise 2.41
"""
def ordered_triples(n,s):
    triples = []
    for i in range(n):
        for pair in unique_pairs(s):
            triples = triples + [[i]+pair]
            triples = triples + [pair + [i]]
    return triples

"""
Exercise 2.42
"""

def isSafe(k, positions):
    safe = True
    for i in range(k-1):
        if positions[i]==positions[k-1]:
            safe = False
        if positions[i]-positions[k-1]==i-k+1:
            safe = False
    return safe
def queen_cols(k):
    if k == 0:
        return [[]]
    trylist = [[]]
    for i in range(k):
        trylist = trylist + map(lambda x: x+[i], queen_cols(k-1))
    #take list of queen_cols(k-1) and make a list with the kth row queen being each of 0 to k-1
    #filter this list 
    return filter (lambda y : isSafe(k,y), trylist)# list of possible placements for kth queen)

def queens(board_size):
    return queen_cols(board_size)

"""
Exercise 2.43

This runs much more slowly since it repeats queen-cols again and again, redundantly. It should be 8^8 times slower.
"""


"""
Exercise 2.44
(define up-split painter n) (if (= n 0) painter (below (beside (up-split painter (- n 1)) (up-split painter (- n 1))) painter))
"""

"""
Exercise 2.45
(define split first second) (if (= n 0) painter (first (second (split painter (- n 1)) (split painter (- n 1))) painter))
"""

"""
Exercise 2.46
"""
def make_vect(x,y):
    return (x,y)

def xcor_vect(v):
    return v[0]
def ycor_vect(v):
    return v[1]

def add_vect(v1,v2):
    return make_vect(xcor_vect(v1)+xcor_vect(v2), ycor_vect(v1)+ycor_vect(v2))

def sub_vect(v1,v2):
    return make_vect(xcor_vect(v1)-xcor_vect(v2), ycor_vect(v1)-ycor_vect(v2))

def scale_vect(v,s):
    return make_vect(s*xcor_vect(v), s*ycor_vect(v))

"""
Exercise 2.47
"""
def make_frame(origin, edge1, edge2):
    return [origin, edge1, edge2]
def make_frame2(origin, edge1, edge2):
    return [origin, [edge1, edge2]]

frameconstructor = make_frame #can switch to make_frame2

def origin_frame(frame):
    return frame[0]
    # return frame[0][0]
def edge1_frame(frame):
    return frame[1]
    # return frame[1][0]
def edge2_frame(frame):
    return frame[2]
    # return frame[1][1]

"""
Exercise 2.48
"""

def make_segment(start, end): # directed segment
    return [start, end]
def start_segment(segment):
    return segment[0]
def end_segment(segment):
    return segment[1]

"""
Exercise 2.49
"""
v1 = make_vect(0,0)
v2 = make_vect(1,0)
v3 = make_vect(0,1)
v4 = make_vect(1,1)

# pretend draw_line exists and draws a line

def segmentstopainters(segment_list):
    def function(frame):
        for segment in segment_list: 
            draw_line(frame_coord_map(frame)(start_segment(segment)), frame_coord_map(frame)(end_segment(segment)))
    return function

# a)
outline_paintera = segmentstopainters([make_segment(v1,v2),make_segment(v2,v4),make_segment(v4,v3),make_segment(v3,v1)])
# b)
outline_painterb = segmentstopainters([make_segment(v1,v4),make_segment(v2,v3)])
# c)
outline_painterc = segmentstopainters([make_segment(scale_vect(add_vect(v3,v1),.5),scale_vect(add_vect(v1,v2),.5)),make_segment(scale_vect(add_vect(v1,v2),.5),scale_vect(add_vect(v2,v4),.5)),make_segment(scale_vect(add_vect(v2,v4),.5),scale_vect(add_vect(v4,v3),.5)),make_segment(scale_vect(add_vect(v4,v3),.5),scale_vect(add_vect(v3,v1),.5))])
# d) is pretty impossible without manually recording all the line segments that make up the wave image

"""
Exercise 2.50
"""

def transform_painter(painter, origin, corner1, corner2):
    def transformation(frame):
        m = frame_coord_map(frame)
        new_origin = m(origin)
        return painter(make_frame(new_origin, sub_vect(m(corner1), new_origin), sub_vect(m(corner2), new_origin)))
    return transformation

def flip_vert(painter):
    return transform_painter(painter, make_vect(0,1),make_vect(1,1),make_vect(0,0))

def flip_horiz(painter):
    return transform_painter(painter, make_vect(1,0),make_vect(0,0),make_vect(1,1))

def rotate90(painter): #counterclockwise
    return transform_painter(painter, make_vect(1,0),make_vect(1,1),make_vect(0,0))

def rotate180(painter):
    def function(frame):
        return rotate90(rotate90(painter))(frame)
    return function

def rotate270(painter):
    def function(frame):
        return rotate90(rotate90(rotate90(painter)))(frame)
    return function

"""
Exercise 2.51
"""
def below1(painter1, painter2):
    split_point = make_vect(0,0.5)
    paint_left = transform_painter(painter1, make_vect(0,0),make_vect(1,0),split_point)
    paint_right = transform_painter(painter2, split_point, make_vect(1,0.5),make_vect(0,1))
    
    def framefunc(frame):
        paint_left(frame)
        paint_right(frame)
    return framefunc

def below2(painter1, painter2): # assumes beside exists
    return rotate270(beside(rotate90(painter1), rotate90(painter2)))

"""
Exercise 2.53
(a b c)
((george))
((y1 y2))??
(y1 y2)
False
False
(red shoes blue socks)
"""

"""
Exercise 2.54
"""
def equal(list1, list2):
    if isinstance(list1,int) or isinstance(list2,int):
        return list1 == list2
    elif not list1 and not list2:
        return True
    elif not (list1 and list2):
        return False
    elif len(list1)==1 and len(list2)==1:
        return list1[0] == list2[0]
    return equal(list1[0],list2[0]) and equal(list1[1:],list2[1:])

# print(equal((1,2,3,4),(1,2,3,4)))
# print(equal((1,2,3,4),(1,2,3,5)))
# print(equal((1,2,3,4),(1,2,(3,4))))

"""
Exercise 2.55

This is because ''abracadabra is the item (quote (quote abracadabra)) and the first quote makes the things in the parentheses literally the list (quote abracadabra), thus car of it is quote.
"""

"""
Exercise 2.56, 2.57, 2.58
General differentiator using (+, a, b, c, ...) and (*, a, b, c, ...) notation. still should implement (a+b) notation
"""

def isVariable(x):
    return isinstance(x,str) and len(x) == 1 # maybe need better implementation

def isSameVariable(x,y):
    return isVariable(x) and isVariable(y) and x==y

def make_sum(a1,a2):
    if not isinstance(a1,tuple) and not isinstance(a2,tuple):
        if (a1.isnumeric() and a2.isnumeric()):
            return str(int(a1)+int(a2))
    elif a1 == "0":
        return a2
    elif a2 == "0":
        return a1
    return ('+',a1,a2)

def make_product(a1,a2):
    if (a1 == "0" or a2 == "0"):
        return "0"
    elif (a1 == "1"):
        return a2
    elif (a2 == "1"):
        return a1
    elif not isinstance(a1,tuple) and not isinstance(a2,tuple):

        if (a1.isnumeric() and a2.isnumeric()):
            return str(int(a1)*int(a2))
    return ('*',a1,a2)

def is_sum(x):
    return isinstance(x,tuple) and x[0]=='+'

def addend(s): #first thing summed
    return s[1]

def augend(s):
    return accumulate(make_sum, "0", s[2:])

def is_prod(x):
    return isinstance(x,tuple) and x[0]=='*'

def multiplier(p): #first thing multiplied
    return p[1]

def multiplicand(p): #second thing multiplied
    return accumulate(make_product, "1", p[2:])

def deriv(exp,var):
    if not isinstance(exp,tuple):
        if exp.isnumeric(): 
            return "0"
        if isVariable(exp):
            if isSameVariable(exp,var):
                return "1"
            else:
                return "0"
    if is_sum(exp):
        return make_sum(deriv(addend(exp),var), deriv(augend(exp),var))
    if is_prod(exp):
        return make_sum(make_product(multiplier(exp),deriv(multiplicand(exp),var)), make_product(deriv(multiplier(exp),var),multiplicand(exp)))

print(deriv(("+",("*","x","x","x"),("*","2","x"),"1"),"x"))

print(deriv("x","x"))

"""
Exercise 2.59
"""
def union_set(s1,s2):
    unionset = s1
    for i in s2:
        adjoin_set(i,s2)
    return unionset

"""
Exercise 2.60
"""
def is_element(S,e):
    if not S:
        return False
    if S[0]==e:
        return True
    return is_element(S[1:],e) # just as efficient except probably slower realistically
def adjoin_set(S,e):
    return S+[e] # nominally more efficient
def intersection_set(S1,S2):
    filter(lambda x: is_element(S2,x), S1) # just as efficient
def union_set(S1,S2):
    return S1+S2 # more efficient

"""
Exercise 2.61
"""
def is_element_of_ordered_set(S,e):
    if not S:
        return False
    if S[0]==e:
        return True
    if S[0]>e:
        return False
    return is_element(S[1:],e)

def adjoin_set(S,e):
    if not S:
        return [e]
    if S[0]==e:
        return S
    if S[0]>e:
        return [e]+S
    return S[0]+adjoin_set(S[1:],e)
# faster by half
def union_set(S1,S2):
    if not S1:
        return S2
    if not S2:
        return S1
    if S1[0]==S2[0]:
        return S1[0]+union_set(S1[1:],S2[1:])
    if S1[0]>S2[0]:
        return S2[0]+union_set(S1,S2[1:])
    return S1[0]+union_set(S1[1:],S2)
# faster by half

"""
Exercise 2.62
"""
