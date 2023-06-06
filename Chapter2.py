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

printRat(mulRat(third,nhalf))


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
print(area(rect))

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

print(car(cons(1,2)))

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
one = lambda f: lambda x: f(f(x))
two = lambda f: lambda x: f(f(f(x)))

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
def div_interval: #assume alyssas program
    return

def divide_interval(x,y):
    if (lower_bound(x)*upper_bound(x)<=0 or lower_bound(y)*upper_bound(y)>=0):
        raise Exception("No intervals cross zero")
    return div_interval(x,y)

"""
Exercise 2.11
"""












