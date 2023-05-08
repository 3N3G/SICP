#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:41:21 2023

@author: gene
"""

"""
Exercise 1.1
10
12
8
3
6


19
False
4
16
6
16

Exercise 1.2
/ (+ 5 4 (- 2 (- 3 (+ 6 (/ 4 5))))) (* 3 (- 6 2) (- 2 7))

Exercise 1.3
define (f a b c) (cond (and (< a b) (< a c)) (+ (* b b) (* c c)) (and (< b a) (< b c)) (+ (* a a) (* c c)) (else (+ (* a a) (* b b))))
^ could use sum-of-squares and stuff as well

Exercise 1.4
If b is greater than 0 then the operation is plus, otherwise it is minus. the procedure then returns a [operation] b

Exercise 1.5
If the interpreter uses applicative-order evaluation then the procedure will just return zero. If it uses normal-order evaluation, then the interpreter will try to figure out what y is, which is p, which loops back to itself infinitely, causing a dead loop

Exercise 1.6
When Alyssa attempts to use this to compute square roots, it may get stuck in an infinite loop, as the third argument of new-if is another sqrt-iter, meaning the computer will get stuck in a recursive unending cycle. "if" doesn't have this problem, since if the condition is true it doesn't check the false case.

Exercise 1.7
Because good-enough uses a constant error, it will fail for small numbers. If, say, the error allowance is 0.001, then if the answer is 0.00001, then a guess could be 100 times the correct answer and good-enough would still return true! For large numbers, any iterative process may go in jumps too large for good enough. A guess may cycle between 100 and 101, always more than the error of 0.01 from the correct answer 100.5
Here is an implemented square root function that uses the decreasing change in "guess" to determine if the guess is good enough:
"""
def goodenough (guess, x, lastguess, acceptederror):
    return (abs((guess - lastguess)/guess) < acceptederror)

def sqriter (guess, x, lastguess, acceptederror):
    # print(str(guess) + ", " + str(x) + ", " + str(lastguess))
    if (guess == 0):
        return 1 # not super sure why 1 should be returned when guess is 0
    if (goodenough(guess, x, lastguess, acceptederror)):
        return guess
    else:
        return sqriter(0.5*(guess+x/guess), x, guess)
        
def sqrter (x):
    if (x<0):
        print("No square rooting negatives")
        return
    elif (x == 0):
        return 0
    return sqriter(1,x,2*x, 0.01)

#print(sqrter(19239))  # correct is 138.7
#print(sqrter(0.0001)) # correct is 0.01

"""
Exercise 1.8
"""
def cbiter (guess, x, lastguess, acceptederror): # cube root is defined for negative numbers
    if (goodenough(guess, x, lastguess, acceptederror)):
        return guess
    else:
        return cbiter((2*guess+x/guess**2)/3, x, guess, acceptederror)
def cbrter (x):
    return cbiter(1,x,2*x, 0.01)
#print(cbrter(27))
#print(cbrter(-1))

"""
Exercise 1.9
Method one: (+ 4 5) -> inc (+ 3 5) -> inc(inc(+ 2 5)) -> inc(inc(inc(+ 1 5))) -> inc(inc(inc(inc(+ 0 5)))) -> inc^4(5) -> inc^3(6) -> inc^2(7) -> inc(8) -> inc(9)
Method two: (+ 4 5) -> (+ 3 6) -> (+ 2 7) -> (+ 1 8) -> (+ 0 9) -> 9

Recursive since it calls itself
"""

"""
Exercise 1.10

Hand Calculation:
A(1,10) = A(0,A(1,9)) = 2*A(1,9) = 4*A(1,8) ... = 512*A(1,1) = 1024
A(2,4) = A(1,A(2,3)) = 2**A(2,3) = 2**2**A(2,2) = 2**2**2**A(2,1) = 2**2**2**2 = 2^16 = 65536
A(3,3) = A(2,A(3,2)) = A(2,A(2,A(3,1))) = A(2,A(2,2)) = A(2,4) = 65536

Just to check:
"""
def A(x,y):
    if y==0:
        return 0
    if x==0:
        return 2*y
    if y==1:
        return 2
    else:
        return A(x-1, A(x,y-1))

# print(A(1,10))
# print(A(2,4))
# print(A(3,3))

"""
Mathematical expressions:
(define (f n) (A 0 n)) = 2*n
(define (g n) (A 1 n)) = 2**n
(define (h n) (A 2 n)) = 2**2**2...**2 n times
(define (k n) (* 5 n n)) = 5n^2
"""
        
"""
Exercise 1.11
"""
def recursivef(n):
    if n<3:
        return n
    return recursivef(n-1)+2*recursivef(n-2)+3*recursivef(n-3)

"""
First a recursive program similar to one in the book that would be considered as iterative as lisp can do, except in Python:
"""
def fiter(n,i,oneprev,twoprev,threeprev):
    if n==i:
        return oneprev + 2*twoprev + 3*threeprev
    else:
        return fiter(n, i+1, oneprev + 2*twoprev + 3*threeprev, oneprev, twoprev)
    
def iterativef(n):
    if n<3:
        return n
    return fiter(n,3,2,1,0)
"""
Actual iterative program:
"""



"""
Exercise 1.12
"""
def recursivePascal(i,j):
    if i == 0: # first row
        return 1
    if j == 0: # left column
        return 1
    if i == j:
        return 1
    if i > j or i < 0 or j < 0:
        return 0
    else:
        return recursivePascal(i-1,j-1) + recursivePascal(i-1,j)

#print(recursivePascal(6,3))

"""
Exercise 1.13

First it is easy to check that the base cases work: Fib(0) and Fib(1) both do what they're supposed to
Now assume that Fib(k)=(phi^k-psi^k)/sqrt(5) for all k from 1 to n-1. 
Now, Fib(n) = Fib(n-1)+Fib(n-2) by definition. Plugging in the formula from the inductive hypothesis, we get:
    Fib(n) = (phi^(n-1)+phi^(n-2)-psi^(n-1)-psi^(n-2))/sqrt(5). Expand and simplify and we get 
    Fib(n) = (phi^n-psi^n)/sqrt(5), as desired. Now since psi<1 as n gets large psi goes to zero, and since all Fibonacci numbers are the sum of integers and therefore integers, Fib(n) must be the closest integer to phi^n/sqrt(5)
"""

"""
Exercise 1.14

count-change(11) = count-change(11 without pennies) + count-change(10) = count-change(10) = count-change(10 without pennies) + count-change(9) = count-change(5 without pennies) + count-change(9) = 1 + count-change(9) = 1 + count-change(8) ... = 1 + count-change(5) = 2
"""

"""
Exercise 1.15
a) 7 times because 12.15/3^7<0.01 but 12.15/3^6>0.01
b) on the order of log(a)
"""

"""
Exercise 1.16

"""
    
def powiter(b,n,a):
    if (n == 1):
        return b*a
    if (n%2 == 0):
        return powiter(b*b,n/2,a)
    return powiter(b*b,(n-1)/2,a*b)

def power(b,n):
    return powiter(b,n,1)

# print(power(-5,4))

"""
Exercise 1.17
"""

def halve(n):
    return n/2
def double(n):
    return 2*n
def iseven(n):
    return (n%2 == 0)
def negate(n):
    return 0-n

def fasttimes(a,b):
    if (a < 0):
        return negate(fasttimes(negate(a),b))
    if (b < 0):
        return negate(fasttimes(a,negate(b)))
    if (b == 0):
        return 0
    if (b == 1):
        return a
    if (iseven(b)):
        return double(fasttimes(a,halve(b)))
    else:
        return b + double(fasttimes(a,halve(b-1)))
    

"""
Exercise 1.18
"""

def betterpowiter(b,n,a):
    if (n == 1):
        return fasttimes(b,a)
    if (n%2 == 0):
        return powiter(fasttimes(b,b),halve(n),a)
    return powiter(fasttimes(b,b),halve(n-1),fasttimes(a,b))

def betterpower(b,n):
    return powiter(b,n,1)

"""
Exercise 1.19

a ← bq + aq + ap and b ← bp + aq twice is equal to
a is now (bp+aq)q+(bq+aq+ap)(p+q) = b (q^2+2pq) + a(2q^2+p^2+2pq)
b is now p(bp+aq) + q(bq+aq+ap) = b(p^2+q^2) + a(q^2+2pq)

t(t(a,b)) using p,q is the same as t(a,b) using p^2+q^2, q^2+2pq

"""
def transform(p,q):
    return lambda pair : (pair[1]*q+pair[0]*p+pair[0]*q, pair[1]*p+pair[0]*q)


def genfibiter(a,b,p,q,count):
    if (count == 0):
        return b
    if (iseven(count)):
        return genfibiter(a,b,p**2+q**2,q**2+2*p*q,count/2)
    else:
        return genfibiter(b*q+a*p+a*q, b*p+a*q, p, q, count-1)

def genfib (n, p, q):
    return genfibiter(1,0,p,q,n)
    

for i in range(10):
    print(genfib(i,0,1))




