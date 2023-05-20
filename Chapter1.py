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
        return sqriter(0.5*(guess+x/guess), x, guess, acceptederror)
        
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
def f(n):
    if (n < 3):
        return n
    oneprev = 2
    twoprev = 1
    threeprev = 0
    for i in range(n-2):
        oneprevnew = oneprev + 2 * twoprev + 3 * threeprev
        threeprev = twoprev
        twoprev = oneprev
        oneprev = oneprevnew
    return oneprev

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

def iterpower(b,n):
    expleft = n
    val = b
    while (expleft > 0):
        if (expleft % 2 == 0):
           val = val * val
           expleft = expleft / 2
        elif (expleft > 1):
            val = val * b
            expleft = expleft - 1
        else:
            break
    return val
            
# print(power(-5,4), iterpower(-5,4))

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
        return genfibiter(a,b,p**2+q**2,q**2+2*p*q,halve(count))
    else:
        return genfibiter(transform(p,q)((a,b))[0], transform(p,q)((a,b))[1], p, q, count-1)

def genfib (n, p, q):
    return genfibiter(1,0,p,q,n)
    
"""
Test:
    for i in range(10):
    print(genfib(i,0,1))
"""

"""
Exercise 1.20
substitution method (normal order):
    gcd(206,40)
    = gcd(40, remainder(206,40))
    = gcd(40, 6)
    = gcd(6, remainder(40,6))
    = gcd(6, 4)
    = gcd(4, remainder(6,4))
    = gcd(4, 2)
    = gcd(2, remainder(4,2))
    = gcd(2, 0)
    = 2
    
4 evaluations of remainder function

applicative-order:
    gcd(206,40)
    = gcd(40, remainder(206,40))
    = gcd(remainder(206,40), remainder(remainder(206,40),40))
    = ...
    
May never end
"""

"""
Exercise 1.21
"""

def finddivisor(n, testdivisor):
    if (testdivisor**2 > n):
        return n
    elif (n % testdivisor == 0):
        return testdivisor
    return finddivisor(n, testdivisor+1)

def smallestdivisor(n):
    return finddivisor(n,2)
# nb I actually forgot to put return before finddivisor in this function so the code wasn't working and I asked chatgpt and it caught my silly mistake
# print(smallestdivisor(199))
# print(smallestdivisor(1999))
# print(smallestdivisor(19999))

"""
Exercise 1.22
"""
import time

def isprime(n):
    return smallestdivisor(n) == n

def timed_prime_test(n):
    numprimesgreaterthann = 0
    guess = n
    while (numprimesgreaterthann < 3):
        if isprime(guess):
            # print(guess)
            numprimesgreaterthann = numprimesgreaterthann + 1
        guess = guess + 1

def time_prime_test(n):
    start_time = time.time()
    timed_prime_test(n)
    end_time = time.time()
    runtime = end_time - start_time
    print("Runtime: ", runtime * 1000, "milliseconds")

"""
time_prime_test(100)
time_prime_test(1000)
time_prime_test(10000)
time_prime_test(100000)
"""

# seems that the time is roughly increasing by a factor of rt1. let's check how linear it is
# with chatgpt's help:
import numpy as np
from scipy.stats import pearsonr
# Input data
x = np.array([1, 2, 3, 4])
y = np.log10(np.array([0.05507469177246094, 0.1659393310546875, 0.7359981536865234, 2.375364303588867]))
corr, _ = pearsonr(x, y)
# print("Pearson correlation coefficient: {:.3f}".format(corr))

"""
Exercise 1.23
"""

def next(n):
    if (n == 2):
        return 3
    else:
        return n + 2
    
def finddivisor2(n, testdivisor):
    if (testdivisor**2 > n):
        return n
    elif (n % testdivisor == 0):
        return testdivisor
    return finddivisor(n, next(testdivisor))

def smallestdivisor2(n):
    return finddivisor(n,2)

def isprimenew(n):
    return smallestdivisor2 == n

def testtime(f, args):
    start_time = time.time()
    f(args)
    end_time = time.time()
    print("Runtime: ", 1000*(end_time - start_time), " milliseconds")

"""
testtime(isprimenew, 100003)
testtime(isprime, 100003)
"""

"""
Exercise 1.24
"""
import random
def expmod (base, exp, mod):
    if (exp == 0):
        return 1
    elif (exp % 2 == 0):
        return (expmod(base, exp/2, mod)**2) % mod
    else:
        return (base*expmod(base, exp - 1, mod)) % mod

def fermattest (n):
    def tryn (a):
        return (expmod(a,n,n) == a)
    tryn(random.randint(1,n-1))

def fastprime (n, times):
    if (times == 0):
        return True
    if (fermattest(n)):
        fastprime(n, times - 1)
    else:
        return False

def testprimefast(n):
    return fastprime(n, int(n**(0.25)))

"""
testtime(testprimefast, 1000003)
testtime(testprimefast, 100003)
testtime(testprimefast, 10003)
testtime(testprimefast, 1003)
Roughly linear decrease which makes sense since exponential decrease in n -> linear decrease in log(n)
"""

"""
Exercise 1.25

The reason why we write expmod as its own method is to reduce the size of the numbers being computed. Expmod ensures that all multiplications result in at most a number of size n*n, whereas quickly using expfast will be unreasonable as n increases, and the computer has to compute arbitrarily large a^n
"""


"""
Exercise 1.26

The reason why this code is so much slower is because Louis is writing the product of two things. Although his intent is to square the value and they are the same, the compiler will calculate each independently, doubling the amount of work. However, it does this at each step of the recursion, increasing its work by a power of two. Thus the speed would be O(2^log(n)) ~ O(n)
"""

"""
Exercise 1.27
"""

def completefermattest (n):
    for i in range(n-1):
        if (expmod(i+1, n, n) != i+1):
            return False
    return True

"""
print(completefermattest(7)) # is a prime, test says yes
print(completefermattest(561)) # isn't a prime, test says yes
print(completefermattest(55)) # isn't a prime, test says no
"""

"""
Exercise 1.28
"""
def mrexpmod(base, exp, mod):
    if (exp == 0):
        return 1
    elif (exp % 2 == 0):
        beforesquare = (expmod(base, exp/2, mod))
        if (beforesquare != 1 and beforesquare != mod - 1):
            if (beforesquare ** 2 % mod == 1):
                return 0
        return beforesquare ** 2 % mod
    else:
        return (base*expmod(base, exp - 1, mod)) % mod


def mrtest (n):
    def tryn (a):
        return (mrexpmod(a,n,n) == a)
    tryn(random.randint(1,n-1))

def mrfastprime (n, times):
    if (times == 0):
        return True
    if (mrtest(n)):
        mrfastprime(n, times - 1)
    else:
        return False

def mrtestprimefast(n):
    return fastprime(n, int(n**(0.25)))

# print (mrtestprimefast(561))


"""
Exercise 1.29
"""

def summing (term, a, nex, b):
    if (a>b):
        return 0
    else:
        return term(a)+summing(term, nex(a), nex, b)

def integral_simpsons(f,a,b,n):
    if (n <= 0):
        return 0
    h = (b-a)/n        
    def thingtoadd(i):
        if i == 0:
            return f(a)
        elif i == n:
            return f(b)
        elif i % 2 == 0:
            return 2*f(a+i*h)
        else:
            return 4*f(a+i*h)
    return summing(thingtoadd, 0, lambda x: x+1, n) * h / 3


# print(integral_simpsons(lambda x: x**3, 0, 1, 100))
# print(integral_simpsons(lambda x: x**3, 0, 1, 1000))

"""
Exercise 1.30
"""

def itersum(term, a, nex, b):
    def iters(a, result):
        if (a > b):
            return 0
        if (a == b):
            return result
        else:
            # print("Calling iters on ", nex(a), result + term(a))
            return iters(nex(a), result + term(a))
    return iters(a, 0)

def iter_integral_simpsons(f,a,b,n):
    h = (b-a)/n        
    def thingtoadd(i):
        if i == 0:
            return f(a)
        elif i == n:
            return f(b)
        elif i % 2 == 0:
            return 2*f(a+i*h)
        else:
            return 4*f(a+i*h)
    return itersum(thingtoadd, 0, lambda x: x+1, n) * h / 3

# print(iter_integral_simpsons(lambda x: x**3, 0, 1, 100))
# print(iter_integral_simpsons(lambda x: x**3, 0, 1, 1000))

"""
Exercise 1.31
"""
def recursiveprod (term, a, nex, b):
    if (a>b):
        return 1
    else:
        return term(a)*recursiveprod(term, nex(a), nex, b)

def approxpi(n):
    finalmultiply = 4 # formula would otherwise return pi/4
    firstdenom = 3 # starts with 3
    jumpby = 2 # the denominators are every odd number
    return finalmultiply * recursiveprod(lambda x: (x-1)/x*(x+1)/x, firstdenom, lambda x: x+jumpby, n+jumpby)
"""
print(approxpi(10))
print(approxpi(100))
print(approxpi(1000))
"""

def iterprod (term, a, nex, b):
    marker = a
    prod = 1
    while (marker <= b):
        prod = prod * term(marker)
        marker = nex(marker)
    return prod

def iterapproxpi(n):
    finalmultiply = 4 # formula would otherwise return pi/4
    firstdenom = 3 # starts with 3
    jumpby = 2 # the denominators are every odd number
    return finalmultiply * iterprod(lambda x: (x-1)/x*(x+1)/x, firstdenom, lambda x: x+jumpby, n+jumpby)

"""
print(iterapproxpi(10))
print(iterapproxpi(100))
print(iterapproxpi(1000))
"""

"""
Exercise 1.32
"""

def recaccumulate (combiningfunction, identity, term, a, nex, b): # for prod the identity is 1, for sum it is 0
    if (a>b):
        return identity
    else:
        return combiningfunction(term(a), recaccumulate(combiningfunction, identity, term, nex(a), nex, b))

def iteraccumulate (combiningfunction, identity, term, a, nex, b): # for prod the identity is 1, for sum it is 0
    marker = a
    prod = identity
    while (marker <= b):
        prod = combiningfunction(prod, term(marker))
        marker = nex(marker)
    return prod

"""
Exercise 1.33
"""

def filtered_accumulate (combiningfunction, identity, term, a, nex, b, filter):
    marker = a
    prod = identity
    while (marker <= b):
        if filter(marker):
            prod = combiningfunction(prod, term(marker))
        marker = nex(marker)
    return prod

def add(a,b):
    return a+b

def prod(a,b):
    return a*b

def sumsquaresofprimesinrange(a,b):
    return filtered_accumulate(add, 0, lambda x: x**2, a, lambda x: x+1, b, isprime)

# print(sumsquaresofprimesinrange(1,10))

import math

def prodrelprimes(n):
    def isrelprime(a):
        return math.gcd(a,n) == 1
    return filtered_accumulate(prod, 1, lambda x: x, 1, lambda x: x+1, n, isrelprime)

# print(prodrelprimes(16))
# print(filtered_accumulate(prod, 1, lambda x: x, 1, lambda x: x+1, 16, lambda x: x%2==1))

"""
Exercise 1.34

Nothing will happen, since the compiler will try to calculate (f 2) which is (2 2) which doesn't make sense
"""


"""
Exercise 1.35
x = 1+1/x means x^2-x-1=0 which we know golden ratio is a root of
"""

tolerance = 0.0001

def closeenough(a,b):
    return abs(a-b) < tolerance

def fixedpoint(f, firstguess):
    def tryit(guess):
        fofit = f(guess)
        if closeenough(guess, fofit):
            return fofit
        else:
            return tryit(fofit)
    
    return tryit(firstguess)

# print(fixedpoint(lambda x: 1+1/x, 1))

"""
Exercise 1.36
"""

print(fixedpoint(lambda x: math.log10(1000)/math.log10(x), 4))


"""
Exercise 1.37
"""








  
