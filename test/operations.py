import unittest
import random
from minigrad.engine import Value

class TestOperations(unittest.TestCase):
    
    # Addition
    def testAdd(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)+Value(b)).data, a+b)

    def testAddInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)+b).data, a+b)

    def testRaddInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((a+Value(b)).data, a+b)

    # Subtraction
    def testSub(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)-Value(b)).data, a-b)

    def testSubInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)-b).data, a-b)
        
    def testRsubInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((a-Value(b)).data, a-b)

    # Multiplication
    def testMul(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)*Value(b)).data, a*b)

    def testMulInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)*b).data, a*b)

    def testRmulInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((a*Value(b)).data, a*b)

    # Divison
    def testDiv(self):
        a = random.randint(0, 100)
        b = random.randint(1, 100)
        self.assertEqual(round((Value(a)/Value(b)).data, 8), round(a/b, 8))
    
    def testDivInt(self):
        a = random.randint(0, 100)
        b = random.randint(1, 100)
        self.assertEqual(round((Value(a)/b).data, 8), round(a/b, 8))

    def testRdivInt(self):
        a = random.randint(0, 100)
        b = random.randint(1, 100)
        self.assertEqual(round((a/Value(b)).data, 8), round(a/b, 8))

    # Power
    def testPow(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)**Value(b)).data, a**b)

    def testPowInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)**b).data, a**b)

    def testRpowInt(self):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        self.assertEqual((a**Value(b)).data, a**b)

    # Root
    def testRootInt(self):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        self.assertEqual((Value(a)**(1/Value(b))).data, a**(1/b))

if __name__ == '__main__':
    unittest.main()