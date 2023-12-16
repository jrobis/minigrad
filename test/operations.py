import unittest
import random
from minigrad.engine import Value

class TestOperations(unittest.TestCase):
    def testAddInt(self):
        a = random.randint
        b = random.randint
        self.assertEqual(Value(a)+Value(b), a+b)

if __name__ == '__main__':
    unittest.main()