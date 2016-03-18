class A(object):
    def __init__(self):
        def boo():
            a = self._foo()
            return a

        for i in range(10):
            boo()

    def _foo(self):
        print 'haha'
        return 1

if __name__ == '__main__':
    a = A()
