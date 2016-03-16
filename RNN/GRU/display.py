from time import sleep
from datetime import datetime
import os, sys

if __name__ == '__main__':
    oldcontent = None
    filename = sys.argv[1]
    while 1:
        with open(filename, 'r') as f:
            content = ''.join(f.readlines())
        if content != oldcontent:
            os.system('clear') 
            print "{} updated in {} {}".format('>'*40,
                datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                '<'*40)
            print content
            oldcontent = content
        else:
            print '.',
        sleep(2)
        sys.stdout.flush()
