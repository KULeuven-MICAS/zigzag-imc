
import pdb, os

stream = os.popen('./cacti -infile self_gen/cache.cfg')

output = stream.readlines()

for l in output:
    print(l, end = '')

pdb.set_trace()
