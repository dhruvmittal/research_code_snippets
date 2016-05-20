import sys

args = sys.argv
if len(args) > 1:
    filename = args[1]
else:
    print "enter a filename"
    exit(0)
if len(args)> 2:
    duplength = int(args[2])
else:
    duplength = 2

f = open(filename, 'r')

num_atoms = int(f.readline())
title = f.readline()
atoms = [line.split() for line in f]

f.close()

lattice_constants = [9.989905, 9.979831, 5.804205]

w = open("ext_" + filename, 'w')
w.write(str(num_atoms * duplength) + '\n')
w.write("extended" + title + '\n')
for t in range(duplength):
    for a in atoms:
    # print type(a[3]), type(lattice_constants[2])
        a[3] = str(float(a[3]) + t*lattice_constants[2])
        w.write("\t".join(a) + '\n')
w.close()
