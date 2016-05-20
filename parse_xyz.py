# Parse an output file, turn it into two .xyz format files (one for initial 
# and one for final position) 

import sys
import string

# Take in filename from command line

args = sys.argv
if len(args) != 2: 
    print "Usage: ./parse.py [siesta_output_file.out]"
    exit(0)

filename = args[1]

# read file line by line
f = open(filename, 'r')

for i in range(18):
    f.readline()

name = ' '.join(string.split(f.readline())[1:])
label = string.split(f.readline())[1:]
numspec = int(string.split(f.readline())[1])
numatom = int(string.split(f.readline())[1])

coords = []

reading_useless = True

def read_coordinates(fi, num_coor): 
    coordlist = []
    for i in range(num_coor):
        coordlist.append(string.split(fi.readline()))
    return coordlist


while(reading_useless):
    a = f.readline()
    if a[0:7] == 'outcoor':
        coords.append(read_coordinates(f, numatom))
    elif a == '': 
        reading_useless = False

f.close()

def sanitize_elements(coords):
    for c in coords: 
        for atom in c: 
            atom[5] = string.split(atom[5], '_')[0]
    return coords

sanitize_elements(coords) 

def make_output(number, name, coordinates, ofile): 
    ofile.write(str(number) + '\n\"' + name + '\"\n')
    b = ['\t'.join([x[5]] + x[0:3])+'\n' for x in coordinates]
    ofile.writelines(b)
    
g = open(filename[0:-4]+'_initial_'+'.xyz','w')
make_output(numatom, name, coords[0], g)
g.close()

h = open(filename[0:-4]+'_final_'+'.xyz','w')
make_output(numatom, name, coords[-1], h)
h.close()
