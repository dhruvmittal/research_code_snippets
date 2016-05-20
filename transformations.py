import numpy
import math

lattice_const = [9.811107,  9.811107, 2.903721]

h2o_mull= [ 
[5.80570231552,	4.87437060417,	0.114704687708],
[6.13127831394	,4.43225211641,	-0.721077288664],
[6.16575109729,	4.29876831049,	0.84890280419]
]

lithloc = [4.48604338,5.05313955,1.39531300]

h2o_lith1 = [
[4.98933152,4.93983400,3.25017771],
[5.87746193,5.02861210,3.63406294],
[4.54266100,4.23961695,3.77968333]
]

hollandite = [
[1.486572,3.259966,1.451860],
[8.324534,6.551140,1.451860],
[6.551140,1.486572,1.451860],
[3.259966,8.324534,1.451860],
[6.392125,8.165520,0.000000],
[3.418981,1.645587,0.000000],
[1.645587,6.392125,0.000000],
[8.165520,3.418981,0.000000],
[1.522808,1.989991,0.000000],
[8.288299,7.821115,0.000000],
[7.821115,1.522808,0.000000],
[1.989991,8.288299,0.000000],
[6.428361,6.895545,1.451860],
[3.382746,2.915562,1.451860],
[2.915562,6.428361,1.451860],
[6.895545,3.382746,1.451860],
[1.605635,4.481268,0.000000],
[8.205472,5.329838,0.000000],
[5.329838,1.605635,0.000000],
[4.481268,8.205472,0.000000],
[6.511188,9.386822,1.451860],
[3.299919,0.424285,1.451860],
[0.424285,6.511188,1.451860],
[9.386822,3.299919,1.451860]
]

def lin_translate(points_array, trans_vector):
    # r = []
    # for point in points_array:
        # r.append([sum(x) for x in zip([point, trans_vector])])
    return [[sum(x) for x in zip(point, trans_vector)] for point in points_array]

def rotz(angle, point):
    return [math.cos(angle) * point[0] - math.sin(angle) * point[1], math.sin(angle)* point[0] + math.cos(angle) * point[1], point[2]]

def roty(angle, point):
    return [math.cos(angle) * point[0] + math.sin(angle) * point[2], point[1], -1 * math.sin(angle)* point[0] + math.cos(angle) * point[2]]

def rot_around_arb(shift, angle, point, axis=3):
    shift_p = lin_translate([point], [-x for x in shift])
    if axis == 2:
        r_p = roty(angle, shift_p[0])
    if axis == 3:
        r_p = rotz(angle, shift_p[0])
    return lin_translate([r_p], shift)[0]

def rot_molecule_arb(shift, angle, molecule, axis=3):
    return [rot_around_arb(shift, angle, atom, axis) for atom in molecule]

def pretty_print_atom(atom):
    print '\t'.join([str(x) for x in atom])


def pretty_print_water_molecule(list_of_atoms):
    for atom in list_of_atoms:
        pretty_print_atom(atom)

print "#og" 
pretty_print_water_molecule(h2o_mull)

print "#up1"
up1 = lin_translate(h2o_mull, [0,0,lattice_const[2]])
pretty_print_water_molecule(up1)


print "#rot 180"
rot180 = rot_molecule_arb([x*.5 for x in lattice_const], math.pi, h2o_mull)
pretty_print_water_molecule(rot180)

print "#up1 rot 45"
up1rot45 = rot_molecule_arb([x*.5 for x in lattice_const], math.pi*.5, up1)
pretty_print_water_molecule(up1rot45)

print "#up1 rot 60"
up1rot60 = rot_molecule_arb([x*.5 for x in lattice_const], math.pi/3.0, up1)
pretty_print_water_molecule(up1rot60)

print "#up1 rot 90"
up1rot90 = rot_molecule_arb([x*.5 for x in lattice_const], math.pi*.5, up1)
pretty_print_water_molecule(up1rot90)

print "#up1 rot 120"
up1rot120 = rot_molecule_arb([x*.5 for x in lattice_const], 2*math.pi/3.0, up1)
pretty_print_water_molecule(up1rot120)

print "#up1 rot 180"
up1rot180 = rot_molecule_arb([x*.5 for x in lattice_const], math.pi*.5, up1)
pretty_print_water_molecule(up1rot180)

print "#up2"
up2 = lin_translate(h2o_mull, [0,0,2*lattice_const[2]])
pretty_print_water_molecule(up2)

print "#up2 rot 240"
up2rot240 = rot_molecule_arb([x*.5 for x in lattice_const], 4*math.pi/3.0, up2)
pretty_print_water_molecule(up2rot240)

print "#second channel"
sc = lin_translate(h2o_mull, [x*.5 for x in lattice_const])
pretty_print_water_molecule(sc)

print "#second channel up1"
scup1 = lin_translate(sc, [0,0,lattice_const[2]])
pretty_print_water_molecule(scup1)

print "#second channel rot180"
scrot180 = lin_translate(rot180, [x*.5 for x in lattice_const])
pretty_print_water_molecule(scrot180)

print "#### LITHIUM ####"

"""
print "#li 0"
li = [x*.25 for x in lattice_const]
pretty_print_atom(li)

print "#lithium up1"
liup1 = lin_translate([li], [0,0,lattice_const[2]])
pretty_print_water_molecule(liup1)

print "#lithium up2"
liup2 = lin_translate([li], [0,0,2*lattice_const[2]])
pretty_print_water_molecule(liup2)

print "#lithum in second channel"
lisc = lin_translate([li], [x*.5 for x in lattice_const])
pretty_print_water_molecule(lisc)

print "#lithium sc up1" 
liscup1 = lin_translate(lisc, [0,0,lattice_const[2]])
pretty_print_water_molecule(liscup1)
"""

print "#lith 0"
pretty_print_atom(lithloc)

print "#lithwater relaxed" 
pretty_print_water_molecule(h2o_lith1)

print "#rot lithwater 180 up" 
rotup = rot_molecule_arb(lithloc, math.pi, h2o_lith1, 2)
pretty_print_water_molecule(rotup)

print "#lith 0 up 2" 
lithloc_up2 = lin_translate([lithloc], [0,0,2*lattice_const[2]])
pretty_print_water_molecule(lithloc_up2)

print "#lithwater up 2" 
h2o_lith1_up2 = lin_translate(h2o_lith1, [0,0,2*lattice_const[2]])
pretty_print_water_molecule(h2o_lith1_up2)

print "#lithwater rot up 180 up 2" 
rotup_up2 = lin_translate(rotup, [0,0,2*lattice_const[2]])
pretty_print_water_molecule(rotup_up2)


#print "#original unit cell" 
#pretty_print_water_molecule(hollandite)

#print "#hollandite up3" 
#holl_up3 = lin_translate(hollandite, [0,0,3*lattice_const[2]])
#pretty_print_water_molecule(holl_up3)
