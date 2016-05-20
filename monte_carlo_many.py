# Metropolis Monte Carlo to determine where water gets inserted in Hollandite
# units: eV, Angstrom, Kelvin
# [0,0,0] in my coordinates is [4.868995,4.868995,1.431023] in the original coordinates


# HOW TO USE:
# Execute python/python2 mc_many.py [steps_in] [temperature] [delta_scale] [number in] | tee mc_output.txt

####### imports #############################################################
import numpy
import numpy.linalg as nl
import math
import random as rand
import sys
from collections import deque
import itertools

###### Take parameters from command line ####################################
args = sys.argv
steps_in = None
temp_in = None
delta_in = None
number_in = 1
if len(args) == 1:
    print "mc_many.py [steps] [temp] [delta_scale] [num_water_mol]"
    exit(0)
if len(args) > 1:
    steps_in = int(args[1])
    temp_in = float(args[2])
if len(args) < 4:
    delta_in = 0
else:
    delta_in = float(args[3])
if len(args) > 4:
    number_in = int(args[4])

##### Options ###############################################################
MIRROR_H2O_IN_PERIODIC = False
FLAG_MIRROR_WATER = True
RANDOMIZE_DELTA_DIST = True
MODEL_LATTICE_VIBRATIONS = True
Mulliken = False
Mulliken_Hollandite = True
Mulliken_Silver_Hollandite = False
if Mulliken_Hollandite or Mulliken_Silver_Hollandite:
  Mulliken = True

####### Constants:###########################################################

#N_A = 6.022*10**23 #avagadro's number
#kcal_mol_2_eV = 3.829294 * 10**-23 * N_A # based on:
# http://www.wolframalpha.com/input/?t=crmtb01&f=ob&i=eV%2Fkcal
kcal_mol_2_eV = .043363

K_B = 8.6173*(10**-5) #eV/K
K_C = 14.39964 # eV*Angstroms / (electron charge)^2

# Van der waals water model constants
#A = 629.4 * 10**3 #kcal Angstrom^12/mol
#B = 625.5 #kcal Angstrom^6/mol
epsilon_lj = .65 #kj/mol
sigma_lj = 3.166 #angstrom
kj_mol_2_eV = .001364
# convert to eV
epsilon_lj = epsilon_lj * kj_mol_2_eV
# construct A and B
A = 4 * epsilon_lj * sigma_lj**12
B = 4 * epsilon_lj * sigma_lj**6
# And convert to eV for single particle
#A = A * kcal_mol_2_eV
#B = B * kcal_mol_2_eV

HYDROGEN_NUCLEAR_RADIUS = 1 * 10**-5 #Angstroms
OXYGEN_NUCLEAR_RADIUS = HYDROGEN_NUCLEAR_RADIUS * 8**(1/3)

E_Labels = ["E_T", "E_C", "E_vdW", "E_LJ", "E_H2O"]

######## PROBLEM GEOMETRY ###################################################
CONSTANT_C = 2.842046 #angstroms 1.431023 * 2 # MC_{MIN,MAX} chosen to restrict MC to one quarter of the tunnel
# within the middle unit cell only.
# this should stop degeneracy due to x-y plane symmetry
MC_MIN = [0,0,-CONSTANT_C/2.] #angstroms
MC_MAX = [2,2,CONSTANT_C/2.] #angstroms

# For more than one molecule , use the full tunnel:
NUMBER_WATER_MOLECULES = number_in
if NUMBER_WATER_MOLECULES > 1:
    MC_MIN = [-2,-2,-CONSTANT_C/2.] #angstroms

# Water molecule information
H_BOND_LENGTH = 1 #angstroms
H_BOND_ANGLE = math.radians(104) #degrees -> radians
O_RAD_LIMIT = 2 #angstroms
#WATER_O_CHARGE = -1.2 #elementary charges
#WATER_H_CHARGE = .6 #elementary charges
WATER_O_CHARGE = -.82 #elementary charges from SPC
WATER_H_CHARGE = .41 #elementary charges from SPC

# Hollandite
# naive charges, non-mulliken
MN_CHARGE = 4 #elementary charges
O_CHARGE = -2 #elementary charges

# for variation of the structural charges parameterized in delta
delta = delta_in

if not RANDOMIZE_DELTA_DIST and delta != 0:
    MN_CHARGE = 4 - delta #elementary charges
    O_CHARGE = -2 + delta*.5 #elementary charges

###### Monte Carlo Tuning ##################################################

if steps_in == None:
    MC_STEPS = 2000000
else:
    MC_STEPS = steps_in
if temp_in == None:
    TEMPERATURE = 300 #Kelvin
else:
    TEMPERATURE = temp_in

BETA = 1./(K_B*TEMPERATURE)
d_step = .001 # displacement scale factor in angstroms for mc step
ang_dis = .087/2 # max angle in radians to randomly displace by, corresponds to ~2.5 degrees

# Convergence:
convergence_window_size = 10000
convergence_threshold = .03

##### Helper function to generate my atom data structure ####################

def generate_atom(name, charge, coordinates, flag=None): #global helper function
    return {"atom":name, "charge":charge, "coord":coordinates, "flag":flag}

#############################################################################
## Super cell with corrected symmetries a la periodic boundary conditions ###
#############################################################################
if not Mulliken:
  new_super_cell = [
      generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,-4.293069]), #1
      generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,-1.431023]), #1
      generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,1.431023]),  #1
      generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,4.293069]),  #1
      generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,-4.293069]), #2
      generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,-1.431023]), #2
      generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,1.431023]),  #2
      generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,4.293069]),  #2
      generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,-4.293069]), #3
      generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,-1.431023]), #3
      generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,1.431023]),  #3
      generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,4.293069]),  #3
      generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,-4.293069]), #4
      generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,-1.431023]), #4
      generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,1.431023]),  #4
      generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,4.293069]),  #4
      generate_atom("Mn", MN_CHARGE,[6.329693,8.022156,-2.862046]), #5
      generate_atom("Mn", MN_CHARGE,[6.329693,8.022156,0.000000]),  #5
      generate_atom("Mn", MN_CHARGE,[6.329693,8.022156,2.862046]),  #5
      generate_atom("Mn", MN_CHARGE,[3.408297,1.715834,-2.862046]), #6
      generate_atom("Mn", MN_CHARGE,[3.408297,1.715834,0.000000]),  #6
      generate_atom("Mn", MN_CHARGE,[3.408297,1.715834,2.862046]),  #6
      generate_atom("Mn", MN_CHARGE,[1.715834,6.329693,-2.862046]), #7
      generate_atom("Mn", MN_CHARGE,[1.715834,6.329693,0.000000]),  #7
      generate_atom("Mn", MN_CHARGE,[1.715834,6.329693,2.862046]),  #7
      generate_atom("Mn", MN_CHARGE,[8.022156,3.408297,-2.862046]), #8
      generate_atom("Mn", MN_CHARGE,[8.022156,3.408297,0.000000]),  #8
      generate_atom("Mn", MN_CHARGE,[8.022156,3.408297,2.862046]),  #8
      generate_atom("O", O_CHARGE,[1.515231,2.033293,-2.862046]),   #9
      generate_atom("O", O_CHARGE,[1.515231,2.033293,0.000000]),    #9
      generate_atom("O", O_CHARGE,[1.515231,2.033293,2.862046]),    #9
      generate_atom("O", O_CHARGE,[8.222759,7.704698,-2.862046]),   #10
      generate_atom("O", O_CHARGE,[8.222759,7.704698,0.000000]),    #10
      generate_atom("O", O_CHARGE,[8.222759,7.704698,2.862046]),    #10
      generate_atom("O", O_CHARGE,[7.704698,1.515231,-2.862046]),   #11
      generate_atom("O", O_CHARGE,[7.704698,1.515231,0.000000]),    #11
      generate_atom("O", O_CHARGE,[7.704698,1.515231,2.862046]),    #11
      generate_atom("O", O_CHARGE,[2.033293,8.222759,-2.862046]),   #12
      generate_atom("O", O_CHARGE,[2.033293,8.222759,0.000000]),    #12
      generate_atom("O", O_CHARGE,[2.033293,8.222759,2.862046]),    #12
      generate_atom("O", O_CHARGE,[6.384227,6.902288,-4.293069]),   #13
      generate_atom("O", O_CHARGE,[6.384227,6.902288,-1.431023]),   #13
      generate_atom("O", O_CHARGE,[6.384227,6.902288,1.431023]),    #13
      generate_atom("O", O_CHARGE,[6.384227,6.902288,4.293069]),    #13
      generate_atom("O", O_CHARGE,[3.353764,2.835703,-4.293069]),   #14
      generate_atom("O", O_CHARGE,[3.353764,2.835703,-1.431023]),   #14
      generate_atom("O", O_CHARGE,[3.353764,2.835703,1.431023]),    #14
      generate_atom("O", O_CHARGE,[3.353764,2.835703,4.293069]),    #14
      generate_atom("O", O_CHARGE,[2.835703,6.384227,-4.293069]),   #15
      generate_atom("O", O_CHARGE,[2.835703,6.384227,-1.431023]),   #15
      generate_atom("O", O_CHARGE,[2.835703,6.384227,1.431023]),    #15
      generate_atom("O", O_CHARGE,[2.835703,6.384227,4.293069]),    #15
      generate_atom("O", O_CHARGE,[6.902288,3.353764,-4.293069]),   #16
      generate_atom("O", O_CHARGE,[6.902288,3.353764,-1.431023]),   #16
      generate_atom("O", O_CHARGE,[6.902288,3.353764,1.431023]),    #16
      generate_atom("O", O_CHARGE,[6.902288,3.353764,4.293069]),    #16
      generate_atom("O", O_CHARGE,[1.696358,4.802777,-2.862046]),   #17
      generate_atom("O", O_CHARGE,[1.696358,4.802777,0.000000]),    #17
      generate_atom("O", O_CHARGE,[1.696358,4.802777,2.862046]),    #17
      generate_atom("O", O_CHARGE,[8.041633,4.935214,-2.862046]),   #18
      generate_atom("O", O_CHARGE,[8.041633,4.935214,0.000000]),    #18
      generate_atom("O", O_CHARGE,[8.041633,4.935214,2.862046]),    #18
      generate_atom("O", O_CHARGE,[4.935214,1.696358,-2.862046]),   #19
      generate_atom("O", O_CHARGE,[4.935214,1.696358,0.000000]),    #19
      generate_atom("O", O_CHARGE,[4.935214,1.696358,2.862046]),    #19
      generate_atom("O", O_CHARGE,[4.802777,8.041633,-2.862046]),   #20
      generate_atom("O", O_CHARGE,[4.802777,8.041633,0.000000]),    #20
      generate_atom("O", O_CHARGE,[4.802777,8.041633,2.862046]),    #20
      generate_atom("O", O_CHARGE,[6.565353,9.671772,-4.293069]),   #21
      generate_atom("O", O_CHARGE,[6.565353,9.671772,-1.431023]),   #21
      generate_atom("O", O_CHARGE,[6.565353,9.671772,1.431023]),    #21
      generate_atom("O", O_CHARGE,[6.565353,9.671772,4.293069]),    #21
      generate_atom("O", O_CHARGE,[3.172637,0.066218,-4.293069]),   #22
      generate_atom("O", O_CHARGE,[3.172637,0.066218,-1.431023]),   #22
      generate_atom("O", O_CHARGE,[3.172637,0.066218,1.431023]),    #22
      generate_atom("O", O_CHARGE,[3.172637,0.066218,4.293069]),    #22
      generate_atom("O", O_CHARGE,[0.066218,6.565353,-4.293069]),   #23
      generate_atom("O", O_CHARGE,[0.066218,6.565353,-1.431023]),   #23
      generate_atom("O", O_CHARGE,[0.066218,6.565353,1.431023]),    #23
      generate_atom("O", O_CHARGE,[0.066218,6.565353,4.293069]),    #23
      generate_atom("O", O_CHARGE,[9.671772,3.172637,-4.293069]),   #24
      generate_atom("O", O_CHARGE,[9.671772,3.172637,-1.431023]),   #24
      generate_atom("O", O_CHARGE,[9.671772,3.172637,1.431023]),    #24
      generate_atom("O", O_CHARGE,[9.671772,3.172637,4.293069])     #24
  ]
else:
  if Mulliken_Hollandite:
    MN_CHARGE = 15 - (8.038 + 5.049) 
    O_CHARGE_1 = 6 - (3.437 + 3.491)
    O_CHARGE_2 = 6 - (3.526 + 3.461)
    new_super_cell = [
        generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,-4.293069]), #1
        generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,-1.431023]), #1
        generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,1.431023]),  #1
        generate_atom("Mn", MN_CHARGE,[1.460699,3.153161,4.293069]),  #1
        generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,-4.293069]), #2
        generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,-1.431023]), #2
        generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,1.431023]),  #2
        generate_atom("Mn", MN_CHARGE,[8.277292,6.584829,4.293069]),  #2
        generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,-4.293069]), #3
        generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,-1.431023]), #3
        generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,1.431023]),  #3
        generate_atom("Mn", MN_CHARGE,[6.584829,1.460699,4.293069]),  #3
        generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,-4.293069]), #4
        generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,-1.431023]), #4
        generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,1.431023]),  #4
        generate_atom("Mn", MN_CHARGE,[3.153161,8.277292,4.293069]),  #4
        generate_atom("Mn", MN_CHARGE,[6.329693,8.022156,-2.862046]), #5
        generate_atom("Mn", MN_CHARGE,[6.329693,8.022156,0.000000]),  #5
        generate_atom("Mn", MN_CHARGE,[6.329693,8.022156,2.862046]),  #5
        generate_atom("Mn", MN_CHARGE,[3.408297,1.715834,-2.862046]), #6
        generate_atom("Mn", MN_CHARGE,[3.408297,1.715834,0.000000]),  #6
        generate_atom("Mn", MN_CHARGE,[3.408297,1.715834,2.862046]),  #6
        generate_atom("Mn", MN_CHARGE,[1.715834,6.329693,-2.862046]), #7
        generate_atom("Mn", MN_CHARGE,[1.715834,6.329693,0.000000]),  #7
        generate_atom("Mn", MN_CHARGE,[1.715834,6.329693,2.862046]),  #7
        generate_atom("Mn", MN_CHARGE,[8.022156,3.408297,-2.862046]), #8
        generate_atom("Mn", MN_CHARGE,[8.022156,3.408297,0.000000]),  #8
        generate_atom("Mn", MN_CHARGE,[8.022156,3.408297,2.862046]),  #8
        generate_atom("O", O_CHARGE_1,[1.515231,2.033293,-2.862046]),   #9
        generate_atom("O", O_CHARGE_1,[1.515231,2.033293,0.000000]),    #9
        generate_atom("O", O_CHARGE_1,[1.515231,2.033293,2.862046]),    #9
        generate_atom("O", O_CHARGE_1,[8.222759,7.704698,-2.862046]),   #10
        generate_atom("O", O_CHARGE_1,[8.222759,7.704698,0.000000]),    #10
        generate_atom("O", O_CHARGE_1,[8.222759,7.704698,2.862046]),    #10
        generate_atom("O", O_CHARGE_1,[7.704698,1.515231,-2.862046]),   #11
        generate_atom("O", O_CHARGE_1,[7.704698,1.515231,0.000000]),    #11
        generate_atom("O", O_CHARGE_1,[7.704698,1.515231,2.862046]),    #11
        generate_atom("O", O_CHARGE_1,[2.033293,8.222759,-2.862046]),   #12
        generate_atom("O", O_CHARGE_1,[2.033293,8.222759,0.000000]),    #12
        generate_atom("O", O_CHARGE_1,[2.033293,8.222759,2.862046]),    #12
        generate_atom("O", O_CHARGE_1,[6.384227,6.902288,-4.293069]),   #13
        generate_atom("O", O_CHARGE_1,[6.384227,6.902288,-1.431023]),   #13
        generate_atom("O", O_CHARGE_1,[6.384227,6.902288,1.431023]),    #13
        generate_atom("O", O_CHARGE_1,[6.384227,6.902288,4.293069]),    #13
        generate_atom("O", O_CHARGE_1,[3.353764,2.835703,-4.293069]),   #14
        generate_atom("O", O_CHARGE_1,[3.353764,2.835703,-1.431023]),   #14
        generate_atom("O", O_CHARGE_1,[3.353764,2.835703,1.431023]),    #14
        generate_atom("O", O_CHARGE_1,[3.353764,2.835703,4.293069]),    #14
        generate_atom("O", O_CHARGE_1,[2.835703,6.384227,-4.293069]),   #15
        generate_atom("O", O_CHARGE_1,[2.835703,6.384227,-1.431023]),   #15
        generate_atom("O", O_CHARGE_1,[2.835703,6.384227,1.431023]),    #15
        generate_atom("O", O_CHARGE_1,[2.835703,6.384227,4.293069]),    #15
        generate_atom("O", O_CHARGE_1,[6.902288,3.353764,-4.293069]),   #16
        generate_atom("O", O_CHARGE_1,[6.902288,3.353764,-1.431023]),   #16
        generate_atom("O", O_CHARGE_1,[6.902288,3.353764,1.431023]),    #16
        generate_atom("O", O_CHARGE_1,[6.902288,3.353764,4.293069]),    #16
        generate_atom("O", O_CHARGE_2,[1.696358,4.802777,-2.862046]),   #17
        generate_atom("O", O_CHARGE_2,[1.696358,4.802777,0.000000]),    #17
        generate_atom("O", O_CHARGE_2,[1.696358,4.802777,2.862046]),    #17
        generate_atom("O", O_CHARGE_2,[8.041633,4.935214,-2.862046]),   #18
        generate_atom("O", O_CHARGE_2,[8.041633,4.935214,0.000000]),    #18
        generate_atom("O", O_CHARGE_2,[8.041633,4.935214,2.862046]),    #18
        generate_atom("O", O_CHARGE_2,[4.935214,1.696358,-2.862046]),   #19
        generate_atom("O", O_CHARGE_2,[4.935214,1.696358,0.000000]),    #19
        generate_atom("O", O_CHARGE_2,[4.935214,1.696358,2.862046]),    #19
        generate_atom("O", O_CHARGE_2,[4.802777,8.041633,-2.862046]),   #20
        generate_atom("O", O_CHARGE_2,[4.802777,8.041633,0.000000]),    #20
        generate_atom("O", O_CHARGE_2,[4.802777,8.041633,2.862046]),    #20
        generate_atom("O", O_CHARGE_2,[6.565353,9.671772,-4.293069]),   #21
        generate_atom("O", O_CHARGE_2,[6.565353,9.671772,-1.431023]),   #21
        generate_atom("O", O_CHARGE_2,[6.565353,9.671772,1.431023]),    #21
        generate_atom("O", O_CHARGE_2,[6.565353,9.671772,4.293069]),    #21
        generate_atom("O", O_CHARGE_2,[3.172637,0.066218,-4.293069]),   #22
        generate_atom("O", O_CHARGE_2,[3.172637,0.066218,-1.431023]),   #22
        generate_atom("O", O_CHARGE_2,[3.172637,0.066218,1.431023]),    #22
        generate_atom("O", O_CHARGE_2,[3.172637,0.066218,4.293069]),    #22
        generate_atom("O", O_CHARGE_2,[0.066218,6.565353,-4.293069]),   #23
        generate_atom("O", O_CHARGE_2,[0.066218,6.565353,-1.431023]),   #23
        generate_atom("O", O_CHARGE_2,[0.066218,6.565353,1.431023]),    #23
        generate_atom("O", O_CHARGE_2,[0.066218,6.565353,4.293069]),    #23
        generate_atom("O", O_CHARGE_2,[9.671772,3.172637,-4.293069]),   #24
        generate_atom("O", O_CHARGE_2,[9.671772,3.172637,-1.431023]),   #24
        generate_atom("O", O_CHARGE_2,[9.671772,3.172637,1.431023]),    #24
        generate_atom("O", O_CHARGE_2,[9.671772,3.172637,4.293069])     #24
    ]
    

def randomize_delta_dist(cell):
    remaining_oxygen_deltas = deque(maxlen=len(cell)+1)
    oxygens_skipped = []
    new_cell = []

    for atom in cell:
        # seperate oxygen and manganese
        if atom["atom"] == "Mn":
            if RANDOMIZE_DELTA_DIST:
                specific_delta = delta*2*(rand.random()-.5)
            else:
                specific_delta = 0
            remaining_oxygen_deltas.append(specific_delta * -.5)
            remaining_oxygen_deltas.append(specific_delta * -.5)
            # optional lattice vibrations
            if MODEL_LATTICE_VIBRATIONS:
                specific_delta = specific_delta + (2*(rand.random() - .5)*.005)
            new_cell.append(generate_atom("Mn", MN_CHARGE + specific_delta, atom["coord"]))
        elif atom["atom"] == "O":
            oxygens_skipped.append(atom)

    # randomize the oxygen delta orders so they're not paired weirdly
    rand.shuffle(remaining_oxygen_deltas)

    # go back through the oxygens and use the random deltas up
    for atom in oxygens_skipped:
        specific_delta = remaining_oxygen_deltas.popleft()
        # optional lattice vibrations
        if MODEL_LATTICE_VIBRATIONS:
            specific_delta = specific_delta + (2*(rand.random() - .5)*.005)
        new_cell.append(generate_atom("O", O_CHARGE + specific_delta, atom["coord"]))

    return new_cell


def center_super_cell(cell):
    # [0,0,0] in my coordinates is [4.868995, 4.868995, 0] in the original coordinates
    def translate_cell(cell,xshift,yshift,zshift):
        new_cell = []
        for atom in cell:
            [x_,y_,z_] = atom["coord"]
            new_cell.append(generate_atom(atom["atom"], atom["charge"],
                [x_+xshift, y_+yshift, z_+zshift]))
        return new_cell
    return translate_cell(cell, -4.868995, -4.868995, 0)
 
def construct_super_cell(cell): #deprecated method to move the old, asymmetric unit cell around.
    # [0,0,0] in my coordinates is [4.868995,4.868995,1.431023] in the original coordinates
    def translate_cell(cell,xshift,yshift,zshift):
        new_cell = []
        for atom in cell:
            [x_,y_,z_] = atom["coord"]
            new_cell.append(generate_atom(atom["atom"], atom["charge"],
                [x_+xshift, y_+yshift, z_+zshift]))
        return new_cell
    return translate_cell(cell + translate_cell(cell, 0,0,CONSTANT_C)
                            + translate_cell(cell, 0,0,-1*CONSTANT_C),
                            -4.868995,-4.868995,-.5*CONSTANT_C)

def placeO():
    # x,y,z within unit cell
    # rand.random only generates positive numbers. myrand fixes that.
    myrand = lambda f: -f*rand.random() if rand.random() <= .5 else f*rand.random()
    return [MC_MAX[0]*rand.random(), MC_MAX[1]*rand.random(), myrand(MC_MAX[2])]

def placeH1(O_loc):
    # alpha, beta on sphere around O
    alpha = math.pi*rand.random() #polar
    beta = 2*math.pi*rand.random() #azimuthal
    # and in cartesian coordinates:
    x = O_loc[0] + H_BOND_LENGTH*math.sin(alpha)*math.cos(beta)
    y = O_loc[1] + H_BOND_LENGTH*math.sin(alpha)*math.sin(beta)
    z = O_loc[2] + H_BOND_LENGTH*math.cos(alpha)
    return [x,y,z,alpha,beta] # keep angles because they're useful for placeH2


def placeH2(O_loc, H1_loc):
    alpha = H1_loc[3]
    beta = H1_loc[4]
    gamma = 2*math.pi*rand.random() #angle on circle at 104 deg from O-H1
    # find x,y,z in rotated coordinate system with O placed at origin
    rad = math.cos(H_BOND_ANGLE - math.pi/2)
    u_x  = rad*math.cos(gamma)
    u_y  = rad*math.sin(gamma)
    u_z = -1*math.sin(H_BOND_ANGLE - math.pi/2)
    # rotate back to original coordinates
    # rotational helper functions in their most general form
    def rotX(coord, angle):
        [x,y,z] = coord
        xnew = x
        ynew = y*math.cos(angle) - z*math.sin(angle)
        znew = y*math.sin(angle) + z*math.cos(angle)
        return [xnew, ynew, znew]
    def rotY(coord, angle):
        [x,y,z] = coord
        xnew = x*math.cos(angle) + z*math.sin(angle)
        ynew = y
        znew = -1*x*math.sin(angle) + z*math.cos(angle)
        return [xnew, ynew, znew]
    def rotZ(coord, angle):
        [x,y,z] = coord
        xnew = x*math.cos(angle) - y*math.sin(angle)
        ynew = x*math.sin(angle) + y*math.cos(angle)
        znew = z
        return [xnew, ynew, znew]
    rcoord = rotZ(rotX([u_x, u_y, u_z], -1*alpha), beta - .5*math.pi)
    # translate back to O not placed at origin
    [x,y,z] = [sum(x) for x in zip(rcoord, O_loc)]
    return [x,y,z]

def rotational_around_o(central_o, h1_coord, h2_coord, angle_x, angle_y, angle_z):
    # small rotational displacement around the current location of oxygen atom.
    # translate molecule so oxygen is at origin:
    o = numpy.array(central_o)
    [h1, h2] = [numpy.array(h1_coord) - o, numpy.array(h2_coord) - o]
    # rotational helper functions, like with the placeH2 function, but explicitly for vectors.
    def rotX(coord, angle):
        [x,y,z] = coord
        xnew = x
        ynew = y*math.cos(angle) - z*math.sin(angle)
        znew = y*math.sin(angle) + z*math.cos(angle)
        return [xnew, ynew, znew]
    def rotY(coord, angle):
        [x,y,z] = coord
        xnew = x*math.cos(angle) + z*math.sin(angle)
        ynew = y
        znew = -1*x*math.sin(angle) + z*math.cos(angle)
        return [xnew, ynew, znew]
    def rotZ(coord, angle):
        [x,y,z] = coord
        xnew = x*math.cos(angle) - y*math.sin(angle)
        ynew = x*math.sin(angle) + y*math.cos(angle)
        znew = z
        return [xnew, ynew, znew]
    # rotate.
    h1 = rotX(rotY(rotZ(h1.tolist(), angle_z), angle_y), angle_x)
    h2 = rotX(rotY(rotZ(h2.tolist(), angle_z), angle_y), angle_x)
    #translate back to position of oxygen
    [h1, h2] = [[sum(x) for x in zip(y,o.tolist())] for y in [h1,h2]]
    return [h1, h2]

def slow_total_energy(crystal, waters_list):
    # gets true total energy of a configuration rather than the shorthand
    # version relevent to the delta E. Developed after I realized I needed
    # to scale better to N molecules
    # Currently used only for final energies for accepted states
    en = len(E_Labels)*[0]
    if MIRROR_H2O_IN_PERIODIC:
        for w in waters_list:
            crystal = mirrorH20(crystal, w)
    def calculate_single_E(atom1, atom2):
    # Helper function, calculates the total energy between any two atoms.
        r = nl.norm(numpy.array(atom1["coord"]) - numpy.array(atom2["coord"]))
        #if r == 0: 
        if r < HYDROGEN_NUCLEAR_RADIUS: 
            # preemptively kill exceptions.
            # In theory, vdw/lj also prevent this, but we play safe.
            # This is an attempt to stop hydrogens in multi-molecule systems from actually 
            # Falling into oxygens for weird configurations of water molecules in which
            # several molecules team up to overcome the lj potentials
            return [float('inf')]*len(en)
        vdw = 0
        lj = 0
        if (atom1["atom"]=="O" and atom2["atom"]=="O"):
            # spc model -> only experience vdw and lj for oxygen-oxygen interactions
            vdw = -B/(r**6) #van der waals attraction
            lj = A/(r**12) # lennard-jones repulsion
        coulomb = K_C * atom1["charge"] * atom2["charge"]/r
        # return, total, coulomb, vdw, lj]
        if (atom1["flag"]=="W" and atom2["flag"]=="W"):
            return [coulomb+vdw+lj,0,0,0,coulomb+vdw+lj]
        else:
            return [coulomb + vdw + lj, coulomb, vdw, lj,0]
    def total_energy_for_one_molecule(molecule_coords, crystal):
        [O_loc, H1_loc, H2_loc] = molecule_coords
        en = len(E_Labels)*[0]
        for atom in crystal:
            en  = [sum(x) for x in zip(en,calculate_single_E(atom, generate_atom(
                "O", WATER_O_CHARGE, O_loc[:3], flag="W")))]
            en = [sum(x) for x in zip(en,calculate_single_E(atom, generate_atom(
                "H1", WATER_H_CHARGE, H1_loc[:3], flag="W")))]
            en  = [sum(x) for x in zip(en,calculate_single_E(atom, generate_atom(
                "H2", WATER_H_CHARGE, H2_loc[:3], flag="W")))]
        return en
    for w in waters_list: 
        en = [sum(x) for x in zip(en, total_energy_for_one_molecule(w,crystal))]
    for subset in itertools.combinations(waters_list, 2):
         bag1_of_atoms = [
             generate_atom("O", WATER_O_CHARGE, subset[0][0][:3], flag="W"),
             generate_atom("H1", WATER_H_CHARGE, subset[0][1][:3], flag="W"),
             generate_atom("H2", WATER_H_CHARGE, subset[0][2][:3], flag="W"),
         ]
         bag2_of_atoms = [
             generate_atom("O", WATER_O_CHARGE, subset[1][0][:3], flag="W"),
             generate_atom("H1", WATER_H_CHARGE, subset[1][1][:3], flag="W"),
             generate_atom("H2", WATER_H_CHARGE, subset[1][2][:3], flag="W")
         ]
         for ss2 in itertools.product(bag1_of_atoms, bag2_of_atoms):
            en = [sum(x) for x in zip(en, calculate_single_E(ss2[0], ss2[1]))]
    return en

def calculate_total_E(crystal, O_loc, H1_loc, H2_loc):
    # [e_tot, e_c, e_vdw, e_lj, e_h20] = [0,0,0,0,0]
    en = len(E_Labels)*[0]
    # If MIRROR_H2O_IN_PERIODIC option is activated, simulate periodic boundary conditions
    # by placing a water molecule at the exact location in the upper and lower unit cells.
    # These temporary water molecules are appended to the overal crystal structure, then
    # energy is calculated in the usual manner.
    # Note that this mirrored water is the current water molecule selected in the monte carlo
    # move, as it needs to have its energy calculated in two locations without varying the
    # other water molecules
    if MIRROR_H2O_IN_PERIODIC:
        crystal = mirrorH20(crystal, [O_loc, H1_loc, H2_loc])
    def calculate_single_E(atom1, atom2):
    # Helper function, calculates the total energy between any two atoms.
        r = nl.norm(numpy.array(atom1["coord"]) - numpy.array(atom2["coord"]))
        if r == 0: # preemptively kill exceptions.
            # In theory, vdw/lj also prevent this, but we play safe.
            return [float('inf')]*4
        vdw = 0
        lj = 0
        if (atom1["atom"]=="O" and atom2["atom"]=="O"):
            # spc model -> only experience vdw and lj for oxygen-oxygen interactions
            vdw = -B/(r**6) #van der waals attraction
            lj = A/(r**12) # lennard-jones repulsion
        coulomb = K_C * atom1["charge"] * atom2["charge"]/r
        # return, total, coulomb, vdw, lj]
        if (atom1["flag"]=="W" and atom2["flag"]=="W"):
            return [coulomb+vdw+lj,0,0,0,coulomb+vdw+lj]
        else:
            return [coulomb + vdw + lj, coulomb, vdw, lj,0]
    for atom in crystal:
        en  = [sum(x) for x in zip(en,calculate_single_E(atom, generate_atom(
            "O", WATER_O_CHARGE, O_loc[:3], flag="W")))]
        en = [sum(x) for x in zip(en,calculate_single_E(atom, generate_atom(
            "H1", WATER_H_CHARGE, H1_loc[:3], flag="W")))]
        en  = [sum(x) for x in zip(en,calculate_single_E(atom, generate_atom(
            "H2", WATER_H_CHARGE, H2_loc[:3], flag="W")))]
        #e_tot = e_tot + calculate_single_E(atom, generate_atom(
        #    "H1", WATER_H_CHARGE, H1_loc[:3]))
        #e_tot = e_tot + calculate_single_E(atom, generate_atom(
        #    "H2", WATER_H_CHARGE, H2_loc[:3]))
    return en

def mirrorH20(crystal, coords):
    # mirror a given h20 molecule in the upper and lower unit cells
    # and append it to the overall crystal
    [o,h1,h2] = coords
    f = "W" if FLAG_MIRROR_WATER else None
    crystal = crystal + [
        # add o, h1, h2 mirrored at x,y,z+c
        generate_atom("O", WATER_O_CHARGE, [o[0],o[1], o[2] + CONSTANT_C],f),
        generate_atom("H", WATER_O_CHARGE, [h1[0],h1[1], h1[2] + CONSTANT_C],f),
        generate_atom("H", WATER_O_CHARGE, [h2[0],h2[1], h2[2] + CONSTANT_C],f),
        # add o, h1, h2 mirrored at x,y,z+c
        generate_atom("O", WATER_O_CHARGE, [o[0],o[1], o[2] - CONSTANT_C],f),
        generate_atom("H", WATER_O_CHARGE, [h1[0],h1[1], h1[2] - CONSTANT_C],f),
        generate_atom("H", WATER_O_CHARGE, [h2[0],h2[1], h2[2] - CONSTANT_C],f)
    ]
    return crystal

def checkIfInDomain(atom_loc, min, max): #min, max define x,y,z ranges
    for i in range(0,2):
        if atom_loc[i] > max[i] or atom_loc[i] < min[i]:
            return False
    return True
    
def monte_carlo(n_cycle, cell, h2o_list):
    # perform n_cycle mc cycles
    # data structures for sampling
    #[e_tot_n, e_c_n, e_vdw_n, e_lj_n] = [0,0,0,0]
    #[o_net, h1_net, h2_net] = [[0,0,0],[0,0,0],[0,0,0]]
    #[o_acc, h1_acc, h2_acc] = [[0,0,0],[0,0,0],[0,0,0]]

    # generate the new cell with the randomized deltas and/or lattice vibrations
    # will produce randomized deltas scaled by the input parameter delta that
    # preserve the neutrality of the overall structure
    # as well as modeled lattice vibrations which do not preserve total charge
    # depending on options enabled
    if RANDOMIZE_DELTA_DIST or MODEL_LATTICE_VIBRATIONS:
        cell = randomize_delta_dist(cell)

    acc_count = 0
    acc_energy_window = deque(maxlen=convergence_window_size+1)
    window_average = 0
    variation = 0
    all_stable_pos_h2o = []
    all_stable_energy_h20 = []
    all_stable_var = []
    for i in range(n_cycle):
        # displace a particle in monte carlo move
        (accept, h2o_list, energies) = monte_carlo_move(cell, h2o_list)
        #[O,H1,H2] = fix_molecule_surface_orient([O[:3],H1[:3],H2[:3]])
        for molecule in h2o_list:
            molecule = fix_molecule_surface_orient(molecule)
        if accept: # tally acceptances
            acc_count += 1
            #h2o_list_acc = h2o_list # record most recent accepted state
            if NUMBER_WATER_MOLECULES > 1: 
                energies = slow_total_energy(cell, h2o_list)
                # slow total energy is O(Num_water factorial * cell_size)
            acc_energy_window.append(energies[0])
            if len(acc_energy_window) == convergence_window_size:
                # slow total average is O(n)
                window_average = numpy.mean(acc_energy_window)
            if len(acc_energy_window) > convergence_window_size:
                oldest = acc_energy_window.popleft()
                newest = energies[0]
                # quick rolling average only takes 4 operations
                window_average=(((window_average * convergence_window_size)
                    - oldest + newest) /convergence_window_size)
                # Calculate a rough percent variation over the window. If this variation is small enough, we
                # record the position this occurred at as well as the energy. To not overcount stable points,
                # I clear the deque.
                variation = abs((oldest-newest)*1./window_average)
                if variation < convergence_threshold:
                    all_stable_pos_h2o.append(h2o_list)
                    all_stable_energy_h20.append(energies)
                    all_stable_var.append(variation)
                    acc_energy_window.clear()
        if i==n_cycle-1:
            # always keep the final position and energy if I haven't done it recently enough.
            if len(acc_energy_window) != 0:
                all_stable_pos_h2o.append(h2o_list)
                all_stable_energy_h20.append(energies)
                all_stable_var.append(variation)
        # sample all positions
        #o_net = [sum(x) for x in zip(o_net, O)]
        #h1_net = [sum(x) for x in zip(h1_net, H1)]
        #h2_net = [sum(x) for x in zip(h2_net, H2)]
        #[e_tot_n, e_c_n, e_vdw_n, e_lj_n] = [sum(x) for x in zip(energies, [e_tot_n, e_c_n, e_vdw_n, e_lj_n])]
    print str(acc_count  * 1./ n_cycle) + " accepted"
    #[avg_o, avg_h1, avg_h2] = [[x *1./n_cycle for x in y] for y in [o_net, h1_net, h2_net]]
    #[avg_tot_e, avg_e_c, avg_e_vdw, avg_e_lj] = [x * 1./n_cycle for x in [e_tot_n, e_c_n, e_vdw_n, e_lj_n]]
    #return [O,H1,H2] #return the finalized position
    #return h2o_list_acc #return the final accepted state in the rigid molecule scheme
    return all_stable_pos_h2o,all_stable_energy_h20,all_stable_var #return all states that fullfill the convergence condition
    #return [avg_o, avg_h1, avg_h2] #return the average position

def monte_carlo_move(crystal,h2o_list):
    # select molecule at random
    index = rand.randint(0, len(h2o_list)-1)
    part = h2o_list[index]
    [O_loc, H1_loc, H2_loc] = part
    # add the rest of the water molecules to the crystal for this move.
    exclude_part = h2o_list[:index] + h2o_list[index+1:]
    effective_structural_water = ([generate_atom("O", WATER_O_CHARGE, x[0],"W") for x in exclude_part]
            + [generate_atom("H", WATER_H_CHARGE, x[1][:3],"W") for x in exclude_part]
            + [generate_atom("H", WATER_H_CHARGE, x[2][:3],"W") for x in exclude_part]
            )
    if MIRROR_H2O_IN_PERIODIC:
        # if simulating periodic boundary conditions for water molecules, also mirror all
        # so-called structural waters in the upper and lower cells. Don't mirror
        # the active molecule in this step, as it will need to be mirrored during
        # the energy calculation step in two different locations.
        for water in exclude_part:
            effective_structural_water = mirrorH20(effective_structural_water, water)
    effective_crystal = crystal + effective_structural_water

    # and now the traditional monte carlo move for a single molecule:
    # get old energy configuration
    old_energy = calculate_total_E(effective_crystal, O_loc[:3], H1_loc[:3], H2_loc[:3])[0]

    #old_energy = slow_total_energy(crystal, h2o_list)[0]

    # monte carlo displacement/reorientation
    good = False
    while not good: # generate new positions until we reach a valid position
        # give particle random cartesian displacement
        r = [d_step*(x-.5) for x in [rand.random(), rand.random(), rand.random()]]
        [nO, nH1, nH2] = [[sum(x) for x in zip(y,r)] for y in [O_loc, H1_loc, H2_loc]]
        # give particle random rotational displacement
        myrandang = lambda ang_scale : ang_scale * (rand.random() - .5)
        [nH1, nH2] = rotational_around_o(nO, nH1[:3], nH2[:3], myrandang(ang_dis), myrandang(ang_dis), myrandang(ang_dis))
        # check if new oxygen position is in mc domain
        good = checkIfInDomain(nO, MC_MIN, MC_MAX)
    # get new energy configuration
    new_energy_list = calculate_total_E(effective_crystal, nO, nH1, nH2)
    #new_energy_list = slow_total_energy(crystal, h2o_list)
    # acceptance rule
    energy_diff = new_energy_list[0] - old_energy
    e_str = ",\t".join([": ".join(x) for x in zip(E_Labels, [str(y) for y in  new_energy_list])])
    if (energy_diff < 0 or rand.random() < numpy.exp(-BETA*energy_diff)):
        # print "ACCEPT\t\t" + "E_T: " + str(new_energy_list[0]) + ",\tE_C: " + str(
                # new_energy_list[1]) + ",\tE_vdW: " + str(new_energy_list[2]) + "\tE_LJ: " + str(new_energy_list[3]) + ",\tE_Diff: " + str(energy_diff)
        print "ACCEPT\t\t" + e_str
        new_h2o_list = h2o_list[:index] + [[nO, nH1, nH2]] +  h2o_list[index+1:]
        return True, new_h2o_list, new_energy_list
    else:
        # print "REJECT\t\t" + "E_T: " + str(new_energy_list[0])  + ",\tE_C: " + str(
                # new_energy_list[1]) + ",\tE_vdW: " + str(new_energy_list[2]) + "\tE_LJ: " + str(new_energy_list[3])+ ",\tE_Diff: " + str(energy_diff)
        print "REJECT\t\t" + e_str
        return False, h2o_list, new_energy_list

def fix_molecule_surface_orient(coord_vec):
    # H1 and H2 are physically identical hydrogens. They are interchangable to the
    # Monte Carlo simulation.  For consistency, we define the valid orientation of
    # the water molecule as having a postive vector(h1-o) cross  vector(h2-o) z component.
    # When this fails to be true, h1 and h2 are swapped.
    [o,h1,h2] = coord_vec
    [a_o, a_h1, a_h2] = [numpy.array(o[:3]), numpy.array(h1[:3]), numpy.array(h2[:3])]
    v1 = numpy.subtract(a_h1,a_o)
    v2 = numpy.subtract(a_h2,a_o)
    if (numpy.cross(v1,v2)[2] < 1):
        return [o,h2,h1]
    else:
        return [o,h1,h2]

def pretty_print_coord(coord):
    # global helper function, return a molecule's coordinates in a human readable format
    print "O\t " + str(coord[0][0]) + "\t"+ str(coord[0][1]) + "\t" + str(coord[0][2]) + "\t"
    print "H\t " + str(coord[1][0]) + "\t"+ str(coord[1][1]) + "\t" + str(coord[1][2]) + "\t"
    print "H\t " + str(coord[2][0]) + "\t"+ str(coord[2][1]) + "\t" + str(coord[2][2]) + "\t"

def pretty_coordinate_shift_back(coords, shiftxyz):
    # shift a coordinate by arbritrary xyz, used to move back the zero for visualization
    pretty_print_coord([[sum(x) for x in zip(y,shiftxyz)] for y in coords])
    

###### Initialize cell#####################################################################
#sc = construct_super_cell(UNIT_CELL)
sc = center_super_cell(new_super_cell)

# visualize super cell for consistency checking:
#print len(sc)
#print "name"
#for thing in sc:
#    c=thing["coord"]
#    print thing["atom"] + "    " + str(c[0]) + "    "+ str(c[1]) + "    " + str(c[2]) + "\t"
#exit()

###### Initial configuration ##############################################################
## Centered
#o_c = [0,0,0]
#h1_c = placeH1(o_c)
#h2_c = placeH2(o_c, h1_c)


##### Try different intial configurations #################################################
## Start at the lower plane:
#o_c = [0,0,-1/2 * CONSTANT_C]
#h1_c = placeH1(o_c)
#h2_c = placeH2(o_c, h1_c)

#### Limited random initial config ########################################################
o_c = placeO()
h1_c = placeH1(o_c)
h2_c = placeH2(o_c, h1_c)
h2o1 = [o_c,h1_c,h2_c]

### random second point ###################################################################
#o_c_2 = placeO()
#h1_c_2= placeH1(o_c_2)
#h2_c_2 = placeH2(o_c_2, h1_c_2)

#h2o2 = [o_c_2, h1_c_2, h2_c_2]

### arb initial points ####################################################################

def place_many_water(number):
    water_list = []
    for i in range(number):
        o_c = placeO()
        h1_c = placeH1(o_c)
        h2_c = placeH2(o_c, h1_c)
        h2o = [o_c,h1_c,h2_c]
        water_list.append(h2o)
    return water_list


###### Check molecule geometry: ############################################################
#def looseEquals(n1, n2, ep=.003):
    #if (n1 + ep > n2 and n1 - ep < n2):
        #return True
    #else:
        #return False
#a_o = numpy.array(o_c)
#a_h1 = numpy.array(h1_c[:3])
#a_h2 = numpy.array(h2_c[:3])
#if(looseEquals(nl.norm(a_h1 - a_o),H_BOND_LENGTH) and looseEquals(nl.norm(a_h2 - a_o),H_BOND_LENGTH)):
    #angle = math.acos(numpy.dot(a_h1 - a_o, a_h2 - a_o))
    #if looseEquals(angle, H_BOND_ANGLE):
        #pass
    #else:
        #print "Invalid bond angle"
        #exit()
    #pass
#else:
    #print "Invalid bond length"
    #exit()

######### Monte Carlo ######################################################################

#mc_instance = monte_carlo(MC_STEPS, sc, [h2o1])
mc_instance = monte_carlo(MC_STEPS,sc,place_many_water(NUMBER_WATER_MOLECULES))

for i in range(0,len(mc_instance[0])):
    e_str = ",\t".join([": ".join(x) for x in zip(E_Labels, [str(y) for y in mc_instance[1][i]])])
    print "Stable point " + str(i+1) + "(var= " + str(mc_instance[2][i]) + "); " + e_str
    for coordinate in mc_instance[0][i]:
        pretty_print_coord(coordinate)

print "Shift back to original coordinates:"
for i in range(0,len(mc_instance[0])):
    e_str = ",\t".join([": ".join(x) for x in zip(E_Labels, [str(y) for y in mc_instance[1][i]])])
    # print 'Stable point ' + str(i+1) + '; ' + e_str
    print "Stable point " + str(i+1) + "(var= " + str(mc_instance[2][i]) + "); " + e_str
    for coordinate in mc_instance[0][i]:
        pretty_coordinate_shift_back(coordinate, [4.868995,4.86899,0])
