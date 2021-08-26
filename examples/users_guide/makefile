
#MY_PATH=../orbiter
MY_PATH=~/DEV.21/GITHUB/orbiter
#MY_PATH=/scratch/betten/COMPILE/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter/


#############################################################################################################
# End of configuration part
#############################################################################################################

GINAC_PATH=$(MY_PATH)/src/apps/ginac
SANDBOX_PATH=$(MY_PATH)/src/apps/sandbox

update:
	cd $(ORBITER_PATH); make clean;
	cd $(MY_PATH); make cleana; git pull; make 


update_all:
	cd $(MY_PATH); make clean; git pull; make 


sandbox:
	$(SANDBOX_PATH)/sandbox.out


#############################################################################################################
# Makefile Variables


MAGMA_PATH=/usr/local/magma


CODE_5_2_3_CODEWORDS="0,7,25,30"


SURFACE_F7_15LINES_POINTS="0,1,2,3,5,6,7,8,9,10,12,18,20,26,28,34,36,42,44,45,50,53,58,59,60,61,62,63,64,67,77,80,90,93,103,107,115,118,122,125,127,142,147,155,157,162,165,170,172,204,208,219,229,240,244,246,251,253,259,277,278,281,286,298,300,302,303,310,312,316,340,343,351,354,358,365,369,372,373,379,384,386,388,393,399"


SURFACE_F7_15LINES_MCKEAN_POINTS="0,1,2,3,4,5,6,7,8,9,10,16,17,28,35,36,59,60,61,62,63,64,65,76,80,91,95,106,107,111,119,121,122,130,138,139,141,146,150,154,155,158,170,175,184,186,199,203,204,206,226,231,234,239,240,252,253,254,278,279,282,287,299,301,302,319,320,330,338,343,345,350,351,357,364,370,371,376,378,382,385,388,392,394,399"

SURFACE_MCKEAN_15_LINES="1,5,-1,16,-2,17,1,10,2,18,1,8,1,9,-1,11,-1,19,-1,14"


CODE_RS_6_4_7="6,2,1,0,0,0, 0,6,2,1,0,0, 0,0,6,2,1,0, 0,0,0,6,2,1"
CODE_RS_10_8_11="7,2,1,0,0,0,0,0,0,0, 0,7,2,1,0,0,0,0,0,0, 0,0,7,2,1,0,0,0,0,0, 0,0,0,7,2,1,0,0,0,0, 0,0,0,0,7,2,1,0,0,0, 0,0,0,0,0,7,2,1,0,0, 0,0,0,0,0,0,7,2,1,0, 0,0,0,0,0,0,0,7,2,1 "


# Dickson:

D1="1,4,1,7,1,12,1,15"
D2= "1,1,1,2,1,3,1,4,1,7,1,11,1,14,1,15,1,19"
D3="1,4,1,7,1,12,1,14,1,15,1,19"
D4="1,3,1,4,1,11,1,9,1,10,1,14,1,17,1,18,1,19"
D5="1,3,1,4,1,8,1,10,1,17,1,18,1,19"
D6="1,2,1,3,1,4,1,7,1,15"
D7="1,2,1,3,1,5,1,9,1,10,1,14,1,15"
D8="1,3,1,4,1,5,1,7,1,10,1,11,1,19"
D9="1,3,1,4,1,7,1,11,1,12,1,15,1,19"
D10="1,4,1,7,1,8,1,9,1,10,1,13,1,15,1,16,1,17"
D11:="1,4,1,7,1,8,1,10,1,13,1,14,1,15,1,16,1,17"
D12="1,4,1,6,1,7,1,8,1,9,1,10,1,15,1,16,1,17"
D13="1,6,1,8,1,9,1,10,1,15,1,16,1,17"
D14="1,6,1,7,1,10,1,11,1,14,1,15"
D15="1,4,1,5,1,6,1,8,1,12,1,14,1,16"
D16=" 1,4,1,7,1,3,1,15,1,12"  
D17="1,4,1,7,1,8,1,9,1,11,1,13,1,15,1,16,1,17,1,18,1,19"
D18="1,4,1,7,1,8,1,11,1,13,1,14,1,15,1,16,1,17,1,18,1,19"
D19= "1,4,1,7,1,6,1,8,1,11,1,14,1,12,1,16,1,17,1,18,1,19 "   
D20="1,4,1,5,1,6,1,7,1,8,1,10,1,13,1,14,1,15"
D21="1,5,1,6,1,8,1,10,1,13,1,14,1,15"
D22="1,4,1,5,1,7,1,9,1,10,1,11,1,13,1,14,1,15,1,18,1,19"
D23="1,4,1,7,1,9,1,11,1,13,1,14,1,15,1,18,1,19"
D24="1,5,1,8,1,9,1,10,1,13,1,14,1,15,1,18,1,19"
D25= "1,5,1,10,1,13,1,11,1,9,1,14,1,15,1,18,1,19 "   
D26="1,5,1,6,1,8,1,10,1,15,1,18,1,19"
D27="1,4,1,5,1,7,1,10,1,11,1,12,1,13,1,14,1,17,1,18"
D28="1,4,1,5,1,6,1,7,1,10,1,11,1,14,1,15,1,17,1,18"
D29="1,4,1,7,1,13,1,11,1,9,1,15,1,17,1,18"
D30= "1,0,1,1,1,2,1,3,1,7,1,10,1,11,1,16"   
D31="1,0,1,1,1,2,1,5,1,7,1,11,1,13,1,14,1,15,1,17,1,18,1,19"
D32="1,0,1,1,1,2,1,3,1,7,1,8,1,10,1,11,1,12,1,15,1,16,1,19"
D33="1,0,1,1,1,2,1,3,1,6,1,7,1,10,1,12,1,13,1,15,1,16,1,19"
D34= "1,0,1,1,1,2,1,3,1,7,1,10,1,16,1,19"  
D35="1,0,1,1,1,2,1,3,1,6,1,7,1,8,1,11,1,13,1,15"
D36="1,0,1,1,1,2,1,3,1,7,1,15"
D37="1,0,1,1,1,2,1,3,1,7,1,12,1,15,1,19"
D38="1,0,1,1,1,2,1,3,1,6,1,7,1,12,1,13,1,15,1,19"
D39= "1,0,1,1,1,2,1,3,1,7,1,5,1,10,1,6,1,13,1,8,1,11,1,9,1,14,1,12,1,15,1,19"  
D40= "1,0,1,1,1,2,1,3,1,7,1,5,1,10,1,6,1,13,1,19 "   
D41="1,2,1,4,1,7,1,5,1,10,1,11,1,14,1,19"
D42="1,2,1,4,1,7,1,11,1,12,1,14,1,15,1,19"
D43= "1,0,1,1,1,2,1,7,1,13,1,8,1,11,1,14,1,15,1,17 "   
D44="1,0,1,1,1,2,1,6,1,8,1,11,1,12,1,14"
D45="1,6,1,8,1,9,1,11,1,12,1,13,1,14,1,15,1,16,1,17,1,18,1,19"
D46="1,5,1,10,1,6,1,13,1,8,1,11,1,9,1,14,1,12,1,15,1,16,1,17,1,18,1,19" 
D47="1,5,1,10,1,6,1,13,1,8,1,11,1,9,1,14,1,16,1,17,1,18,1,19 "



F_ALPHA_BETA_GAMMA_DELTA="beta*(gamma + 1)*x0*x0*x2 + (alpha*delta - beta*gamma + alpha - beta - delta - 1)*x0*x1*x2 -1*(alpha*beta -alpha*delta + delta)*(gamma + 1)*x0*x1*x3 + (0-alpha*delta + alpha*gamma -beta*gamma -beta + delta -gamma)*x0*x2*x2 -(alpha*delta +beta -delta)*(gamma +1)*x0*x2*x3 -(delta + 1)*(alpha - 1)*x1*x1*x2 - (delta + 1)*(alpha - 1)*x1*x1*x3 + (alpha*delta - alpha*gamma + beta*gamma + beta - delta + gamma)*x1*x2*x2 + (alpha*beta*gamma + alpha*beta + alpha*delta - alpha*gamma + beta*gamma + beta - delta + gamma)*x1*x2*x3 + alpha*beta*(gamma + 1)*x1*x3*x3"

F_abcd_eqn = "-(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*(b - d)*X0*X0*X2 + (a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*(a + b - c - d)*X0*X1*X2 + (a*a*c - a*a*d - a*c*c + b*c*c + a*d - b*c)*(b - d)*X0*X1*X3 - (a*d - b*c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X0*X2*X2 - (a*a*c*d - a*b*c*c - a*a*d + a*b*d + b*c*c - b*c*d)*(b - d)*X0*X2*X3 - (a - c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X1*X1*X2 - (a - c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X1*X1*X3 + (a*d - b*c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X1*X2*X2 + ((1+1)*a*a*b*c*d - a*a*b*d*d - (1+1)*a*a*c*d*d - (1+1)*a*b*b*c*c + a*b*b*c*d + (1+1)*a*b*c*c*d + a*b*c*d*d - b*b*c*c*d - a*a*b*c + a*a*c*d + a*a*d*d + a*b*b*c + a*b*c*c - (1+1+1+1)*a*b*c*d - a*c*c*d + a*c*d*d + b*b*c*c)*X1*X2*X3 + c*a*(a*d - b*c - a + b + c - d)*(b - d)*X1*X3*X3"

GOLAY_23_COLUMN_RANKS_PROJECTIVELY="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 132, 913, 1460, 1750, 1898, 2518, 2787, 2874, 3320, 3357, 3662" 

HAMMING_CODE_CODEWORDS="0, 67, 37, 102, 22, 85, 51, 112, 15, 76, 42, 105, 25, 90, 60, 127"


CODE_RM_3_1_GENMA="1,1,1,1,1,1,1,1, 0,1,0,1,0,1,0,1, 0,0,1,1,0,0,1,1, 0,0,0,0,1,1,1,1"
CODE_RM_4_1_GENMA="1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1"
CODE_RS_8="5,6,1,0,0,0,0, 0,5,6,1,0,0,0, 0,0,5,6,1,0,0, 0,0,0,5,6,1,0, 0,0,0,0,5,6,1"	
CODE_RS_11_RREF="1, 0, 0, 0, 0, 0, 0, 0, 7, 2, 0, 1, 0, 0, 0, 0, 0, 0, 8, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 1, 0, 0, 0, 10, 3, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 1, 5, 8"
RS_8_reduced="0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1"

CODE_21_15_4="1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0, 0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0, 1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0, 1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0, 0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0, 1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0, 0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0, 0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0, 1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0, 1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0, 1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0, 0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0, 1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1"


# there are 5 [15,6,6]

# ago=12
CODE_15_6_6_A="1,1,1,1,1,1,1,1,1,1,0,0,0,0,0, 1,1,1,1,1,0,0,0,0,0,1,0,0,0,0, 1,1,1,0,0,1,1,0,0,0,0,1,0,0,0, 1,1,0,1,0,1,0,1,0,0,0,0,1,0,0, 1,0,1,0,1,0,1,1,0,0,0,0,0,1,0, 1,0,1,1,0,1,0,0,1,0,0,0,0,0,1"

# ago=12
CODE_15_6_6_B="1,1,1,1,1,1,1,1,1,1,0,0,0,0,0, 1,1,1,1,1,0,0,0,0,0,1,0,0,0,0, 1,1,1,0,0,1,1,0,0,0,0,1,0,0,0, 1,1,0,1,0,1,0,1,0,0,0,0,1,0,0, 1,0,1,0,1,0,1,1,0,0,0,0,0,1,0, 0,1,1,0,1,1,0,0,1,0,0,0,0,0,1"

#ago=720:
CODE_15_6_6_C="1,1,1,1,1,1,1,1,1,1,0,0,0,0,0, 1,1,1,1,1,0,0,0,0,0,1,0,0,0,0, 1,1,1,0,0,1,1,0,0,0,0,1,0,0,0, 1,1,0,1,0,1,0,1,0,0,0,0,1,0,0, 1,0,1,1,0,1,0,0,1,0,0,0,0,1,0, 1,0,0,0,1,0,1,1,1,0,0,0,0,0,1"

#ago=96:
CODE_15_6_6_D="1,1,1,1,1,1,1,1,1,1,0,0,0,0,0, 1,1,1,1,1,0,0,0,0,0,1,0,0,0,0, 1,1,1,0,0,1,1,0,0,0,0,1,0,0,0, 1,1,0,1,0,1,0,1,0,0,0,0,1,0,0, 1,0,1,0,1,1,0,0,1,0,0,0,0,1,0, 0,1,1,0,0,1,0,1,1,0,0,0,0,0,1"

#ago=360
CODE_15_6_6_E="1,1,1,1,1,1,1,1,1,1,0,0,0,0,0, 1,1,1,1,1,0,0,0,0,0,1,0,0,0,0, 1,1,1,0,0,1,1,0,0,0,0,1,0,0,0, 1,0,0,1,1,1,0,1,0,0,0,0,1,0,0, 0,1,0,1,0,1,1,1,0,0,0,0,0,1,0, 0,1,0,1,1,0,1,0,1,0,0,0,0,0,1"


# created in the combinatorics section:
ELEMENTARY_SYMMETRIC_8_1="x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7"
ELEMENTARY_SYMMETRIC_8_2="x0*x1 + x0*x2 + x0*x3 + x0*x4 + x0*x5 + x0*x6 + x0*x7 + x1*x2 + x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x7 + x2*x3 + x2*x4 + x2*x5 + x2*x6 + x2*x7 + x3*x4 + x3*x5 + x3*x6 + x3*x7 + x4*x5 + x4*x6 + x4*x7 + x5*x6 + x5*x7 + x6*x7"
ELEMENTARY_SYMMETRIC_8_3="x0*x1*x2 + x0*x1*x3 + x0*x1*x4 + x0*x1*x5 + x0*x1*x6 + x0*x1*x7 + x0*x2*x3 + x0*x2*x4 + x0*x2*x5 + x0*x2*x6 + x0*x2*x7 + x0*x3*x4 + x0*x3*x5 + x0*x3*x6 + x0*x3*x7 + x0*x4*x5 + x0*x4*x6 + x0*x4*x7 + x0*x5*x6 + x0*x5*x7 + x0*x6*x7 + x1*x2*x3 + x1*x2*x4 + x1*x2*x5 + x1*x2*x6 + x1*x2*x7 + x1*x3*x4 + x1*x3*x5 + x1*x3*x6 + x1*x3*x7 + x1*x4*x5 + x1*x4*x6 + x1*x4*x7 + x1*x5*x6 + x1*x5*x7 + x1*x6*x7 + x2*x3*x4 + x2*x3*x5 + x2*x3*x6 + x2*x3*x7 + x2*x4*x5 + x2*x4*x6 + x2*x4*x7 + x2*x5*x6 + x2*x5*x7 + x2*x6*x7 + x3*x4*x5 + x3*x4*x6 + x3*x4*x7 + x3*x5*x6 + x3*x5*x7 + x3*x6*x7 + x4*x5*x6 + x4*x5*x7 + x4*x6*x7 + x5*x6*x7"
ELEMENTARY_SYMMETRIC_8_4="x0*x1*x2*x3 + x0*x1*x2*x4 + x0*x1*x2*x5 + x0*x1*x2*x6 + x0*x1*x2*x7 + x0*x1*x3*x4 + x0*x1*x3*x5 + x0*x1*x3*x6 + x0*x1*x3*x7 + x0*x1*x4*x5 + x0*x1*x4*x6 + x0*x1*x4*x7 + x0*x1*x5*x6 + x0*x1*x5*x7 + x0*x1*x6*x7 + x0*x2*x3*x4 + x0*x2*x3*x5 + x0*x2*x3*x6 + x0*x2*x3*x7 + x0*x2*x4*x5 + x0*x2*x4*x6 + x0*x2*x4*x7 + x0*x2*x5*x6 + x0*x2*x5*x7 + x0*x2*x6*x7 + x0*x3*x4*x5 + x0*x3*x4*x6 + x0*x3*x4*x7 + x0*x3*x5*x6 + x0*x3*x5*x7 + x0*x3*x6*x7 + x0*x4*x5*x6 + x0*x4*x5*x7 + x0*x4*x6*x7 + x0*x5*x6*x7 + x1*x2*x3*x4 + x1*x2*x3*x5 + x1*x2*x3*x6 + x1*x2*x3*x7 + x1*x2*x4*x5 + x1*x2*x4*x6 + x1*x2*x4*x7 + x1*x2*x5*x6 + x1*x2*x5*x7 + x1*x2*x6*x7 + x1*x3*x4*x5 + x1*x3*x4*x6 + x1*x3*x4*x7 + x1*x3*x5*x6 + x1*x3*x5*x7 + x1*x3*x6*x7 + x1*x4*x5*x6 + x1*x4*x5*x7 + x1*x4*x6*x7 + x1*x5*x6*x7 + x2*x3*x4*x5 + x2*x3*x4*x6 + x2*x3*x4*x7 + x2*x3*x5*x6 + x2*x3*x5*x7 + x2*x3*x6*x7 + x2*x4*x5*x6 + x2*x4*x5*x7 + x2*x4*x6*x7 + x2*x5*x6*x7 + x3*x4*x5*x6 + x3*x4*x5*x7 + x3*x4*x6*x7 + x3*x5*x6*x7 + x4*x5*x6*x7"
ELEMENTARY_SYMMETRIC_8_5="x0*x1*x2*x3*x4 + x0*x1*x2*x3*x5 + x0*x1*x2*x3*x6 + x0*x1*x2*x3*x7 + x0*x1*x2*x4*x5 + x0*x1*x2*x4*x6 + x0*x1*x2*x4*x7 + x0*x1*x2*x5*x6 + x0*x1*x2*x5*x7 + x0*x1*x2*x6*x7 + x0*x1*x3*x4*x5 + x0*x1*x3*x4*x6 + x0*x1*x3*x4*x7 + x0*x1*x3*x5*x6 + x0*x1*x3*x5*x7 + x0*x1*x3*x6*x7 + x0*x1*x4*x5*x6 + x0*x1*x4*x5*x7 + x0*x1*x4*x6*x7 + x0*x1*x5*x6*x7 + x0*x2*x3*x4*x5 + x0*x2*x3*x4*x6 + x0*x2*x3*x4*x7 + x0*x2*x3*x5*x6 + x0*x2*x3*x5*x7 + x0*x2*x3*x6*x7 + x0*x2*x4*x5*x6 + x0*x2*x4*x5*x7 + x0*x2*x4*x6*x7 + x0*x2*x5*x6*x7 + x0*x3*x4*x5*x6 + x0*x3*x4*x5*x7 + x0*x3*x4*x6*x7 + x0*x3*x5*x6*x7 + x0*x4*x5*x6*x7 + x1*x2*x3*x4*x5 + x1*x2*x3*x4*x6 + x1*x2*x3*x4*x7 + x1*x2*x3*x5*x6 + x1*x2*x3*x5*x7 + x1*x2*x3*x6*x7 + x1*x2*x4*x5*x6 + x1*x2*x4*x5*x7 + x1*x2*x4*x6*x7 + x1*x2*x5*x6*x7 + x1*x3*x4*x5*x6 + x1*x3*x4*x5*x7 + x1*x3*x4*x6*x7 + x1*x3*x5*x6*x7 + x1*x4*x5*x6*x7 + x2*x3*x4*x5*x6 + x2*x3*x4*x5*x7 + x2*x3*x4*x6*x7 + x2*x3*x5*x6*x7 + x2*x4*x5*x6*x7 + x3*x4*x5*x6*x7"
ELEMENTARY_SYMMETRIC_8_6="x0*x1*x2*x3*x4*x5 + x0*x1*x2*x3*x4*x6 + x0*x1*x2*x3*x4*x7 + x0*x1*x2*x3*x5*x6 + x0*x1*x2*x3*x5*x7 + x0*x1*x2*x3*x6*x7 + x0*x1*x2*x4*x5*x6 + x0*x1*x2*x4*x5*x7 + x0*x1*x2*x4*x6*x7 + x0*x1*x2*x5*x6*x7 + x0*x1*x3*x4*x5*x6 + x0*x1*x3*x4*x5*x7 + x0*x1*x3*x4*x6*x7 + x0*x1*x3*x5*x6*x7 + x0*x1*x4*x5*x6*x7 + x0*x2*x3*x4*x5*x6 + x0*x2*x3*x4*x5*x7 + x0*x2*x3*x4*x6*x7 + x0*x2*x3*x5*x6*x7 + x0*x2*x4*x5*x6*x7 + x0*x3*x4*x5*x6*x7 + x1*x2*x3*x4*x5*x6 + x1*x2*x3*x4*x5*x7 + x1*x2*x3*x4*x6*x7 + x1*x2*x3*x5*x6*x7 + x1*x2*x4*x5*x6*x7 + x1*x3*x4*x5*x6*x7 + x2*x3*x4*x5*x6*x7"
ELEMENTARY_SYMMETRIC_8_7="x0*x1*x2*x3*x4*x5*x6 + x0*x1*x2*x3*x4*x5*x7 + x0*x1*x2*x3*x4*x6*x7 + x0*x1*x2*x3*x5*x6*x7 + x0*x1*x2*x4*x5*x6*x7 + x0*x1*x3*x4*x5*x6*x7 + x0*x2*x3*x4*x5*x6*x7 + x1*x2*x3*x4*x5*x6*x7"
ELEMENTARY_SYMMETRIC_8_8="x0*x1*x2*x3*x4*x5*x6*x7"


E0="-1,55,1,45,-1,46,1,57,1,61,-1,62"
E1="1,39,-1,40,-1,49,1,51,1,61,-1,62"
E2="-1,67,1,69,1,59,-1,64,-1,22,1,24"
E3="-1,38,1,47,1,40,-1,67,-1,51,1,69"
E4="1,61,-1,62,-1,67,1,69"
E5="1,38,-1,47,-1,39,2,66,-1,67,1,49,-2,68,1,69,1,61,-1,62"
E6="1,62,-1,63,-1,64,1,24"
E7="1,35,-1,36,-1,38,2,65,-1,29,1,40,-2,66,1,45,1,51,-1,69"
E8="1,61,-1,63,-1,59,1,22"
E9="1,27,1,47,-2,65,-1,48,1,50,-1,40,1,67,2,68,-1,55,-1,51"
E10="1,66,-1,68,-1,59,1,64,1,21,-1,23"
E11="-1,53,1,44,1,66,-1,68,1,55,-1,45"
E12="1,61,-1,62,1,66,-1,68"
E13="1,53,-1,44,1,66,-2,67,-1,68,1,46,2,69,-1,57,1,61,-1,62"
E14="1,60,-1,61,1,64,-1,23"
E15="1,27,1,41,-1,42,-2,65,1,44,-1,40,2,67,1,68,-1,55,-1,45"
E16="1,60,-1,62,1,59,-1,21"
E17="2,65,-1,53,-1,54,-1,29,1,56,-1,66,1,55,1,45,1,51,-2,69"
E18="-1,67,1,69,1,66,-1,68"
E19="-1,66,1,67,1,68,-1,69,-2,61,2,62,1,21,-1,22,-1,23,1,24"
E20="-1,68,1,61,1,69,-1,62"
E21="1,36,-1,41,-1,37,1,43,2,66,-2,67,-1,68,1,69,1,61,-1,62"
E22="-1,66,1,62,1,67,-1,61"
E23="1,54,-1,50,1,66,-1,67,-2,68,2,69,1,52,-1,58,1,61,-1,62"
E24="1,68,-1,69,-1,60,1,63,1,23,-1,24"
E25="1,36,-1,41,-1,40,1,68,1,45,-1,69"
E26="1,66,-1,67,-1,60,1,63,1,21,-1,22"
E27="-1,54,1,50,1,66,-1,67,1,55,-1,51"
E28="1,37,-1,40,-1,43,1,45,1,61,-1,62"
E29="1,55,-1,51,1,52,-1,58,-1,61,1,62"
E30="1,21,-1,23,-1,22,1,24"
E31="1,61,-1,21,-1,62,1,22"
E32="1,61,-1,24,-1,62,1,23"
E33="1,61,-1,24,-1,62,1,22"
E34="1,61,-1,21,-1,24,1,23"
E35="1,62,-1,22,-1,23,1,21"
E36="1,61,-1,21,-1,62,1,23"
E37="1,61,-1,23"
E38="1,21,-1,62"
E39="1,62,-1,22,-1,23,1,24"
E40="1,62,-1,24"
E41="1,22,-1,23"
E42="1,61,-1,21,-1,24,1,22"
E43="1,61,-1,22"
E44="1,21,-1,24"
H0="1,21"
H1="1,21,-1,4"
H2="1,22"
H3="1,22,-1,4"
H4="1,23"
H5="1,23,-1,4"
H6="1,24"
H7="1,24,-1,4"
H8="1,21,-1,22"
H9="1,21,-1,23"
H10="1,22,-1,24"
H11="1,23,-1,24"
H12="1,61,-1,62"
H13="1,61,-1,21,-1,24,-1,62,1,22,1,23"
H14="-1,67,-1,68,1,61,1,66,1,69,-1,62"



Orb0="0"
Orb1="4"
Orb2="16"
Orb3="20"
Orb4="21"
Orb5="24"
Orb6="31"
Orb7="34"
Orb8="35"
Orb9="38"
Orb10="43"
Orb11="46"
Orb12="68"
Orb13="69"
Orb14="76"
Orb15="77"
Orb16="139"
Orb17="140"
Orb18="156"
Orb19="192"
Orb20="264"
Orb21="265"
Orb22="331"
Orb23="337"
Orb24="346"
Orb25="355"
Orb26="362"
Orb27="487"
Orb28="10566"
Orb29="65540"
Orb30="65542"
Orb31="65546"
Orb32="65547"
Orb33="65548"
Orb34="65550"
Orb35="65554"
Orb36="65561"
Orb37="65562"
Orb38="65569"
Orb39="65570"
Orb40="65603"
Orb41="65605"
Orb42="65609"
Orb43="65611"
Orb44="65612"
Orb45="65613"
Orb46="65614"
Orb47="65617"
Orb48="65618"
Orb49="65633"
Orb50="65634"
Orb51="65665"
Orb52="65666"
Orb53="65687"
Orb54="65695"
Orb55="65735"
Orb56="65743"
Orb57="65744"
Orb58="65759"
Orb59="65760"
Orb60="65831"
Orb61="65839"
Orb62="65843"
Orb63="65850"
Orb64="65851"
Orb65="65858"
Orb66="65859"
Orb67="65863"
Orb68="65865"
Orb69="65867"
Orb70="65868"
Orb71="65869"
Orb72="65871"
Orb73="65872"
Orb74="65873"
Orb75="65874"
Orb76="65887"
Orb77="65899"
Orb78="65900"
Orb79="65903"
Orb80="65904"
Orb81="65915"
Orb82="65919"
Orb83="65921"
Orb84="65922"
Orb85="66755"
Orb86="66763"
Orb87="66764"
Orb88="67107"
Orb89="67115"
Orb90="67117"
Orb91="67150"
Orb92="67243"
Orb93="73731"
Orb94="73733"
Orb95="73737"
Orb96="73753"
Orb97="73795"
Orb98="73796"
Orb99="73797"
Orb100="73798"
Orb101="73801"
Orb102="73802"
Orb103="73987"
Orb104="73993"
Orb105="74007"
Orb106="74051"
Orb107="74052"
Orb108="74053"
Orb109="74055"
Orb110="74057"
Orb111="74099"
Orb112="74105"
Orb113="74106"
Orb114="74243"
Orb115="74247"
Orb116="74248"
Orb117="74263"
Orb118="74264"
Orb119="74275"
Orb120="74276"
Orb121="74279"
Orb122="74280"
Orb123="74295"
Orb124="74296"
Orb125="74499"
Orb126="74500"
Orb127="74531"
Orb128="74532"
Orb129="76051"
Orb130="76099"
Orb131="76100"
Orb132="76291"
Orb133="76292"
Orb134="76307"
Orb135="76308"
Orb136="76323"
Orb137="76355"
Orb138="76356"
Orb139="76387"
Orb140="76389"



PG_3_5_DESARGUESIAN_SPREAD="0, 805, 36, 108, 72, 144, 581, 509, 686, 415, 639, 758, 285, 722, 332, 343, 202, 592, 473, 238, 675, 379, 166, 545, 249, 451"

# elements of order 2:
# conjugacy class reps:
# elt order, class size, centralizer order 

#2A: 2 48960 40320 Baer involution

#2B: 2 5355 368640  one block of 10,11

#2C: 2 64260 30720 two blocks of 10,11 (problem group)


#	pdflatex PGGL_4_4_classes_out.tex
#	open PGGL_4_4_classes_out.pdf


# elements of order 2:
# conjugacy class reps:
# elt order, class size, centralizer order 

#2 48960 40320 Baer involution

#2 5355 368640  one block of 10,11

#2 64260 30720 two blocks of 10,11 (problem group)


CLASS_2A=-centralizer_of_element "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" -label "2A" 
# Baer involution

CLASS_2B=-centralizer_of_element "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1, 0" -label "2B"

CLASS_2C=-centralizer_of_element "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" -label "2C"  
# problem group


# 3 classes of elements of order 3
# 4 classes of elements of order 4


# Baer involution:
PGGL_4_4_SUBGROUP_2A=-PGGL 4 4 \
	-subgroup_by_generators "2A" 2 1 "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1"
PGGL_4_4_SUBGROUP_2A_NORMALIZER=-PGGL 4 4 \
	-subgroup_by_generators "centralizer_2A" "40320" 10 \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
	"1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1," \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1," \
	"1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0," \
	"1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0," \
	"1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1," \
	"1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0," \
	"1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0," \
	"1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0," \
	"0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,"



# the problem group, two blocks of 10,11:

PGGL_4_4_SUBGROUP_2C=-PGGL 4 4 \
	-subgroup_by_generators "2C" 2 1 \
	"1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0"
PGGL_4_4_SUBGROUP_2C_NORMALIZER=-PGGL 4 4 \
	-subgroup_by_generators "centralizer_2C" "30720" 9 \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
	"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
	"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
	"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
	"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
	"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
	"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
	"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
	"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1,"


PGGL_4_4_SUBGROUP_5A=-PGGL 4 4 \
	-subgroup_by_generators "5A" 5 1 \
	"0,2,0,0, 1,1,0,0, 0,0,3,0, 0,0,0,3, 0"
PGGL_4_4_SUBGROUP_5A_NORMALIZER=-PGGL 4 4 \
	-subgroup_by_generators "normalizer_5A" "3600" 6 \
	"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0," \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,2,0," \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0," \
	"1,0,0,0,0,1,0,0,0,0,0,2,0,0,2,0,0," \
	"1,0,0,0,2,2,0,0,0,0,1,0,0,0,0,1,1," \
	"0,1,0,0,3,3,0,0,0,0,2,0,0,0,0,2,0," 

PGGL_4_4_SUBGROUP_5B=-PGGL 4 4 \
	-subgroup_by_generators "5B" 5 1 \
	"0,2,0,0,1,1,0,0,0,0,0,2,0,0,1,1,0"
PGGL_4_4_SUBGROUP_5B_NORMALIZER=-PGGL 4 4 \
	-subgroup_by_generators "normalizer_5B" "81600" 6 \
	"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0," \
	"1,0,0,0,0,1,0,0,2,2,2,2,1,3,1,3,0," \
	"1,0,0,0,0,1,0,0,0,0,1,1,0,0,3,2,0," \
	"1,0,0,0,2,2,0,0,0,0,3,0,0,0,1,1,1," \
	"0,1,0,0,3,3,0,0,0,0,0,1,0,0,3,3,0," \
	"0,0,1,0,0,0,1,2,2,0,0,0,2,3,0,0,1,"



PGGL_4_4_SUBGROUP_2Cx2_0=-PGGL 4 4 \
	-subgroup_by_generators "2Cx2_0" 4 2 \
	"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0" \
	"1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0"
PGGL_4_4_SUBGROUP_2Cx2_0_NORMALIZER=-PGGL 4 4 \
	-subgroup_by_generators "normalizer_2Cx2_0" "768" 8 \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1" \
	"1,0,0,0,0,1,0,0,0,0,1,0,2,0,0,1,0" \
	"1,0,0,0,0,1,0,0,1,0,1,0,3,1,0,1,1" \
	"1,0,0,0,0,1,0,0,3,0,1,0,2,3,0,1,1" \
	"1,0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0" \
	"1,0,0,0,2,1,1,0,1,0,1,0,1,1,2,1,1" \
	"1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0" \
	"1,0,0,0,3,1,0,0,1,0,1,0,0,1,3,1,1"





#PGL_4_5_SUBGROUP_3B=-PGL 4 5 \
#	-subgroup_by_generators "3B" 3 1 \
#	"1,0,0,0, 0,1,0,0, 0,0,2,1, 0,0,3,2"
#PGL_4_5_SUBGROUP_3B_NORMALIZER=-PGL 4 5 \
#	-subgroup_by_generators "normalizer_3B" "5760" 8 \
#	"1,0,0,0,0,4,0,0,0,0,1,0,0,0,0,1," \
#	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,4," \
#	"1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4," \
#	"1,0,0,0,0,1,0,0,0,0,3,0,0,0,0,3," \
#	"1,0,0,0,0,3,0,0,0,0,1,0,0,0,0,1," \
#	"1,0,0,0,0,1,0,0,0,0,1,4,0,0,2,1," \
#	"1,0,0,0,4,4,0,0,0,0,1,0,0,0,0,1," \
#	"0,1,0,0,1,0,0,0,0,0,4,0,0,0,0,4," 


#elementary abelian subgroups of order 4 with 3 elements of class 2C:






# nice generators, from Michael Epstein:

PGL_4_5_SUBGROUP_3B_ME=-PGL 4 5 \
	-subgroup_by_generators "3B" 3 1 \
	"1,0,0,0, 0,1,0,0, 0,0,2,2, 0,0,4,2"
PGL_4_5_SUBGROUP_3B_ME_NORMALIZER=-PGL 4 5 \
	-subgroup_by_generators "normalizer_3B" "5760" 8 \
	"1,0,0,0,0,4,0,0,0,0,1,0,0,0,0,1, " \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,4, " \
	"1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4, " \
	"1,0,0,0,0,1,0,0,0,0,3,0,0,0,0,3, " \
	"1,0,0,0,0,3,0,0,0,0,1,0,0,0,0,1, " \
	"1,0,0,0,0,1,0,0,0,0,2,4,0,0,2,3, " \
	"1,0,0,0,4,4,0,0,0,0,1,0,0,0,0,1, " \
	"0,1,0,0,1,0,0,0,0,0,4,0,0,0,0,4," 

PGL_4_5_SUBGROUP_31_ME=-PGL 4 5 \
	-subgroup_by_generators "31" 31 1 \
	"1,0,0,0, 0,3,4,3, 0,3,3,4, 0,3,2,3"
PGL_4_5_SUBGROUP_31_ME_NORMALIZER=-PGL 4 5 \
	-subgroup_by_generators "normalizer_31" "372" 4 \
	"1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4, " \
	"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3, " \
	"1,0,0,0,0,4,0,0,0,0,2,1,0,3,2,4, " \
	"1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3," 





# subgroup of order 31 for the construction of regular packings in PG_3_5:

PGL_4_5_SUBGROUP_31=-PGL 4 5 \
	-subgroup_by_generators "31" 31 1 \
	"2,0,0,0, 0,0,1,0, 0,0,0,1, 0,3,0,4"
PGL_4_5_SUBGROUP_31_NORMALIZER=-PGL 4 5 \
	-subgroup_by_generators "normalizer_31" "372" 4 \
	"1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4," \
	"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3," \
	"1,0,0,0,0,4,0,0,0,0,2,1,0,3,2,4," \
	"1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3,"






#372:
#"1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4, " \
#"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3, " \
#"1,0,0,0,0,4,0,0,0,0,2,1,0,3,2,4, " \
#"1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3," \



#Exterior square roots:

#elt of order 3:
#the exterior square root of f is X=
#[1 0 0 0]
#[0 1 0 0]
#[0 0 2 2]
#[0 0 4 2]

#elt of order 31:
#the exterior square root of g is Z=
#[1 0 0 0]
#[0 3 4 3]
#[0 3 3 4]
#[0 3 2 3]

#Michael


CODE_HAMMING="1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1"
CODE_GV_N15_K6="1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1"
CODE_GV_N15_K6_CHECK="1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1"


DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13 = -d1 1 -q1 7 -d2 1 -q2 13 -K 6 -search_control -W  -end -problem_label DD_CC_7_13
DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13_GROUP1=-subgroup "1,1,1,1, " "91" -group_label "cyclic91"
DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13_MASK1=\
	-nb_orbits_on_blocks 3 \
	-depth 6 \
	-mask_label "no_mask"



DELANDTSHEER_DOYEN_PROBLEM_27_53 = -d1 1 -q1 27 -d2 1 -q2 53 -K 11 -DDx 2 -DDy 1 -search_control -W -end 
DELANDTSHEER_DOYEN_PROBLEM_27_53_GROUP1=-subgroup "1,1,1,0, 1,3,1,0, 1,9,1,0, 1,0,1,1, -2,0,-4,0" "18603" -group_label "group1"
# mask 1:
#  XX.
#  X.X

DELANDTSHEER_DOYEN_PROBLEM_27_53_MASK1=\
	-masktest 3 x ge 1 \
	-masktest 4 x+y ge 3 \
	-depth 4 \
	-mask_label "mask1"

DELANDTSHEER_DOYEN_PROBLEM_3_7 = -d1 1 -q1 3 -d2 1 -q2 7 -K 5 -DDx 3 -DDy 1 -search_control -W -end 
DELANDTSHEER_DOYEN_PROBLEM_3_7_GROUP1=-subgroup "1,1,1,0, 1,0,1,1 " "21" -group_label "group_cyclic"
DELANDTSHEER_DOYEN_PROBLEM_3_7_MASK1= -mask_label "mask1" -depth 5



#############################################################################################################
# Chapter 2 - Getting Started
#############################################################################################################



#############################################################################################################
# Section 2.2: Orbiter Session
	
	
SECTION_ORBITER_SESSION:
	
test_session:
	$(ORBITER_PATH)orbiter.out

		

#############################################################################################################
# Section 2.3: Makefiles and Shell Scripts

SECTION_MAKEFILES_AND_SHELL_SCRIPTS:


#############################################################################################################
# Section 2.4: Objects and Activities

SECTION_OBJECTS_AND_ACTIVITIES:


object_F_2:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 2 -end


object_PG_3_2:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 3 F -end 



#############################################################################################################
# Section 2.5: Mathematical Data

SECTION_MATHEMATICAL_DATA:


create_BLT_5_1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 5 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -catalogue 1 -end \
		-end

create_surface_4_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S4_0 -q 4 -catalogue 0 -end \
		-end \
		-with S4_0 -do \
			-cubic_surface_activity \
			-report \
		-end



#############################################################################################################
# Section 2.6: Set Builder

SECTION_SET_BUILDER:

set_builder_examples:
	$(ORBITER_PATH)orbiter.out -v 2 -long_code 64 7 \
		-set_builder -loop 0 64 1 -end \
		-set_builder -loop 0 32 1 -affine_function 2 1 -end \
		-set_builder -loop 0 16 1 -affine_function 4 2 -clone_with_affine_function 4 3 -end \
		-set_builder -set_builder -set_builder -loop 0 4 1 -affine_function 1 4 \
			-clone_with_affine_function 1 12 -end -clone_with_affine_function 1 16 \
			-end -clone_with_affine_function 1 32 -end\
		-set_builder -set_builder -loop 0 8 1 -affine_function 1 8 \
			-clone_with_affine_function 1 24 -end -clone_with_affine_function 1 32 -end\
		-set_builder -loop 0 16 1 -affine_function 1 16 -clone_with_affine_function 1 48 -end \
		-set_builder -loop 0 32 1 -affine_function 1 32 -end 



#############################################################################################################
# Section 2.7: Input Streams

SECTION_INPUT_STREAMS:



#############################################################################################################
# Chapter 3 - Basic Algebra
#############################################################################################################

#############################################################################################################
# Section 3.1: Basic Number Theory

SECTION_BASIC_NUMBER_THEORY:

PR7:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 7

PR11:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 11

PR13:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 13

PR17:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 17

PR19:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 19

PR23:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 23

PR29:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 29

PR31:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 31

PR37:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root 37

PR_100:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root_interval 2 100

PR_1000:
		$(ORBITER_PATH)orbiter.out -v 1 -smallest_primitive_root_interval 2 1000
		$(ORBITER_PATH)orbiter.out -v 1 -csv_file_latex primitive_element_table_2_1000.csv
		pdflatex primitive_element_table_2_1000.tex
		open primitive_element_table_2_1000.pdf

PE_number_1000:
		$(ORBITER_PATH)orbiter.out -v 1 -number_of_primitive_roots_interval 2 1000
		$(ORBITER_PATH)orbiter.out -v 1 -csv_file_latex table_number_of_pe_2_1000.csv
		pdflatex table_number_of_pe_2_1000.tex
		open table_number_of_pe_2_1000.pdf

Eulerfunction_10000:
		$(ORBITER_PATH)orbiter.out -v 1 -number_of_primitive_roots_interval 10000 10001

# randomized algo:

PR2:
		$(ORBITER_PATH)orbiter.out -v 5 -primitive_root 915839

#a primitive root modulo 915839 is 43085

PM2:
		$(ORBITER_PATH)orbiter.out -v 5 -power_mod 43085 49842 915839

#the power of 43085 to the 49842 mod 915839 is 487320
	
DL2:
		$(ORBITER_PATH)orbiter.out -v 5 -discrete_log 487320 43085 915839


#The discrete log is 49842 since 487320 = 43085^49842 mod 915839, time: 0:22

IM_723:
	$(ORBITER_PATH)orbiter.out -v 5 -inverse_mod 723 4060



IM_3_19:
	$(ORBITER_PATH)orbiter.out -v 5 -inverse_mod 3 19


IM:
	$(ORBITER_PATH)orbiter.out -v 5 -inverse_mod 1865025205 2147483647

IM_gcd:
	$(ORBITER_PATH)orbiter.out -v 5 -extended_gcd 1865025205 2147483647


PM3a:
		$(ORBITER_PATH)orbiter.out -v 5 -power_mod 16807 1073741823 2147483647

sqrt_mod:
		$(ORBITER_PATH)orbiter.out -v 2 -square_root_mod 33 41


sqrt_5_mod_11:
		$(ORBITER_PATH)orbiter.out -v 2 -square_root_mod 5 11 


sqrt_5_mod_19:
		$(ORBITER_PATH)orbiter.out -v 2 -square_root_mod 5 19 


draw_mod_8:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -end \
			-draw_mod_n -n 8 -file mod_8  -end
			pdflatex mod_8_draw.tex
			open mod_8_draw.pdf


draw_mod_13:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -end \
			-draw_mod_n -n 13 -file mod_13 -power_cycle 2 -end
			pdflatex mod_13_draw.tex
			open mod_13_draw.pdf


draw_mod_3:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -nodes_empty -end \
			-draw_mod_n -n 3 -file mod_3 -end 
			pdflatex mod_3_draw.tex
			open mod_3_draw.pdf

draw_mod_3_c:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -nodes_empty -end \
			-draw_mod_n -n 3 -file mod_3_c \
			-cyclotomic_sets 2 "1" -end
			pdflatex mod_3_c_draw.tex
			open mod_3_c_draw.pdf

draw_mod_4:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -nodes_empty -end \
			-draw_mod_n -n 4 -file mod_4 -end 
			pdflatex mod_4_draw.tex
			open mod_4_draw.pdf

draw_mod_6:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -nodes_empty -end \
			-draw_mod_n -n 6 -file mod_6 -end 
			pdflatex mod_6_draw.tex
			open mod_6_draw.pdf

draw_mod_7:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -nodes_empty -end \
			-draw_mod_n -n 7 -file mod_7 -end 
			pdflatex mod_7_draw.tex
			open mod_7_draw.pdf



draw_mod_15:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -end \
			-draw_mod_n -n 15 -file mod_15 -end 
			pdflatex mod_15_draw.tex
			open mod_15_draw.pdf


draw_mod_127:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -scale 0.8 -embedded -end \
			-draw_mod_n -n 127 -file mod_127 -power_cycle 3 -end
		pdflatex mod_127_draw.tex
		open mod_127_draw.pdf


sqrt_mod_20_31:
		$(ORBITER_PATH)orbiter.out -v 2 -square_root_mod 20 31


#############################################################################################################
# Section 3.2: Prime Fields

SECTION_PRIME_FIELDS:
	

F_2:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 2 -end  \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_2.tex
	open GF_2.pdf


F_3:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 3 -end  \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix  \
		-input_csv_file GF_q3_addition_table.csv \
		-box_width 40 -bit_depth 24 -partition 4 3 3 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix  \
		-input_csv_file GF_q3_multiplication_table.csv \
		-box_width 40 -bit_depth 24 -partition 4 2 2 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix  \
		-input_csv_file GF_q3_addition_table_reordered.csv \
		-box_width 40 -bit_depth 24 -partition 4 3 3 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix  \
		-input_csv_file GF_q3_multiplication_table_reordered.csv \
		-box_width 40 -bit_depth 24 -partition 4 2 2 -end
	#pdflatex GF_3.tex
	#open GF_3.pdf

F_5:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 5 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_5.tex
	open GF_5.pdf

F_7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q7_addition_table.csv 40 24 -draw_matrix_partition 3 7 7 
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q7_multiplication_table.csv 40 24 -draw_matrix_partition 3 6 6
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q7_addition_table_reordered.csv 40 24 -draw_matrix_partition 3 7 7 
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q7_multiplication_table_reordered.csv 40 24 -draw_matrix_partition 3 6 6
	pdflatex GF_7.tex
	open GF_7.pdf

F_13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_13.tex
	open GF_13.pdf




F_17:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 17 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q17_addition_table.csv 40 8
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q17_multiplication_table.csv 40 8
	pdflatex GF_17.tex
	open GF_17.pdf






F_19:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 19 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_19.tex
	open GF_19.pdf

F_31:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 31 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q31_addition_table.csv 10 24 -draw_matrix_partition 3 31 31 
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q31_multiplication_table.csv 10 24 -draw_matrix_partition 3 30 30
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q31_addition_table_reordered.csv 10 24 -draw_matrix_partition 3 31 31
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q31_multiplication_table_reordered.csv 10 24 -draw_matrix_partition 3 30 30

F_127:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 127 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q127_addition_table.csv 10 24 -draw_matrix_partition 3 127 127 
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q127_multiplication_table.csv 10 24 -draw_matrix_partition 3 126 126
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q127_addition_table_reordered.csv 10 24 -draw_matrix_partition 3 127 127 
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix GF_q127_multiplication_table_reordered.csv 10 24 -draw_matrix_partition 3 126 126
	#pdflatex GF_7.tex
	#open GF_7.pdf


Wash_matrix_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
			-Walsh_matrix 4 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Walsh_01_4.csv \
		-box_width 10 -bit_depth 24 -partition 3 16 16 -end
	#pdflatex GF_2.tex
	#open GF_2.pdf


#############################################################################################################
# Section 3.3: Polynomials over Finite Fields

SECTION_POLYNOMIALS:




# check which polynomials are irreducible and which are primitive:

sift_polynomials_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -sift_polynomials 8 16 -end



sift_polynomials_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -sift_polynomials 16 32 -end




poly_division:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division "1,0,0,0,0,0,0,0,0,0,1" "1,0,1,1" -end
	
poly_gcd:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-extended_gcd_for_polynomials "1,0,0,0,0,0,0,0,0,0,1" "1,0,1,1" -end

poly_mult_mod1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod "1,2,3" "3,4,5" "6,0,0,1" -end

poly_mult_mod2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod "3,1,2" "5,3,4" "6,0,0,1" -end

Berlekamp_matrix_2_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-Berlekamp_matrix  "1,1,0,1" -end

# the polynomial X^3+X+1 is irreducible over GF(2) because the rank of the Berlekamp matrix is 2.

Berlekamp_matrix_2_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-Berlekamp_matrix "1,1,0,0,1" -end

# the polynomial X^4+X+1 is irreducible over GF(2) because the rank of the Berlekamp matrix is 3.


Berlekamp_matrix_4_3a:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-Berlekamp_matrix  "1,3,0,1" -end

Berlekamp_matrix_4_3b:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-Berlekamp_matrix "1,3,1,1" -end




find_roots_a:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 19 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_find_roots "18,1,1" -end

find_roots_b:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 19 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_find_roots "1,3,1" -end

find_roots_c:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 19 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_find_roots "1,16,1" -end

find_roots_d:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 19 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_find_roots "1,18,1" -end

find_roots_e:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 19 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_find_roots "1,16,3" -end


roots_over_F2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2   -end \
		-with F -do \
		-finite_field_activity  \
		-polynomial_find_roots "0,1,0,1,1,1" -end



roots_over_F8:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -override_polynomial 11  -end \
		-with F -do \
		-finite_field_activity  \
		-polynomial_find_roots "0,1,0,1,1,1" -end




# degree and then order of the field of coefficients:

irred_3_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 3 -end
	pdflatex Irred_q2_d3.tex
	open Irred_q2_d3.pdf


irred_4_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 4 -end
	pdflatex Irred_q2_d4.tex
	open Irred_q2_d4.pdf

# 3 polys

irred_5_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 5 -end
	pdflatex Irred_q2_d5.tex
	open Irred_q2_d5.pdf

# 6 polys

irred_6_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 6 -end
	pdflatex Irred_q2_d6.tex
	open Irred_q2_d6.pdf

# 9 polys

irred_7_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 7 -end
	pdflatex Irred_q2_d7.tex
	open Irred_q2_d7.pdf

# 18 polys

irred_8_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 8 -end
	pdflatex Irred_q2_d8.tex
	open Irred_q2_d8.pdf

# 30 polys

irred_9_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 9 -end
	pdflatex Irred_q2_d9.tex
	open Irred_q2_d9.pdf

# 56 polys

irred_10_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 10 -end
	pdflatex Irred_q2_d10.tex
	open Irred_q2_d10.pdf

# 99 polys

irred_2_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 2 -end
	pdflatex Irred_q4_d2.tex
	open Irred_q4_d2.pdf

# 6 polys


irred_3_4:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-make_table_of_irreducible_polynomials 3 -end
	pdflatex Irred_q4_d3.tex
	open Irred_q4_d3.pdf

# 20 polys


Dedekind_10_10:
	$(ORBITER_PATH)orbiter.out -v 3 -Dedekind_numbers 2 10 2 10


Dedekind_30_2:
	$(ORBITER_PATH)orbiter.out -v 3 -Dedekind_numbers 2 30 2 2


Dedekind_100_2:
	$(ORBITER_PATH)orbiter.out -v 3 -Dedekind_numbers 2 100 2 2


order_of_2_mod_n:
	$(ORBITER_PATH)orbiter.out -v 3 -order_of_q_mod_n 2 3 300
	$(ORBITER_PATH)orbiter.out -v 1 -csv_file_latex order_of_q_mod_n_q2_3_300.csv
	pdflatex order_of_q_mod_n_q2_3_300.tex
	open order_of_q_mod_n_q2_3_300.pdf


search_primitive_poly_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-search_for_primitive_polynomial_in_range 2 2 2 10 | grep //

# stuck in factoring 2^61-1 (which is prime)

search_primitive_poly_3:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 3 3 2 60
	

search_primitive_poly_4:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 4 4 2 30
	
search_primitive_poly_5:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 5 5 2 30
	

search_primitive_poly_7:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 7 7 2 20
	

search_primitive_poly_8:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 8 8 2 20
	
search_primitive_poly_9:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 9 9 2 15
	
search_primitive_poly_11:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 11 11 2 15
	
search_primitive_poly_13:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 13 13 2 15
	
search_primitive_poly_degree_16:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 2 2 16 16
	
search_primitive_poly_32:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 32 32 2 10
	


poly_mult_mod_F4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod "1,1" "1,1" "1,1,1" -end
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod "0,1" "1,1" "1,1,1" -end
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod "0,1" "0,1" "1,1,1" -end





mult_polynomials_2_5_7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -mult_polynomials 5 7 -end
	pdflatex polynomial_mult_5_7.tex
	open polynomial_mult_5_7.pdf

polynomial_division_ranked_2_27_13:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -polynomial_division_ranked 27 13 -end
	pdflatex polynomial_division_27_13.tex
	open polynomial_division_27_13.pdf




mult_polynomials_2_8_15:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -mult_polynomials 8 15 -end
	pdflatex polynomial_mult_8_15.tex
	open polynomial_mult_8_15.pdf

polynomial_division_ranked_2_120_25:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -polynomial_division_ranked 120 25 -end
	pdflatex polynomial_division_120_25.tex
	open polynomial_division_120_25.pdf

# the answer is 5


mult_polynomials_2_7_7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -mult_polynomials 7 7 -end
	pdflatex polynomial_mult_7_7.tex
	open polynomial_mult_7_7.pdf



mult_polynomials_2_4_6:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -mult_polynomials 4 6 -end
	pdflatex polynomial_mult_4_6.tex
	open polynomial_mult_4_6.pdf

polynomial_division_ranked_2_24_13:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -polynomial_division_ranked 24 13 -end
	pdflatex polynomial_division_24_13.tex
	open polynomial_division_24_13.pdf



mult_polynomials_1024_999_997:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -mult_polynomials 999 997 -end
	pdflatex polynomial_mult_999_997.tex
	open polynomial_mult_999_997.pdf


polynomial_division_ranked_2_349147_1033:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division_ranked 349147 1033 \
		-end
	pdflatex polynomial_division_349147_1033.tex
	open polynomial_division_349147_1033.pdf



mult_polynomials_1024_999_997_check:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 1024 -end \
		-with F -do \
		-finite_field_activity -parse_and_evaluate \
		"test" "" "a*b" "a=999,b=997" -end

# evaluates to 61


mult_polynomials_17_12:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -mult_polynomials 17 12 -end
	pdflatex polynomial_mult_17_12.tex
	open polynomial_mult_17_12.pdf
	
# gives 204

polynomial_division_ranked_2_204_37:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -polynomial_division_ranked 204 37 -end
	pdflatex polynomial_division_204_37.tex
	open polynomial_division_204_37.pdf
	
# answer is 18
	
	





#############################################################################################################
# Section 3.4: Extension Fields

SECTION_EXTENSION_FIELDS:
	



F_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end


F_4_tables:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end \
		-draw_matrix -input_csv_file GF_q4_addition_table.csv \
			-box_width 40 -bit_depth 24 -partition 3 4 4 -end \
		-draw_matrix -input_csv_file GF_q4_multiplication_table.csv \
			-box_width 40 -bit_depth 24 -partition 3 3 3 -end
	#pdflatex GF_4.tex
	#open GF_4.pdf

trace_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q4_trace.csv F_q4_trace_2x2.csv 2
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix -input_csv_file F_q4_trace_2x2.csv \
		-box_width 40 -bit_depth 24 -partition 4 2 2 -end
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity -algebraic_normal_form F_q4_trace.csv 2 -end
	
trace_4_WH_transform:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity -Walsh_Hadamard_transform F_q4_trace.csv 2 -end


F_8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 8 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_8.tex
	open GF_8.pdf


F_8_near_bent_5:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 8 -end \
		-with F -do -finite_field_activity -identity_function  F8.csv -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 8 -end \
		-with F -do -finite_field_activity -apply_power_function  F8.csv 5 -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 8 -end \
		-with F -do -finite_field_activity -apply_trace_function  F8_power_5.csv -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -Walsh_Hadamard_transform F8_power_5_trace.csv 4 -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F8_power_5_trace.csv F8_power_5_trace_8x1.csv 1
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F8_power_5_trace_8x1.csv \
		-box_width 40 -bit_depth 24  -partition 4 8 1 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Walsh_01_3.csv \
		-box_width 40 -bit_depth 24  -partition 4 8 8 -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F8_power_5_trace_transformed.csv F8_power_5_trace_transformed_8x1.csv 1
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F8_power_5_trace_transformed_8x1.csv \
		-box_width 40 -bit_depth 24  -partition 4 8 1 -end

bent_4:
	$(ORBITER_PATH)orbiter.out -v 3 -bent 4

bent_4a:
	$(ORBITER_PATH)orbiter.out -v 3 -reformat bent_function_n4.csv bent_function_n4_16x1.csv 1
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-Walsh_Hadamard_transform bent_function_n4.csv 4 -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-reformat bent_function_n4_transformed.csv \
		bent_function_n4_transformed_16x1.csv 1
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix -input_csv_file bent_function_n4_transformed_16x1.csv -box_width 40 -bit_depth 24  -partition 4 16 1 -end

#0 : 0 / 448 : ( 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0 ) : 27328 : ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...0, 0, 0 ) : homogeneous_polynomial_domain::print_equation
#X0*X3*X4^2 + X1*X2*X4^2


F8_over_F2_field_reduction:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -end \
		-loop L 0 8 1 \
		-with F -do \
		-finite_field_activity -field_reduction "F8_red_%L" 2 1 1 "%L" -end \
		-end_loop
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 8 1 \
			-draw_matrix -input_csv_file F8_red_%L.csv \
			-box_width 40 -bit_depth 24  -partition 4 3 3 -end \
		-end_loop
	#pdflatex field_reduction_Q8_q2_5_7.tex



F_9:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 9 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q9_addition_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 "3,6" "3,6" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q9_multiplication_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 "2,6" "2,6" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q9_addition_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 9 9 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q9_multiplication_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 8 8 -end
	pdflatex GF_9.tex
	open GF_9.pdf

trace_9:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 9 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q9_trace.csv F_q9_trace_3x3.csv 3
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F_q9_trace_3x3.csv \
		-box_width 40 -bit_depth 24  -partition 4 3 3 -end
	





F_16:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_16.tex



F_16_tables:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q16_addition_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 16 16 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q16_multiplication_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 15 15 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q16_addition_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 16 16 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q16_multiplication_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 15 15 -end



trace_16:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q16_trace.csv F_q16_trace_4x4.csv 4
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F_q16_trace_4x4.csv \
		-box_width 40 -bit_depth 24  -partition 4 4 4 -end
	

F_16_bent_wrong:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity -identity_function  F16.csv -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity -apply_power_function  F16.csv 9 -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity -apply_trace_function  F16_power_9.csv -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -Walsh_Hadamard_transform F16_power_9_trace.csv 4 -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F16_power_9_trace.csv F16_power_9_trace_16x1.csv 1
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F16_power_9_trace_16x1.csv \
		-box_width 40 -bit_depth 24  -partition 4 16 1 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Walsh_01_4.csv -box_width 40 -bit_depth 24  -partition 4 16 16 -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F16_power_9_trace_transformed.csv F16_power_9_trace_transformed_16x1.csv 1
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F16_power_9_trace_transformed_16x1.csv \
		-box_width 40 -bit_depth 24  -partition 4 16 1 -end


F_16_over_F_4_field_reduction:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-loop L 0 16 1 \
			-with F -do \
			-finite_field_activity \
			-field_reduction "F16_over_F4_%L" 4 1 1 "%L" -end \
		-end_loop \
		-loop L 0 16 1 \
			-draw_matrix -input_csv_file F16_over_F4_%L.csv \
			-box_width 40 -bit_depth 24  -partition 4 2 2 -end \
		-end_loop




# the polynomial 31 is not primitive:

F_16_poly31:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -override_polynomial 31 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_GF -end


F_25:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 25 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q25_addition_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 25 25 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q25_multiplication_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 24 24 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q25_addition_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 25 25 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q25_multiplication_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 24 24 -end

trace_25:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 25 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q25_trace.csv F_q25_trace_5x5.csv 5
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q25_addition_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 25 25 -end



F_32:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 32 -end \
		-with F -do -finite_field_activity \
		-cheat_sheet_GF -end
	pdflatex GF_32.tex
	open GF_32.pdf



F_49:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 49 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q49_addition_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 49 49 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q49_multiplication_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 48 48 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q49_addition_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 49 49 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q49_multiplication_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 48 48 -end
	#pdflatex GF_49.tex
	#open GF_49.pdf

trace_49:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 49 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q49_trace.csv F_q49_trace_7x7.csv 7
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F_q49_trace_7x7.csv \
		-box_width 40 -bit_depth 24  -partition 4 7 7  -end



F_64:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 64 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q64_addition_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 64 64 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q64_multiplication_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 63 63 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q64_addition_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 64 64 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q64_multiplication_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 63 63 -end
	pdflatex GF_64.tex
	open GF_64.pdf


trace_64:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 64 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q64_trace.csv F_q64_trace_8x8.csv 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F_q64_trace_8x8.csv \
		-box_width 40 -bit_depth 24  -partition 3 4 "1,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,1" -end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-algebraic_normal_form F_q64_trace.csv 6 -end


trace_64_poly_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 64 -override_polynomial 115 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q64_trace.csv F_q64_trace_8x8.csv 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F_q64_trace_8x8.csv \
		-box_width 40 -bit_depth 24  -partition 3 8 8 -end


F_64_over_F_8_field_reduction:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 64 -end \
		-with F -do \
		-finite_field_activity -field_reduction "F64_over_F8" 8 8 8 "0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23, 24,25,26,27,28,29,30,31, 32,33,34,35,36,37,38,39, 40,41,42,43,44,45,46,47, 48,49,50,51,52,53,54,55, 56,57,58,59,60,61,62,63" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F64_over_F8.csv \
		-box_width 40 -bit_depth 24  -partition 4 "2,2,2,2,2,2,2,2" "2,2,2,2,2,2,2,2" -end
	pdflatex field_reduction_Q64_q8_8_8.tex
	open field_reduction_Q64_q8_8_8.pdf

F_64_over_F_4_field_reduction:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 64 -end \
		-with F -do \
		-finite_field_activity \
			-field_reduction "F64_over_F4" 4 8 8 "0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23, 24,25,26,27,28,29,30,31, 32,33,34,35,36,37,38,39, 40,41,42,43,44,45,46,47, 48,49,50,51,52,53,54,55, 56,57,58,59,60,61,62,63" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F64_over_F4.csv \
		-box_width 40 -bit_depth 24 \
		-partition 4 "3,3,3,3,3,3,3,3" "3,3,3,3,3,3,3,3" -end
	pdflatex field_reduction_Q64_q4_8_8.tex
	open field_reduction_Q64_q4_8_8.pdf

F64_over_F2_field_reduction:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 64 -end \
		-with F -do \
		-finite_field_activity \
			-field_reduction "F64_over_F2" 2 8 8 "0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23, 24,25,26,27,28,29,30,31, 32,33,34,35,36,37,38,39, 40,41,42,43,44,45,46,47, 48,49,50,51,52,53,54,55, 56,57,58,59,60,61,62,63" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F64_over_F2.csv \
		-box_width 40 -bit_depth 24 \
		-partition 4 "6,6,6,6,6,6,6,6" "6,6,6,6,6,6,6,6" -end
	pdflatex field_reduction_Q64_q2_8_8.tex
	open field_reduction_Q64_q2_8_8.pdf



trace_81:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 81 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q81_trace.csv F_q81_trace_9x9.csv 9
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F_q81_trace_9x9.csv \
		-box_width 40 -bit_depth 24  -partition 4 9 9 -end


F_125:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 125 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_125.tex
	open GF_125.pdf



F_256:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 256 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q256_addition_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 256 256 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q256_multiplication_table.csv \
		-box_width 40 -bit_depth 24  -partition 3 255 255 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q256_addition_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 256 256 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file GF_q256_multiplication_table_reordered.csv \
		-box_width 40 -bit_depth 24  -partition 3 255 255 -end
	pdflatex GF_256.tex
	open GF_256.pdf


trace_256:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 256 -end \
		-with F -do -finite_field_activity -trace -end
	$(ORBITER_PATH)orbiter.out -v 3 -reformat F_q256_trace.csv F_q256_trace_16x16.csv 16
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file F_q256_trace_16x16.csv \
		-box_width 40 -bit_depth 24 \
		-partition 4 "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" \
		"1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" -end



F_289:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 289 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_289.tex
	open GF_289.pdf



normal_basis_2_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-normal_basis 3 -end







#############################################################################################################
# Section 3.5: Linear Algebra over Finite Fields


SECTION_LINEAR_ALGEBRA:


RREF:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-RREF 2 5 "1,1,1,1,0,1,1,0,0,1" -normalize_from_the_right \
		-end

	
	

nullspace:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F2 -finite_field -q 2 -end \
		-with F2 -do -finite_field_activity  \
		-nullspace 2 5 "1,1,1,1,0,1,1,0,0,1" \
		-normalize_from_the_right \
		-end



eigenstuff:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define F -finite_field -q 5 -end \
		-eigenstuff F 4 "0,1,0,2,0,1,2,1,4,2,3,1,2,0,4,3" 

classes_GL_3_2:
	$(ORBITER_PATH)orbiter.out -v 7 \
		-define F -finite_field -q 2 -end \
		-all_rational_normal_forms F 3 
	#pdflatex Class_reps_GL_3_2.tex
	#open Class_reps_GL_3_2.pdf


classes_GL_4_2:
	$(ORBITER_PATH)orbiter.out -v 7 \
		-define F -finite_field -q 2 -end \
		-all_rational_normal_forms F 4 
	pdflatex Class_reps_GL_4_2.tex
	open Class_reps_GL_4_2.pdf




# 252 classes



RREF_demo_4_4_q5:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 5 -end \
		-with F -do \
		-finite_field_activity -RREF_demo 4 4 -end
	pdflatex RREF_example_q5_4_4.tex
	#open RREF_example_q5_4_4.pdf
	gs -sDEVICE=png16 -dFIXEDMEDIA -dDEVICEWIDTHPOINTS=500 -dDEVICEHEIGHTPOINTS=450 \
		-r240 -oRREF_example_q5_4_4_page%02d.png \
		RREF_example_q5_4_4.pdf

#-dDEVICEWIDTHPOINTS=100 -dDEVICEHEIGHTPOINTS=100 
#-sPAPERSIZE=hagaki

RREF_demo_4_6_q7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-with F -do \
		-finite_field_activity -RREF_random_matrix 4 6 -end
	pdflatex RREF_example_q7_4_6.tex
	gs -sDEVICE=png16 -dFIXEDMEDIA -dDEVICEWIDTHPOINTS=500 -dDEVICEHEIGHTPOINTS=450 \
		-r240 -oRREF_example_q7_4_6_page%02d.png \
		RREF_example_q7_4_6.pdf
	open RREF_example_q7_4_6.pdf


RREF_demo_4_8_q8:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -end \
		-with F -do \
		-finite_field_activity -RREF_random_matrix 4 8 -end
	pdflatex RREF_example_q8_4_8.tex
	#open RREF_example_q8_4_8.pdf
	gs -sDEVICE=png16 -dFIXEDMEDIA -dDEVICEWIDTHPOINTS=500 -dDEVICEHEIGHTPOINTS=450 \
		-r240 -oRREF_example_q8_4_8_page%02d.png \
		RREF_example_q8_4_8.pdf



RREF_RS_6_4_7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-with F -do \
		-finite_field_activity -RREF 4 6 $(CODE_RS_6_4_7) -end
	pdflatex RREF_example_q7_4_6.tex
	gs -sDEVICE=png16 -dFIXEDMEDIA -dDEVICEWIDTHPOINTS=500 -dDEVICEHEIGHTPOINTS=450 \
		-r240 -oRREF_example_q7_4_6_page%02d.png \
		RREF_example_q7_4_6.pdf





#############################################################################################################
# Chapter 4 - Geometry
#############################################################################################################


#############################################################################################################
# Section 4.1: Finite Projective Spaces

SECTION_FINITE_PROJECTIVE_SPACES:

PG_1_16:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 1 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_1_16.tex 
	open PG_1_16.pdf


PG_2_13:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_2_13.tex 
	open PG_2_13.pdf

PG_2_64:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 64 -end \
		-define P -projective_space 2 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_2_64.tex 
	open PG_2_64.pdf







PG_3_2:
	$(ORBITER_PATH)orbiter.out -v 0 \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 3 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_3_2.tex 
	open PG_3_2.pdf


PG_3_4:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_3_4.tex 
	open PG_3_4.pdf

PG_3_5:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_3_5.tex 
	open PG_3_5.pdf


PG_3_7:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_3_7.tex 
	open PG_3_7.pdf




PG_3_8:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_3_8.tex 
	open PG_3_8.pdf


PG_3_16:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_3_16.tex 
	open PG_3_16.pdf




PG_3_25:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_3_25.tex 
	open PG_3_25.pdf




PG_4_3:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 3 -end \
		-define P -projective_space 4 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_4_3.tex 
	open PG_4_3.pdf


PG_8_2:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 8 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_8_2.tex 
	open PG_8_2.pdf




PG_2_4_decomposition_singer_power:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-cheat_sheet_for_decomposition_by_element_PG \
				7 "0,1,0, 0,0,1, 2,1,1, 0" "PG_2_4_singer.tex" \
		-end
	pdflatex PG_2_4_singer.tex
	open PG_2_4_singer.pdf


trans:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 16 -end -end \
		-with F -do -finite_field_activity \
		-move_two_lines_in_hyperplane_stabilizer_text  \
		"1,0,0,0, 0,0,0,1"  "1,1,0,2, 0,0,1,0" \
		"1,0,0,0, 0,0,0,1" "0,1,0,1, 0,0,1,0" \
	-end


#############################################################################################################
# Section 4.2: Finite Desarguesian Projective Planes


SECTION_FINITE_DESARGUESIAN_PROJECTIVE_PLANES:


PG_2_2:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -xout 500000 -yout 500000 -scale 0.3 -radius 250 -end \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_PG 2 -end
	pdflatex PG_2_2.tex 
	open PG_2_2.pdf





PG_2_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_PG 2 -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf


PG_2_4_with_decomposition:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity \
		-decomposition_by_element 2 1 "0,1,0, 0,0,1, 2,1,1, 0" "PG_2_4_singer" -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf

#PG_2_4_singer_incma_transitive.csv


PG_2_4_sub3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity \
		-cheat_sheet_PG 2  \
		-decomposition_by_element 3 "0,1,0, 0,0,1, 2,1,1, 0" "PG_2_4_singer" -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf

PG_2_4_sub7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_PG 2  \
			-decomposition_by_element 7 "0,1,0, 0,0,1, 2,1,1, 0" "PG_2_4_singer" -end
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf


PG_2_4_incma_singer:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file PG_2_4_singer_incma_transitive.csv \
		-box_width 40 -bit_depth 24 \
		-partition 3 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -end
	open PG_2_4_singer_incma_transitive_draw.bmp


PG_2_4_incma_singer_sub_3:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file PG_2_4_singer_incma_subgroup_index_3.csv \
		-box_width 20 -bit_depth 24 \
		-partition 3 7,7,7 7,7,7 -end
	open PG_2_4_singer_incma_subgroup_index_3_draw.bmp

PG_2_4_incma_singer_sub_7:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file PG_2_4_singer_incma_subgroup_index_7.csv \
		-box_width 20 -bit_depth 24 \
		-partition 3 3,3,3,3,3,3,3 3,3,3,3,3,3,3 -end
	open PG_2_4_singer_incma_subgroup_index_7_draw.bmp


PG_2_4_incma:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_matrix -input_csv_file PG_n2_q4_incidence_matrix.csv \
		-box_width 20 -bit_depth 8 \
		-partition 3 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 -end
	open PG_n2_q4_incidence_matrix_draw.bmp


#PG_n2_q4_incidence_matrix.csv




PG_2_8:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -radius 200 -end \
		-define F -finite_field -q 8 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_PG 2 -end
	pdflatex PG_2_8.tex 
	open PG_2_8.pdf

PG_2_16:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -xin 20000 -yin 20000 \
			-radius 200 -line_width 0.3 -nodes_empty -end \
		-define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_PG 2 -end
	pdflatex PG_2_16.tex 
	open PG_2_16.pdf







#############################################################################################################
# Section 4.3: The Grassmannian


SECTION_GRASSMANNIAN:

GR_3_2_2:
	$(ORBITER_PATH)orbiter.out \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_Gr 3 2 -end
	pdflatex Gr_3_2_2.tex 
	open Gr_3_2_2.pdf


#############################################################################################################
# Section 4.4: Algebraic Sets


SECTION_ALGEBRAIC_SETS:



EC_11.txt:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object EC \
				-projective_variety "EC_11" 3 "1,0,3,2,10,6,1,7" \
				-monomial_type_PART \
			-end \
		-end \
		-with EC -do -combinatorial_object_activity -save \
		-end


conic_ideal_q13:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object S \
				-set "0,1,2,3,43" \
			-end \
		-end \
		-with S -do -combinatorial_object_activity \
			-ideal 2 \
		-end



Hirschfeld_surface_q4.txt:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object Hirschfeld_surface_q4 \
				-projective_variety "Hirschfeld_surface_q4" 3 "1,6,1,8,1,11,1,13" \
				-monomial_type_PART \
			-end \
		-end \
		-with Hirschfeld_surface_q4 -do -combinatorial_object_activity -save \
		-end

# creates Hirschfeld_surface_q4.txt



Endrass_F7.txt:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object Endrass_F7 \
				-projective_variety "Endrass_F7" 8 \
					-long_string \
					"6,0,4,4,2,7,5,9,6,20,6,23,1,25,3,30,1,32,3,34,4,56,6,59,1,61,6,66," \
					"2,68,6,70,3,77,2,79,6,83,6,120,2,123,5,125,3,130,1,132,3,134,3,141," \
					"2,143,6,147,3,156" \
					-end_string \
				-monomial_type_LEX \
			-end \
		-end \
		-with Endrass_F7 -do -combinatorial_object_activity -save \
		-end



# we created a set of 33 points, called Endrass_F7.txt



octic_prepare:
	$(ORBITER_PATH)orbiter.out -v 4 -define D \
		-diophant -label octic_monomials  \
		-coefficient_matrix 1 4 "1,1,1,1" -RHS "8,8,1" \
		-x_min_global 0 -x_max_global 8 \
		-end
		-with D -do \
			-diophant_activity -solve_mckay \
		-end
	sort -r octic_monomials.sol >octic_monomials_sorted.txt

#Found 165 solutions with 210 backtrack steps
# 165=binomial(11,3)



del_Pezzo_F13ab_report:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-define f3 -formula "del_Pezzo_F13a" "del\_Pezzo\_F13a" "x,y,z" \
			"x*x*x*x+y*y*y*y+z*z*z*z+8*x*x*y*y+8*x*x*z*z+8*y*y*z*z" \
		-define f4 -formula "del_Pezzo_F13b" "del\_Pezzo\_F13b" "x,y,z" \
					"x*x*x*x+y*y*y*y+z*z*z*z-x*x*y*y"   \
		-define del_Pezzo13 -collection "f3,f4" \
	-with P -do \
	-projective_space_activity \
	-analyze_del_Pezzo_surface del_Pezzo13 "" \
	-end
	#pdflatex del_Pezzo_F13a_report.tex
	#pdflatex del_Pezzo_F13b_report.tex
	#open del_Pezzo_F13a_report.pdf
	#open del_Pezzo_F13b_report.pdf



del_Pezzo_F13a_points.txt:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-define f1 -formula "del_Pezzo_F9" "del\_Pezzo\_F9" "x,y,z" "x*x*x*x+y*y*y*y+z*z*z*z"   \
		-define f2 -formula "del_Pezzo_F11" "del\_Pezzo\_F11" "x,y,z" "x*x*x*x+y*y*y*y+z*z*z*z+x*x*y*y+x*x*z*z+y*y*z*z"   \
		-define f3 -formula "del_Pezzo_F13a" "del\_Pezzo\_F13a" "x,y,z" "x*x*x*x+y*y*y*y+z*z*z*z+8*x*x*y*y+8*x*x*z*z+8*y*y*z*z"   \
		-define f4 -formula "del_Pezzo_F13b" "del\_Pezzo\_F13b" "x,y,z" "x*x*x*x+y*y*y*y+z*z*z*z-x*x*y*y"   \
		-define del_Pezzo9 -collection "f1" \
		-define del_Pezzo11 -collection "f2" \
		-define del_Pezzo13 -collection "f3,f4" \
	-with P -do \
	-projective_space_activity \
	-analyze_del_Pezzo_surface del_Pezzo13 "" \
	-end
	pdflatex del_Pezzo_F13a_report.tex
	pdflatex del_Pezzo_F13b_report.tex
	open del_Pezzo_F13a_report.pdf
	open del_Pezzo_F13b_report.pdf
	#dot -Tpng del_Pezzo_F13a.gv >del_Pezzo_F13a.png
	#open del_Pezzo_F13a.png

# writes del_Pezzo_F13a_points.txt




del_Pezzo_169:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 169 -end \
		-define P -projective_space 3 F -end \
		-define f3 -formula "del_Pezzo_F169a" "del\_Pezzo\_F169a" "x,y,z" "x*x*x*x+y*y*y*y+z*z*z*z+8*x*x*y*y+8*x*x*z*z+8*y*y*z*z"  \
		-define f4 -formula "del_Pezzo_F169b" "del\_Pezzo\_F169b" "x,y,z" "x*x*x*x+y*y*y*y+z*z*z*z-x*x*y*y"  \
		-define del_Pezzo13 -collection "f3,f4" \
	-with P -do \
	-projective_space_activity \
	-analyze_del_Pezzo_surface del_Pezzo13 ""
	pdflatex del_Pezzo_F169a_report.tex
	pdflatex del_Pezzo_F169b_report.tex
	open del_Pezzo_F169a_report.pdf
	open del_Pezzo_F169b_report.pdf




#############################################################################################################
# Section 4.5: The Klein Quadric and Pluecker coordinates


SECTION_KLEIN_QUADRIC_AND_PLUECKER_COORDINATES:



GR_4_2_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_Gr 4 2 -end
	pdflatex Gr_4_2_2.tex 
	open Gr_4_2_2.pdf


#############################################################################################################
# Section 4.6: Orthogonal spaces


SECTION_ORTHOGONAL_SPACES:



Op_4_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define O -orthogonal_space 1 4 F -without_group -end \
		-with O -do -orthogonal_space_activity \
			-cheat_sheet_orthogonal -end
	pdflatex O_1_4_2_report.tex 
	open O_1_4_2_report.pdf


Op_6_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define O -orthogonal_space 1 6 F -without_group -end \
		-with O -do -orthogonal_space_activity \
			-cheat_sheet_orthogonal -end
	pdflatex O_1_6_2_report.tex 
	open O_1_6_2_report.pdf

Op_6_2_with_group:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define O -orthogonal_space 1 6 F -end \
		-with O -do -orthogonal_space_activity \
			-cheat_sheet_orthogonal -end
	pdflatex O_1_6_2_report.tex 
	open O_1_6_2_report.pdf

# problem:
# error message:
#stabilizer_chain_base_data::allocate_base_data degree is too large


Op_6_64:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 64 -end \
		-define O -orthogonal_space 1 6 F -without_group -end \
		-with O -do -orthogonal_space_activity \
			-cheat_sheet_orthogonal -end
	pdflatex O_1_6_64_report.tex 
	open O_1_6_64_report.pdf


# problem:

Op_6_64_line_rank:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 64 -end \
		-define O -orthogonal_space 1 6 F -without_group -end \
		-with O -do -orthogonal_space_activity \
			-unrank_line_through_two_points 15447347 15225451 \
		-end

Op_6_8_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -end \
		-define O -orthogonal_space 1 6 F -without_group -end \
		-with O -do -orthogonal_space_activity \
			-cheat_sheet_orthogonal \
		-end
	pdflatex O_1_6_8_report.tex
	open O_1_6_8_report.pdf

#############################################################################################################
# Section 4.7: Hermitian varieties


SECTION_HERMITIAN_VARIETIES:



H_2_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_hermitian 2 -end
	pdflatex H_2_4.tex 
	open H_2_4.pdf

H_3_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity \
			-cheat_sheet_hermitian 3 -end
	pdflatex H_3_4.tex 
	open H_3_4.pdf


# H_3_4 = the Hirschfeld surface



#############################################################################################################
# Chapter 5 - Group Theory
#############################################################################################################


#############################################################################################################
# Section 5.1: Permutation groups


SECTION_PERMUTATION_GROUPS:

PGL_4_2a:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F -finite_field -q 2 -end \
		-define G -linear_group -PGL 4 F -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_4_2_report.tex
	open PGL_4_2_report.pdf



#############################################################################################################
# Section 5.2: Linear Groups


SECTION_LINEAR_GROUPS:


PGL_3_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F -finite_field -q 2 -end \
		-define G -linear_group -PGL 3 F -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_3_2_report.tex
	open PGL_3_2_report.pdf

PGL_4_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F -finite_field -q 2 -end \
		-define G -linear_group -PGL 4 F -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_4_2_report.tex
	open PGL_4_2_report.pdf




AGL_1_27:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F27 -finite_field -q 27 -end \
		-define G -linear_group -AGL 1 F27 -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex AGL_1_27_report.tex
	open AGL_1_27_report.pdf

#		-group_table \



PGL_4_5:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F -finite_field -q 5 -end \
		-define G -linear_group -PGL 4 F -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_4_5_report.tex
	open PGL_4_5_report.pdf





PGL_4_2_wd:
	$(ORBITER_PATH)orbiter.out -v 12 \
		-draw_options -radius 200 -end \
		-define G -linear_group -PGL 4 2 -wedge_detached -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_4_2_Wedge_4_0_detached_report.tex
	open PGL_4_2_Wedge_4_0_detached_report.pdf




PGL_4_2_wd_reverse:
	$(ORBITER_PATH)orbiter.out -v 12 \
		-linear_group -PGL 4 2 -wedge_detached -end \
		-group_theoretic_activities \
		-reverse_isomorphism_exterior_square \
		-end

	
PGGL_3_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-sylow -classes \
		-end
	pdflatex PGGL_3_4_report.tex
	open PGGL_3_4_report.pdf

#-sylow  
#		-export_schreier_trees \
#		-tools_path $(GRAPH_THEORY_PATH) \
#		-report \
#		-report_schreier_trees



PGO_5_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F -finite_field -q 2 -end \
		-define G -linear_group -PGO 5 F -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGO_5_2_report.tex
	open PGO_5_2_report.pdf

PGGO_5_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F4 -finite_field -q 4 -end \
		-define G -linear_group -PGGO 5 F4 -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGGO_5_4_report.tex
	open PGGO_5_4_report.pdf



PGOp_6_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F2 -finite_field -q 2 -end \
		-define G -linear_group -PGOp 6 F2 -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGOp_6_2_report.tex
	open PGOp_6_2_report.pdf

PGOm_6_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 200 -end \
		-define F2 -finite_field -q 2 -end \
		-define G -linear_group -PGOm 6 F2 -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGOm_6_2_report.tex
	open PGOm_6_2_report.pdf



#############################################################################################################
# Section 5.3: Subgroups

SECTION_SUBGROUPS:

J1:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 7 11 -Janko1 -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGL_7_11_Subgroup_Janko1_report.tex
	open PGL_7_11_Subgroup_Janko1_report.pdf

PGL_3_11_singer:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 3 11 -singer 19 -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGL_3_11_Singer_3_11_19_report.tex
	open PGL_3_11_Singer_3_11_19_report.pdf


PGL_3_11_singer_and_frobenius:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 3 11 -singer_and_frobenius 19 -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
		pdflatex PGL_3_11_Singer_and_Frob3_11_19_report.tex
		open PGL_3_11_Singer_and_Frob3_11_19_report.pdf

PG_2_4_order_21:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define G -linear_group -PGL 3 4  -end \
		-with G -do \
		-group_theoretic_activities -search_element_of_order 21




quaternion:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -radius 200 -end \
		-define G -linear_group -SL 2 3 \
		-subgroup_by_generators "quaternion" "8" 3 \
			"1,1,1,2" \
			"2,1,1,1" \
			"0,2,1,0" \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-group_table \
		-report \
		-end
	pdflatex GL_2_3_Subgroup_quaternion_8_elements.tex
	open GL_2_3_Subgroup_quaternion_8_elements.pdf
	pdflatex GL_2_3_Subgroup_quaternion_8_report.tex
	open GL_2_3_Subgroup_quaternion_8_report.pdf


cube_group:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -GL 3 3 \
		-subgroup_by_generators "cube" "48" 3 \
			"0,1,0,2,0,0,0,0,1" \
			"0,0,1,0,1,0,2,0,0" \
			"2,0,0,0,1,0,0,0,1" \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-report \
		-end
	pdflatex GL_3_3_Subgroup_cube_48_report.tex
	open GL_3_3_Subgroup_cube_48_report.pdf
	pdflatex GL_3_3_Subgroup_cube_48_elements.tex
	open GL_3_3_Subgroup_cube_48_elements.pdf


tetra_group:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -GL 3 3 \
		-subgroup_by_generators "tetra" "12" 2 \
			"0,1,0,0,0,1,1,0,0" \
			"0,0,1,2,0,0,0,2,0" \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-report \
		-end
	pdflatex GL_3_3_Subgroup_tetra_12_report.tex
	open GL_3_3_Subgroup_tetra_12_report.pdf
	pdflatex GL_3_3_Subgroup_tetra_12_elements.tex
	open GL_3_3_Subgroup_tetra_12_elements.pdf


Hesse_group:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGGL 3 4 \
		-subgroup_by_generators "Hesse" "432" 7 \
		"3,0,0,0,3,0,0,0,3,0" \
		"2,0,0,0,2,0,1,2,3,0" \
		"1,0,0,0,1,0,0,1,1,1" \
		"1,0,0,0,2,2,0,2,0,0" \
		"1,0,0,2,3,1,2,0,1,0" \
		"0,3,3,1,0,0,3,2,1,1" \
		"2,2,0,0,0,1,1,3,3,1" \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-print_elements_tex \
		-report \
		-end
	#pdflatex PGGL_3_4_Subgroup_Hesse_432_report.tex
	#open PGGL_3_4_Subgroup_Hesse_432_report.pdf

#Hesse group:
#1,0,0,0,0,1,0,0,0,0,1,0,3,2,3,2,0, 
#1,0,0,0,0,1,0,0,3,1,2,0,1,0,1,3,0, 
#1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1, 
#1,0,0,0,0,2,2,0,0,2,0,0,0,0,0,1,0, 
#1,0,0,0,2,3,1,0,2,0,1,0,3,1,3,1,0, 
#0,1,1,0,2,0,0,0,1,3,2,0,2,1,1,2,1, 
#1,1,0,0,0,0,3,0,3,2,2,0,1,2,3,3,1,




Weyl_E8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -GL 8 3 \
		-subgroup_by_generators "Weyl_E8" "696729600" 2 \
		"-1,-1,-1,-1,0,0,0,0, \
		0,0,0,1,0,0,0,0, \
		1,0,0,0,0,0,0,0, \
		0,0,1,0,0,0,0,0, \
		0,1,0,1,1,0,0,0, \
		0,0,0,0,0,1,0,0, \
		0,0,0,0,0,0,1,0, \
		0,0,0,0,0,0,0,1" \
		"-1,0,-1,-1,-1,-1,-1,-1, \
		0,1,0,1,1,1,1,1, \
		1,0,0,0,0,0,0,0, \
		0,0,1,0,0,0,0,0, \
		0,0,0,1,0,0,0,0, \
		0,0,0,0,1,0,0,0, \
		0,0,0,0,0,1,0,0, \
		0,0,0,0,0,0,1,0" \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex GL_8_3_Subgroup_Weyl_E8_696729600_report.tex
	open GL_8_3_Subgroup_Weyl_E8_696729600_report.pdf

# group generators from http://www.math.rwth-aachen.de/~Gabriele.Nebe/LATTICES/E8.b.html




#############################################################################################################
# Section 5.4:  Linear Groups, Advanced Topics


SECTION_LINEAR_GROUPS_ADVANCED_TOPICS:


U_3_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 9 -override_polynomial "17"  -end \
		-define G -linear_group -PGL 3 F \
			-subgroup_by_generators "U_3_3" "6048" 2 \
			"1,6,4, 5,0,6, 8,5,1" \
			"6,2,1, 7,8,4, 0,6,6" \
			-end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	#pdflatex PGL_3_9_Subgroup_U_3_3_6048_report.tex
	#open PGL_3_9_Subgroup_U_3_3_6048_report.pdf




PGL_2_3:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 2 3 -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
			-group_table \
		-end
	pdflatex PGL_2_3_group_table_order_24.tex
	pdflatex PGL_2_3_report.tex
	open PGL_2_3_group_table_order_24.pdf
	open PGL_2_3_report.pdf



PG_3_2_again:
	$(ORBITER_PATH)orbiter.out \
		-define F -finite_field -q 3 -end 
		-finite_field_activity \
		-cheat_sheet_PG 2 -end
	pdflatex PG_2_3.tex 
	open PG_2_3.pdf








#Co3 from Conway et al., 1985 (ATLAS)
#order = 495766656000
#Co3 from the paper by Suleiman and Wilson 1997

#1101110001000001010000
#1111010111110100001011
#0000001000000100010101
#1111100110110001001110
#0101010000000010011101
#0000010000000100010101
#0010000000000100010101
#0001000011000000111111
#1110100100110100010011
#0000000000000110010101
#0000000000100100010101
#0110111111010011101111
#0000000000001100010101
#0000000000000100000101
#0000000001000100010101
#0000000000000100011101
#0001000110000010011010
#0000000000000000010101
#0000000000000101010101
#0000000000000100010100
#0000000000000100010111
#0000000000000100010001


#0101000010111010111111
#0110010100011110110000
#0011010000111111010111
#0001101110001011010011
#1010010000100001011110
#1101000000001010100011
#1100101010001111010101
#1000110100110101010101
#0100110001010000000111
#1100000010100101010010
#0101110110011100000101
#0101111101010011111001
#1000010101010101010001
#0001010000111100100111
#0011010010111011001111
#0100110010110011111010
#1101011001111101100011
#0100101001001000100001
#1100101100001001110011
#0101110110010100000001
#0000001101111000101110
#1101101010101110000101

Co3:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -nodes_empty -end \
		-define G -linear_group -PGL 22 2 \
		-subgroup_by_generators "Co3" "495766656000" 2 \
			-long_string \
			"1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0, " \
			"1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1, " \
			"0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"1,1,1,1,1,0,0,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0, " \
			"0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,1, " \
			"0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1, " \
			"1,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1, " \
			"0,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,1,0, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1, " \
			"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1," \
			-end_string \
			-long_string \
			"0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1, " \
			"0,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,0,0, " \
			"0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,0,1,1,1, " \
			"0,0,0,1,1,0,1,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1, " \
			"1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0, " \
			"1,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,1, " \
			"1,1,0,0,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1, " \
			"1,0,0,0,1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1, " \
			"0,1,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,1, " \
			"1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0, " \
			"0,1,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,0,0,1,0,1, " \
			"0,1,0,1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,1, " \
			"1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1, " \
			"0,0,0,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,1,1,1, " \
			"0,0,1,1,0,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,1,1, " \
			"0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,0, " \
			"1,1,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,1, " \
			"0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1, " \
			"1,1,0,0,1,0,1,1,0,0,0,0,1,0,0,1,1,1,0,0,1,1, " \
			"0,1,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1, " \
			"0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,1,1,1,0, " \
			"1,1,0,1,1,0,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0,1," \
			-end_string \
			-end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGL_22_2_Subgroup_Co3_495766656000_report.tex
	open PGL_22_2_Subgroup_Co3_495766656000_report.pdf



Ree_27:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -end \
		-define F -finite_field -q 27 -override_polynomial "34" -end \
		-define G -linear_group -PGL 7 F \
			-subgroup_by_generators "Ree_27" "10073444472" 2 \
			-long_string \
			"21,5,1,6,17,1,1, 3,13,5,21,6,6,18, 21,3,21,21,22,6,14, " \
			"14,18,1,5,13,6,7, 3,3,2,1,24,16,3, 17,3,22,10,16,24,26, " \
			"21,21,6,18,20,2,5" \
			-end_string \
			-long_string \
			"16,3,11,5,16,22,20, 24,6,18,24,7,1,26, 9,23,17,18,23,20,13, " \
			"9,7,2,15,17,5,11, 3,3,6,21,4,24,16, 25,8,6,24,21,12,7, " \
			"24,15,2,13,11,14,24" \
			-end_string \
			-end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end


#############################################################################################################
# Section 5.5:  Induced Actions


SECTION_INDUCED_ACTIONS:



T3_on_tensors:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 3 -on_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex GL_2_2_wreath_Sym3_report.tex
	open GL_2_2_wreath_Sym3_report.pdf
	


T3r1:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 3 -on_rank_one_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex GL_2_2_wreath_Sym3_report.tex
	open GL_2_2_wreath_Sym3_report.pdf





T4_on_tensors:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex GL_2_2_wreath_Sym4_report.tex
	open GL_2_2_wreath_Sym4_report.pdf
	
	
T4r1:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_rank_one_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex GL_2_2_wreath_Sym4_report.tex
	open GL_2_2_wreath_Sym4_report.pdf


PGGL_2_8_on_conic:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -radius 250 -end \
		-define G \
		-linear_group -PGGL 2 8 -PGL2OnConic -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGGL_2_8_OnConic_2_8_report.tex
	open PGGL_2_8_OnConic_2_8_report.pdf



#############################################################################################################
# Section 5.6:  Group Theoretic Activities


SECTION_GROUP_THEORETIC_ACTIVITIES:

PGL_3_4_singer:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define G -linear_group -PGL 3 4 -end \
		-with G -do \
		-group_theoretic_activities -find_singer_cycle -end




GL_2_8_multiply:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define G -linear_group -GL 2 8 -end \
		-with G -do \
		-group_theoretic_activities -multiply "0,1,2,3" "4,5,6,7" -end
	pdflatex GL_2_8_mult.tex
	open GL_2_8_mult.pdf

GL_2_7_multiply:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define G -linear_group -GL 2 7 -end \
		-with G -do \
		-group_theoretic_activities -multiply "0,1,2,3" "4,5,6,0" -end
	pdflatex GL_2_7_mult.tex
	open GL_2_7_mult.pdf


GL_2_7_inv:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define G -linear_group -GL 2 7 -end \
		-with G -do \
		-group_theoretic_activities -inverse "0,1,2,3" -end
	pdflatex GL_2_7_inv.tex
	open GL_2_7_inv.pdf

GL_2_7_power:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define G -linear_group -GL 2 7 -end \
		-with G -do \
		-group_theoretic_activities -raise_to_the_power "0,1,2,3" 2 -end
	pdflatex GL_2_7_power.tex
	open GL_2_7_power.pdf



PGL_3_2_classes:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define G -linear_group -PGL 3 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-classes_based_on_normal_form \
		-end
	pdflatex PGL_3_2_classes_normal_form.tex
	open PGL_3_2_classes_normal_form.pdf
	#pdflatex PGL_3_2_classes_out.tex
	#open PGL_3_2_classes_out.pdf

#		-classes \




normal_forms_PGL_4_4:
	$(ORBITER_PATH)orbiter.out -v 7 \
		-define G -linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-classes_based_on_normal_form \
		-end
	pdflatex PGGL_4_4_classes_normal_form.tex
	open PGGL_4_4_classes_normal_form.pdf
	



PGL_4_4_2A_rank:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G -linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-element_rank "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" \
		-end
		
PGL_4_4_2A_unrank:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G -linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-element_unrank "1" \
		-end
		

cent_2A:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-linear_group -PGGL 4 4 \
		-subgroup_by_generators "centralizer_2A" "40320" 10 \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0," \
			"1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0," \
			"1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0," \
			"1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0," \
			"1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0," \
			"0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1," \
		-end




PGL_4_5_3B_rank:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G -linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-element_rank "0,0,0,1, 2,3,0,1, 0,3,4,4, 0,1,2,1"
		
PGL_4_5_3B_unrank:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G -linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-element_unrank "701459351"
		

#eigen_3A:
#	$(ORBITER_PATH)orbiter.out -v 6  -finite_field_activity -q 5 -eigenstuff 4 "0,1,0,2,0,1,2,1,4,2,3,1,2,0,4,3" -end
	
#eigen_3B:
#	$(ORBITER_PATH)orbiter.out -v 6  -finite_field_activity -q 5 -eigenstuff 4 "0,0,0,1,2,3,0,1,0,3,4,4,0,1,2,1" -end
	

# element of order 31 in PGL(4,5):
#		int data[] = {1,0,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,3, 0 };


normal_forms_PGL_4_5:
	$(ORBITER_PATH)orbiter.out -v 7 \
		-define G -linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-classes_based_on_normal_form
	pdflatex PGL_4_5_classes_normal_form.tex
	open PGL_4_5_classes_normal_form.pdf





#############################################################################################################
# Section 5.7:  Group Theoretic Activities Based on Magma


SECTION_GROUP_THEORETIC_ACTIVITIES_BASED_ON_MAGMA:


PGGL_2_4_classes:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define G \
		-linear_group -PGGL 2 4 \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-classes \
		-end
	$(MAGMA_PATH)/magma PGGL_2_4_classes.magma
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define G \
		-linear_group -PGGL 2 4 \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-classes \
		-end
	pdflatex PGGL_2_4_classes_out.tex
	open PGGL_2_4_classes_out.pdf



PGGL_2_4_cent_2A:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 2 4 -end \
		-with G -do \
		-group_theoretic_activities \
			-centralizer_of_element "2A" "1,0, 0,1, 1" \
			-report \
		-end
	$(MAGMA_PATH)/magma element_2A_centralizer.magma
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 2 4 -end \
		-with G -do \
		-group_theoretic_activities \
			-centralizer_of_element "2A" "1,0, 0,1, 1" \
			-report \
		-end
	pdflatex PGGL_2_4_elt_2A_centralizer.tex
	open PGGL_2_4_elt_2A_centralizer.pdf
	






PGGL_3_4_classes:
	$(ORBITER_PATH)orbiter.out -v 10 \
		-define G \
		-linear_group -PGGL 3 4 \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-classes
	pdflatex PGGL_3_4_classes_out.tex
	open PGGL_3_4_classes_out.pdf


#1,3,3,1,3,2,3,0,3,0,
# is an element of order 21


#		-subgroup_by_generators "singer" "21" 1 \
#		"0,1,0, 0,0,1, 2,1,1" \





classes_PGGL_4_4:
	$(ORBITER_PATH)orbiter.out \
		-magma_path /usr/local/magma/ \
		-v 6 \
		-define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-classes

# group order 1974067200 = 2^13 * 3^4 * 5^2 * 7 * 17


PGGL_4_4_cent_2A:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
			-centralizer_of_element "2A" "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" \
			 \
		-end
	#pdflatex PGGL_4_4_report.tex
	#open PGGL_4_4_report.pdf
	pdflatex PGGL_4_4_elt_2A_centralizer.tex
	open PGGL_4_4_elt_2A_centralizer.pdf
	

#Strong generators for a group of order 40320:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,0,\\
#1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,\\
#1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,\\
#1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0,\\
#1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,0,\\
#0,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,0,\\


PGGL_4_4_cent_2A_recreate:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -radius 200 -end \
		-define G -linear_group -PGGL 4 4 \
			-subgroup_by_generators "centralizer_2A" "40320" 9 \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,0," \
			"1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0," \
			"1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0," \
			"1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0," \
			"1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,0," \
			"0,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,0," \
		-end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGGL_4_4_Subgroup_centralizer_2A_40320_report.tex
	open PGGL_4_4_Subgroup_centralizer_2A_40320_report.pdf


PGL_4_4_2A_class:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-conjugacy_class_of "1"





PGGL_4_4_cent_2B:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-centralizer_of_element "2B" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-report \
		-end

#Strong generators for a group of order 368640:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,2,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,3,0,1,3,1,\\
#1,0,0,0,0,1,0,0,1,0,1,0,3,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,3,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,3,2,0,0,3,1,\\
#1,0,0,0,0,1,0,0,1,0,0,3,3,0,2,3,0,\\
#1,0,0,0,1,1,0,0,2,0,1,0,2,0,0,1,0,\\
#1,0,0,0,1,1,0,2,2,0,1,3,2,0,0,2,0,\\
#1,0,0,0,2,1,1,2,1,0,2,1,3,0,2,2,0,\\
#1,0,0,0,2,1,0,2,0,0,1,0,2,0,0,3,1,\\

PGGL_4_4_cent_2C:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-centralizer_of_element "2C" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0"


#Strong generators for a group of order 30720:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,\\
#1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0,\\
#1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1,\\



PGGL_4_4_norm_4A:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4A" "1,0,0,0,1,1,0,0, 0,1,1,0, 0,0,1,1, 0"


# normalizer has order 256:
#Strong generators for a group of order 256:\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,3,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,\\
#1,0,0,0,0,1,0,0,3,1,1,0,1,2,0,1,0,\\
#1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0,\\
#1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,\\
#1,0,0,0,2,1,0,0,1,3,1,0,2,2,2,1,0,\\



PGGL_4_4_norm_4B:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4B" "1,0,0,0,0,1,0,0, 0,1,1,0, 0,0,1,1, 0"

#Strong generators for a group of order 3072:\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,2,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,2,1,1,0,\\
#1,0,0,0,0,2,0,0,0,2,2,0,0,3,0,2,0,\\
#1,0,0,0,0,3,0,0,0,1,3,0,0,2,1,3,1,\\
#1,2,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,\\




PGGL_4_4_norm_4C:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4C" "3,0,0,0, 1,3,0,0, 0,0,3,0, 0,0,1,3, 1"



#Strong generators for a group of order 384:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,\\
#1,0,0,0,2,1,0,0,0,0,1,0,1,0,2,1,0,\\
#1,0,0,0,3,1,1,0,0,0,1,0,1,0,2,1,0,\\
#0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0,1,\\



PGGL_4_4_norm_4D:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4D" "3,0,0,0, 0,3,0,0, 0,0,3,0, 0,0,1,3, 1"


#Strong generators for a group of order 768:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,2,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,2,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,1,2,1,0,\\
#1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,\\
#1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,\\
#0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,\\





PGGL_4_4_cent_5A:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-centralizer_of_element "5A" "0,2,0,0, 1,1,0,0, 0,0,3,0, 0,0,0,3, 0"

#centralizer has order 900
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,3,0,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,3,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,2,0,0,0,2,0,\\
#0,1,0,0,3,3,0,0,0,0,2,0,0,0,0,2,0,\\

PGGL_4_4_norm_5A:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "5A" "0,2,0,0, 1,1,0,0, 0,0,3,0, 0,0,0,3, 0"

# normalizer has order 3600
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,0,2,0,0,2,0,0,\\
#1,0,0,0,2,2,0,0,0,0,1,0,0,0,0,1,1,\\
#0,1,0,0,3,3,0,0,0,0,2,0,0,0,0,2,0,\\


PGGL_4_4_norm_5B:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "5B" "0,2,0,0,1,1,0,0,0,0,0,2,0,0,1,1,0"

# normalizer has order 81600:
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,2,2,2,2,1,3,1,3,0,\\
#1,0,0,0,0,1,0,0,0,0,1,1,0,0,3,2,0,\\
#1,0,0,0,2,2,0,0,0,0,3,0,0,0,1,1,1,\\
#0,1,0,0,3,3,0,0,0,0,0,1,0,0,3,3,0,\\
#0,0,1,0,0,0,1,2,2,0,0,0,2,3,0,0,1,\\

PGGL_4_4_norm_5C:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "5C" "0,1,0,0,3,3,0,0,0,0,1,2,0,0,2,0,0"


# normalizer has order 600:
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,2,0,0,2,0,0,\\
#1,0,0,0,2,2,0,0,0,0,3,0,0,0,3,1,1,\\
#0,1,0,0,3,3,0,0,0,0,1,2,0,0,2,0,0,\\
#0,0,1,0,0,0,3,2,1,0,0,0,3,2,0,0,1,\\


subgroups_PGGL_4_4:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-find_subgroup 4 \
		-end
	pdflatex PGGL_4_4_report.tex
	open PGGL_4_4_report.pdf







# the -find_subgroup  command is too specialized

subgroups_PGL_4_5:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-find_subgroup 3 \
		-end
	pdflatex PGL_4_5_report.tex
	open PGL_4_5_report.pdf


classes_PGL_4_5:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-classes \
		-end
	pdflatex PGL_4_5_classes_out.tex
	open PGL_4_5_classes_out.pdf
	
# 163 classes


# two classes of elements of order 3
#Order of element = 3 Class size = 310000 Centralizer order = 93600 Normalizer order = 187200
#of order 3 and with 0 fixed points.
#0,1,0,2,0,1,2,1,4,2,3,1,2,0,4,3,

#Class size = 10075000 Centralizer order = 2880 Normalizer order = 5760
#of order 3 and with 6 fixed points.
#0,0,0,1,2,3,0,1,0,3,4,4,0,1,2,1,

PGL_4_5_3B_class_again:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-conjugacy_class_of "0,0,0,1, 2,3,0,1, 0,3,4,4, 0,1,2,1"
		

search_primitive_poly_q5_deg3:
	$(ORBITER_PATH)orbiter.out -v 6 -search_for_primitive_polynomial_in_range 5 5 3 3

#OK, we found an irreducible and primitive polynomial X^{3} + X^{2} + 2

GL_3_5_singer_power:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -GL 3 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-raise_to_the_power "0,1,0, 0,0,1, 3,0,4" 31
	pdflatex GL_3_5_power.tex
	open GL_3_5_power.pdf

PGL_4_5_norm_31:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "31" \
			"2,0,0,0, 0,0,1,0, 0,0,0,1, 0,3,0,4"
		-end
	pdflatex normalizer_of_31_in_PGL_4_5.tex
	open normalizer_of_31_in_PGL_4_5.pdf

# normalizer has order 372
#1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4, 
#1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3, 
#1,0,0,0,0,4,0,0,0,0,2,1,0,3,2,4, 
#1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3,

#		-normalizer_of_cyclic_subgroup "31" "1,0,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,3"
# normalizer has order 372
#1,0,0,0,0,4,0,0,0,0,4,0,0,0,0,4,\\
#1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,3,\\
#1,0,0,0,0,1,0,0,0,3,1,1,0,4,2,3,\\
#1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,3,\\


Normalizer_of_Z22_in_PGL_2_9:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -linear_group -PGL 2 9 \
		-subgroup_by_generators Z22 4 2 "2,0,0,1" "0,1,1,0" -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer \
		-end
	pdflatex PGL_2_9_Subgroup_Z22_4_normalizer.tex
	open PGL_2_9_Subgroup_Z22_4_normalizer.pdf



#############################################################################################################
# Chapter 6 - Orbit Algorithms
#############################################################################################################


#############################################################################################################
# Section 6.1:  Orbit Algorithms


SECTION_ORBIT_ALGORITHMS_SCHREIER_TREES:


T3r1_orbits:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 3 -on_rank_one_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
			-orbits_on_points \
			-export_trees \
		-end
	pdflatex GL_2_2_wreath_Sym3_res27_orbits.tex
	open GL_2_2_wreath_Sym3_res27_orbits.pdf
		
T3r1_orbits_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph GL_2_2_wreath_Sym3_res27_0.layered_graph \
		-radius 500 -spanning_tree -embedded \
			-line_width 1.1  -x_stretch 1.4 -scale 0.25 \
		-end
	#pdflatex GL_2_2_wreath_Sym3_report.tex
	#open GL_2_2_wreath_Sym3_report.pdf
	pdflatex GL_2_2_wreath_Sym3_res27_0_draw.tex
	open GL_2_2_wreath_Sym3_res27_0_draw.pdf


# write GL_2_2_wreath_Sym3_res27_0.layered_graph



2C_orbit_under_PGGL_4_4_elements_coded.csv: 
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G -linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
		-conjugacy_class_of_element "2C" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
		-end

# class of size 64260
# creates:
# 2C_orbit_under_PGGL_4_4.csv
# 2C_orbit_under_PGGL_4_4.txt
# 2C_orbit_under_PGGL_4_4_elements_coded.csv
# 2C_orbit_under_PGGL_4_4_transporter.csv
# 1:33 on Mac

PGGL_4_4_subgroups_of_type_2C_2C: 2C_orbit_under_PGGL_4_4_elements_coded.csv
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G -linear_group -PGGL 4 4 \
		-subgroup_by_generators "centralizer_2C" "30720" 9 \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
			"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
			"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
			"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
			"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
			"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_group_elements_under_conjugation \
			2C_orbit_under_PGGL_4_4_elements_coded.csv \
			2C_orbit_under_PGGL_4_4_transporter.csv \
		-end
	pdflatex subgroups_of_order_4.tex
	open subgroups_of_order_4.pdf


#The distribution of orbit lengths is: ( 1, 2, 15, 20, 24^3, 30, 40, 240, 256, 480, 512, 960^2, 1280, 1920^2, 2560^4, 3840, 5120, 6144^3, 7680, 10240 )
#group_theoretic_activity::do_orbits_on_group_elements_under_conjugation after Classes.compute_all_point_orbits
#found 29 conjugacy classes
#User time: 0:57



orbits_on_conics_q13:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G -linear_group -PGL 3 13 -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 2 \
		-end 
	pdflatex poly_orbits_d2_n2_q13.tex
	open poly_orbits_d2_n2_q13.pdf


orbits_cubic_curves_q2:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G -linear_group -PGL 3 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-end 
	pdflatex poly_orbits_d3_n3_q2.tex
	open poly_orbits_d3_n3_q2.pdf


orbits_cubic_curves_q2_with_draw_tree:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -yout 500000 -radius 15 -nodes_empty \
			-line_width 0.5 -y_stretch 0.25 -embedded -end \
		-define G -linear_group -PGL 3 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-orbits_on_polynomials_draw_tree 6 \
		-end 


poly_orbits_d3_n3_q2.csv:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -yout 500000 -radius 15 -nodes_empty \
			-line_width 0.5 -y_stretch 0.25 -embedded -end \
		-define G -linear_group -PGL 4 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-orbits_on_polynomials_draw_tree 6 \
		-end 

poly_orbits_d3_n3_q2_get_ranks:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-csv_file_select_cols poly_orbits_d3_n3_q2.csv 0
	#pdflatex poly_orbits_d3_n3_q2.tex
	#open poly_orbits_d3_n3_q2.pdf



T4_orbits:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
			-orbits_on_points \
		-end
	pdflatex GL_2_2_wreath_Sym4_res65535_orbits.tex
	open GL_2_2_wreath_Sym4_res65535_orbits.pdf
	#pdflatex GL_2_2_wreath_Sym4_report.tex
	#open GL_2_2_wreath_Sym4_report.pdf



T4r1_orbits:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G -linear_group -GL_d_q_wr_Sym_n 2 2 4 \
			-on_rank_one_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-orbits_on_points\
			-export_trees \
			-report \
		-end
	pdflatex GL_2_2_wreath_Sym4_orbits_report.tex
	open GL_2_2_wreath_Sym4_orbits_report.pdf


T4r1_orbits_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph GL_2_2_wreath_Sym4_res81_0.layered_graph \
		-radius 400 -spanning_tree -embedded \
			-line_width 1.1  -x_stretch 2.5 -scale 0.15 \
		-end
	#pdflatex GL_2_2_wreath_Sym3_report.tex
	#open GL_2_2_wreath_Sym3_report.pdf
	pdflatex GL_2_2_wreath_Sym4_res81_0_draw.tex
	open GL_2_2_wreath_Sym4_res81_0_draw.pdf


T4r1_orbits_4:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -GL_d_q_wr_Sym_n 2 2 4 \
			-on_rank_one_tensors -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label T4r1 -W \
				-bit_depth 4 -draw_options -end -draw_poset -report -end \
			-end \
			-orbits_on_subsets 4 \
			-report \
		-end
	pdflatex T4r1_poset.tex
	open T4r1_poset.pdf
	#pdflatex GL_2_2_wreath_Sym4_report.tex
	#open GL_2_2_wreath_Sym4_report.pdf



PGGL_2_8_on_conic_orbits:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -radius 250 -end \
		-define G \
		-linear_group -PGGL 2 8 -PGL2OnConic -end \
		-with G -do \
		-group_theoretic_activities \
			-orbits_on_points\
			-report \
		-end
	pdflatex PGGL_2_8_OnConic_2_8_orbits_report.tex
	open PGGL_2_8_OnConic_2_8_orbits_report.pdf



# example from the Fining manual, page 107:

PGGL_7_8_orbits:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-draw_options -end \
		-define G \
		-linear_group -PGGL 7 8 -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
			-orbits_on_points \
		-end

# 1 min 31 sec on Mac



#############################################################################################################
# Section 6.1: Poset Classification


SECTION_POSET_CLASSIFICATION:

poset_of_4subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options  -end \
		-define G -linear_group -PGL 2 3 -identity_group -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label poset_4 \
				-W -depth 4 -draw_options -radius 200 -end -report -end -draw_poset \
			-end \
			-orbits_on_subsets 4 \
			-report \
		-end
	pdflatex PGL_2_3_Identity_2_3_report.tex
	pdflatex poset_4_poset.tex
	open PGL_2_3_Identity_2_3_report.pdf
	open poset_4_poset.pdf

poset_of_4subsets_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph poset_4_poset_lvl_4.layered_graph \
		-radius 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
		-end
	pdflatex poset_4_poset_lvl_4_draw.tex
	open poset_4_poset_lvl_4_draw.pdf


poset_of_5subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options  -end \
		-define G -linear_group -PGL 2 4 -identity_group -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label poset_5 \
				-W -depth 5 -draw_options -radius 150 -end \
				-report -end -draw_poset -end \
			-orbits_on_subsets 5 \
			-report \
		-end
	pdflatex poset_5_poset.tex
	open poset_5_poset.pdf

poset_of_5subsets_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph poset_5_poset_lvl_5.layered_graph \
		-radius 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
		-end
	pdflatex poset_5_poset_lvl_5_draw.tex
	open poset_5_poset_lvl_5_draw.pdf



V_3_2_trivial:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 3 2 -identity_group -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label V_3_2_trivial \
			-W -depth 3 -node_label_is_element \
			-draw_options -radius 200 -embedded -end -report -end -draw_poset -end \
		-orbits_on_subspaces 3 \
		-report \
		-end
	#$(ORBITER_PATH)orbiter.out -v 5 \
	#	-draw_layered_graph PGL_3_2_Identity_3_2_poset_lvl_3.layered_graph \
	#			-radius 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
	#			-end
	#pdflatex PGL_3_2_Identity_3_2_report.tex
	#open PGL_3_2_Identity_3_2_report.pdf
	#pdflatex PGL_3_2_Identity_3_2_poset_lvl_3_draw.tex
	#open PGL_3_2_Identity_3_2_poset_lvl_3_draw.pdf
	pdflatex PGL_3_2_Identity_3_2_poset.tex
	open PGL_3_2_Identity_3_2_poset.pdf




V_4_2_trivial:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -identity_group -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label V_4_2_trivial \
			-W -depth 4 -node_label_is_element \
			-draw_options -radius 200 -embedded -end -report -end -draw_poset -end \
		-orbits_on_subspaces 4 \
		-report \
		-end
	#pdflatex PGL_4_2_Identity_4_2_report.tex
	#open PGL_4_2_Identity_4_2_report.pdf
	pdflatex PGL_4_2_Identity_4_2_tree_lvl_4.tex
	open PGL_4_2_Identity_4_2_tree_lvl_4.pdf
	pdflatex PGL_4_2_Identity_4_2_poset.tex
	open PGL_4_2_Identity_4_2_poset.pdf




#############################################################################################################
# Section 6.3: Orbits on Subsets


SECTION_ORBITS_ON_SUBSETS:

PG_2_2_subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 3 2 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label PGL_3_2 \
			-depth 7 -draw_poset -draw_options \
			-radius 200 -embedded -end -report -end -end \
			-orbits_on_subsets 7 \
			-report \
		-end
	pdflatex PGL_3_2_poset_lvl_7.tex
	open PGL_3_2_poset_lvl_7.pdf
	pdflatex PGL_3_2_poset.tex
	open PGL_3_2_poset.pdf
	#pdflatex PGL_3_2_poset_detailed_lvl_7.tex
	#open PGL_3_2_poset_detailed_lvl_7.pdf


# PG(3,2) has 2^3+2^2+2^1+1 = 15 points:
# PG(3,3) has 3^3+3^2+3^1+1 = 27 + 9 + 3 + 1 = 40 points.


PG_3_2_subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define F2 -finite_field -q 2 -end \
		-define G -linear_group -PGL 4 F2 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label PGL_4_2 \
				-depth 15 -draw_poset -draw_options \
				-radius 200 -embedded -end -report -end \
			-end \
			-orbits_on_subsets 15 \
			-report \
		-end
	pdflatex PGL_4_2_poset_lvl_15.tex
	open PGL_4_2_poset_lvl_15.pdf
	pdflatex PGL_4_2_poset.tex
	pdflatex PGL_4_2_poset.tex
	open PGL_4_2_poset.pdf
	#pdflatex PGL_4_2_poset_detailed_lvl_15.tex
	#open PGL_4_2_poset_detailed_lvl_15.pdf


PG_3_3_subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 4 3 -end \
		-with G -do \
			-group_theoretic_activities \
				-poset_classification_control -problem_label PGL_4_3 \
					-depth 14 -draw_poset \
					-draw_options -radius 200 -embedded -end \
				-end \
				-orbits_on_subsets 40 \
				-report \
			-end
	pdflatex PGL_4_3_poset_lvl_14.tex
	open PGL_4_3_poset_lvl_14.pdf



	

GL_2_2_singer_on_subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -GL 2 2 -singer 1 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-problem_label GL_2_2_singer -W -depth 4 \
				-draw_poset \
				-report -end \
				-draw_options -end \
			-end \
			-orbits_on_subsets 4 \
			-report \
		-end
	pdflatex GL_2_2_singer_poset.tex
	open GL_2_2_singer_poset.pdf


PGGL_2_16_on_subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGGL 2 16 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-problem_label PGGL_2_16 -W -depth 10 \
				-draw_poset \
				-report -end \
				-draw_options -end \
			-end \
			-orbits_on_subsets 10 \
			-report \
		-end
	pdflatex PGGL_2_16_poset.tex
	open PGGL_2_16_poset.pdf







PG_3_4_subsets:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control -problem_label PGGL_4_4 \
				-depth 5 \
				-draw_poset -draw_options -radius 200 -end \
				-report -end \
			-end \
			-orbits_on_subsets 5 \
			-report \
		-end
	pdflatex PGGL_4_4_poset.tex
	open PGGL_4_4_poset.pdf



PGGL_2_9_orbits:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGGL 2 9 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-problem_label PGGL_2_9 -W -depth 5 \
				-report -end \
				-draw_poset -draw_options -radius 200 -end \
			-end \
			-orbits_on_subsets 5 \
			-report \
		-end
	pdflatex PGGL_2_9_poset.tex
	open PGGL_2_9_poset.pdf



PG_3_7_lines:
	$(ORBITER_PATH)orbiter.out -v 7 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define G -linear_group -PGL 4 F -on_k_subspaces 2 -end \
		-with G -do \
			-group_theoretic_activities \
				-poset_classification_control -problem_label 7_lines \
				-depth 3 -report -end \
				-draw_poset \
				-draw_options -radius 200 -embedded -end \
			-end \
			-orbits_on_subsets 3 \
			-report \
		-end
	pdflatex 7_lines_poset.tex
	open 7_lines_poset.pdf


#############################################################################################################
# Section 6.4: Orbits on Subspaces


SECTION_ORBITS_ON_SUBSPACES:
	

subspaces_Op_4_2:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options  -end \
		-define G -linear_group -PGL 4 2 -orthogonal 1 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control \
			-node_label_is_element \
			-draw_poset -draw_options -radius 200 -end \
			-problem_label Op_4_2 -W -depth 4 -report -end \
		-end \
		-orbits_on_subspaces 4 \
		-report \
		-end
	pdflatex PGL_4_2_Orthogonal_plus_4_2_poset.tex
	open PGL_4_2_Orthogonal_plus_4_2_poset.pdf



#	$(ORBITER_PATH)orbiter.out -v 5 \
#		-draw_layered_graph PGL_4_2_Orthogonal_plus_4_2_poset_lvl_4.layered_graph \
#			-radius 300  -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
#			-end
#	pdflatex PGL_4_2_Orthogonal_plus_4_2_poset_lvl_4_draw.tex
#	open PGL_4_2_Orthogonal_plus_4_2_poset_lvl_4_draw.pdf
#	pdflatex PGL_4_2_Orthogonal_plus_4_2_report.tex
#	open PGL_4_2_Orthogonal_plus_4_2_report.pdf


PGL_4_2_on_subspaces:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-problem_label PGL_4_2 -W -depth 4 \
				-report -end \
			-end \
			-orbits_on_subspaces 4 \
			-report \
		-end
	pdflatex PGL_4_2_report.tex
	open PGL_4_2_report.pdf
	pdflatex PGL_4_2_poset.tex
	open PGL_4_2_poset.pdf


	
PGL_4_2_singer_on_subspaces:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -singer 1 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-node_label_is_element \
				-draw_poset \
				-draw_options -end \
				-problem_label PGL_4_2_singer -W -depth 4 \
				-report -end \
			-end \
			-orbits_on_subspaces 4 \
			-report \
		-end
	pdflatex PGL_4_2_Singer_4_2_1_poset.tex
	open PGL_4_2_Singer_4_2_1_poset.pdf



PGL_8_2_singer_on_subspaces:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 8 2 -singer 1 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-node_label_is_element \
				-draw_poset -draw_options -radius 150 -end \
				-problem_label PGL_8_2_singer -W -depth 8 -report -end \
			-end \
		-orbits_on_subspaces 8 \
		-report \
		-end
	pdflatex PGL_8_2_Singer_8_2_1_poset.tex
	open PGL_8_2_Singer_8_2_1_poset.pdf

# May 7, 2020: 16 sec on Mac
# 1643 orbits in total

Op_6_2_orbits_on_subspaces:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 6 2 -orthogonal 1 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-node_label_is_element \
				-draw_poset -draw_options -radius 200 -end \
				-problem_label Op_6_2 -W -depth 6 -report -end \
			-end \
		-orbits_on_subspaces 6 \
		-report \
		-end
	pdflatex PGL_6_2_Orthogonal_plus_6_2_report.tex
	open PGL_6_2_Orthogonal_plus_6_2_report.pdf
	pdflatex PGL_6_2_Orthogonal_plus_6_2_poset.tex
	open PGL_6_2_Orthogonal_plus_6_2_poset.pdf



Op_6_3_orbits_on_subspaces:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 6 3 -orthogonal 1 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-node_label_is_element \
				-draw_poset -draw_options -radius 200 -end \
				-problem_label Op_6_3 -W -depth 6 -report -end \
			-end \
		-orbits_on_subspaces 6 \
		-report \
		-end
	pdflatex PGL_6_3_Orthogonal_plus_6_3_report.tex
	open PGL_6_3_Orthogonal_plus_6_3_report.pdf

# June 3, 2020 on Mac: 0 sec



Op_6_11_orbits_on_subspaces:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -nodes_empty -end \
		-define G -linear_group -PGL 6 11 -orthogonal 1 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-node_label_is_element \
				-draw_poset -draw_options -radius 200 -end \
				-problem_label Op_6_11 -W -depth 6 -report -end \
			-end \
		-orbits_on_subspaces 6 \
		-report \
		-end
	pdflatex PGL_6_11_Orthogonal_plus_6_11_report.tex
	open PGL_6_11_Orthogonal_plus_6_11_report.pdf


# June 3, 2020 on Mac: 12 sec


Op_8_2_orbits_on_subspaces:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGL 8 2 -orthogonal 1 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-node_label_is_element \
				-draw_poset -draw_options -radius 200 -end \
				-problem_label Op_8_2 -W -depth 8 -report -end \
			-end \
		-orbits_on_subspaces 8 \
		-report \
		-end
	pdflatex PGL_8_2_Orthogonal_plus_8_2_poset.tex
	open PGL_8_2_Orthogonal_plus_8_2_poset.pdf



#############################################################################################################
# Section 6.5: Creating Objects in Projective Geometry

SECTION_CREATING_OBJECTS_IN_PG:




elliptic_curve_b1_c3_q11.txt:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object EC \
				-elliptic_curve 1 3 \
			-end \
		-end \
		-with EC -do -combinatorial_object_activity -save \
		-end





Hirschfeld_surface_q16.txt:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object Hirschfeld_surface_q16 \
				-projective_variety "Hirschfeld_surface_q16" 3 "1,6,1,8,1,11,1,13" \
				-monomial_type_PART \
			-end \
		-end \
		-with Hirschfeld_surface_q16 -do -combinatorial_object_activity -save \
		-end





#############################################################################################################
# Section 6.6: Canonical Forms of Objects in Finite Projective Spaces


SECTION_CANONICAL_FORMS:


EC_canon: elliptic_curve_b1_c3_q11.txt
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
			-input \
				-file_of_points elliptic_curve_b1_c3_q11.txt \
			-end \
			-classification_prefix EC \
			-report \
		-end \
		-end
	pdflatex EC_classification.tex
	open EC_classification.pdf





arcs_4_2_classify: arcs_4_2_lift.solutions
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG  \
		-input -file_of_points arcs_4_2_lift.solutions \
		-end \
		-classification_prefix arcs_4_2_lift \
		-report \
		-end \
		-end
	pdflatex arcs_4_2_lift_classification.tex
	open arcs_4_2_lift_classification.pdf





Hirschfeld4c: Hirschfeld_surface_q4.txt
	$(ORBITER_PATH)orbiter.out -v 10 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input -file_of_points Hirschfeld_surface_q4.txt -end \
				-classification_prefix Hirschfeld_surface_q4 \
				-report \
			-end \
		-end
	pdflatex Hirschfeld_surface_q4_classification.tex
	open Hirschfeld_surface_q4_classification.pdf


# group order is 51840

HIRSCHFELD_SURFACE_Q4_SET_OF_POINTS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,23,26,27,\
30,31,34,35,38,39,42,47,48,51,52,53,54,59,60,61,62,67,68,69,70,75,76,79,80,81,82"

Hirschfeld4c_set:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input \
					-set_of_points $(HIRSCHFELD_SURFACE_Q4_SET_OF_POINTS) \
				-end \
				-classification_prefix Hirschfeld_surface_q4 \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling Hirschfeld_surface_q4_can_lab.csv \
				-save_cumulative_data Hirschfeld_surface_q4_data.csv \
				-report \
			-end \
		-end 
	pdflatex Hirschfeld_surface_q4_classification.tex
	open Hirschfeld_surface_q4_classification.pdf



Dickson_set_stabilizer:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input \
					-set_of_points "0,1,2,5,6" \
					-set_of_points "0,1,2,3,6" \
					-set_of_points "0,1,2,3,4" \
					-set_of_points "0,1,2,3,8" \
					-set_of_points "0,1,2,5,6,7,8" \
					-set_of_points "0,1,2,3,5,6,7" \
					-set_of_points "0,1,2,3,5,6,9" \
					-set_of_points "0,1,2,3,5,6,10" \
					-set_of_points "0,1,2,3,5,6,4" \
					-set_of_points "0,1,2,3,8,11,13" \
					-set_of_points "3,6,9,7,10,12,8,11,13,14,4" \
					-set_of_points "3,5,6,9,7,10,12,11,13,14,4" \
					-set_of_points "0,1,2,3,5,6,9,7,10,12,4" \
				-end \
				-classification_prefix Dickson_set \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling Dickson_sets_can_lab.csv \
				-save_cumulative_data Dickson_sets_data.csv \
				-report \
			-end \
		-end 
	pdflatex Dickson_set_classification.tex
	open Dickson_set_classification.pdf



Endrass_7c: Endrass_F7.txt
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_points Endrass_F7.txt -end \
		-classification_prefix Endrass_F7 \
		-report \
		-end \
		-end
	pdflatex Endrass_F7_classification.tex
	open Endrass_F7_classification.pdf


# group order is 32


Hirschfeld16c: Hirschfeld_surface_q16.txt
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_points \
			Hirschfeld_surface_q16.txt -end \
		-classification_prefix Hirschfeld_surface_q16 \
		-report \
		-end \
		-end
	pdflatex Hirschfeld_surface_q16_classification.tex
	open Hirschfeld_surface_q16_classification.pdf

# nauty_interface_matrix_int n >= MAXN
# nauty_interface_matrix_int n = 74532
# nauty_interface_matrix_int MAXN = 22000



cuspidal_cubic_q16_subsets_canonical: cuspidal_cubic_16_subsets_6.txt
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_points cuspidal_cubic_16_subsets_6.txt \
		-end \
		-classification_prefix cuspidal_cubic_q16 \
		-report \
		-end \
		-end
	pdflatex cuspidal_cubic_q16_classification.tex
	open cuspidal_cubic_q16_classification.pdf




del_Pezzo_13_group: del_Pezzo_F13a_points.txt
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
		-file_of_points del_Pezzo_F13a_points.txt \
		-file_of_points del_Pezzo_F13b_points.txt \
		-end \
		-classification_prefix del_Pezzo_F13 \
		-report \
		-end
	pdflatex del_Pezzo_F13_classification.tex
	open del_Pezzo_F13_classification.pdf

# 12 minutes on Macintosh
# both have the same group of order 8, 
# generated by -1s on the diagonals for the first three variables.

#nauty_interface::nauty_interface_matrix_int n >= MAXN
#nauty_interface::nauty_interface_matrix_int n = 33492
#nauty_interface::nauty_interface_matrix_int MAXN = 22000




#############################################################################################################
# Section 6.7: Arcs and Caps in Projective Spaces


SECTION_ARCS_AND_CAPS_IN_PROJECTIVE_SPACES:




PGL_3_27:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-draw_options -end \
		-define G \
		-linear_group -PGL 3 27 -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGL_3_27_report.tex
	open PGL_3_27_report.pdf
	


AGGL_2_27:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-draw_options -end \
		-define G \
		-linear_group -AGGL 2 27 -end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex AGGL_2_27_report.tex
	open AGGL_2_27_report.pdf
	

AGL_4_3:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-draw_options -end \
		-define G \
		-linear_group -PGL 5 3 \
			-subgroup_by_generators "AGL_4_3" "1965150720" 14 \
			"1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0," \
			"1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1," \
			"1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1," \
			-end \
		-with G -do \
		-group_theoretic_activities \
			-report \
		-end
	pdflatex PGL_5_3_Subgroup_AGL_4_3_1965150720_report.tex
	open PGL_5_3_Subgroup_AGL_4_3_1965150720_report.pdf



#strong_generators::print_generators_gap(ostream &ost)

hyperoval_4:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label hyperoval_q4 -W -depth 6 \
					-report -end \
				-end \
				-target_size 6 \
				-d 2 \
			-end \
		-end
	pdflatex hyperoval_q4_poset.tex
	open hyperoval_q4_poset.pdf




hyperoval_8:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label hyperoval_q8 -W -depth 10 \
					-report -end \
				-end \
				-target_size 10 \
				-d 2 \
			-end \
		-end
	pdflatex hyperoval_q8_poset.tex
	open hyperoval_q8_poset.pdf




frame_stabilizer_PGGL:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G \
		-linear_group -PGGL 3 8 -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control \
				-problem_label frame_q8 -W -depth 4 \
				-draw_options -end \
				-report -end \
				-end \
			-classify_arcs \
				-target_size 4 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end
	pdflatex frame_q8_poset.tex
	open frame_q8_poset.pdf

frame_stabilizer_PGL:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 3 8 -end \
		-with G -do \
		-group_theoretic_activities \
				-poset_classification_control \
				-problem_label frame_q8 -W -depth 4 \
				-draw_options -end \
				-report -end \
				-end \
			-classify_arcs \
				-target_size 4 \
				-q 8 \
				-n 3 \
				-d 2 \
			-end \
		-end
	pdflatex frame_q8_poset.tex
	open frame_q8_poset.pdf



hyperoval_16:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label hyperoval_q16 -W -depth 18 \
					-report -end \
				-end \
				-target_size 18 \
				-d 2 \
			-end \
		-end
	pdflatex hyperoval_q16_poset.tex
	open hyperoval_q16_poset.pdf


HYPEROVAL_16_144="0, 1, 2, 3, 52, 67, 89, 106, 126, 141, 159, 176, 184, 199, 220, 235, 245, 262"
HYPEROVAL_16_16320="0, 1, 2, 3, 52, 70, 83, 109, 127, 139, 156, 174, 186, 199, 217, 229, 256, 264"


hyperoval_16_1_conic_type:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object H_16_1 \
				-set $(HYPEROVAL_16_144) \
			-end \
		-end \
		-with H_16_1 -do \
		-combinatorial_object_activity \
			-save \
		-end \
		-with H_16_1 -do \
		-combinatorial_object_activity \
			-conic_type 6 \
		-end \
		-print_symbols

hyperoval_16_1_nonconical_type:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object H_16_1 \
				-set $(HYPEROVAL_16_144) \
			-end \
		-end \
		-with H_16_1 -do \
		-combinatorial_object_activity \
			-save \
		-end \
		-with H_16_1 -do \
		-combinatorial_object_activity \
			-non_conical_type \
		-end \
		-print_symbols


#We found 17028 non-conical 6 subsets
#Eckardt point number distribution : $13^{252},\,$ $9^{720},\,$ $5^{2304},\,$ $3^{13752}$ 


hyperoval_16_2_nonconical_type:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object H_16_2 \
				-set $(HYPEROVAL_16_16320) \
			-end \
		-end \
		-with H_16_2 -do \
		-combinatorial_object_activity \
			-save \
		-end \
		-with H_16_2 -do \
		-combinatorial_object_activity \
			-non_conical_type \
		-end \
		-print_symbols

#We found 6188 = {17 choose 5} non-conical 6 subsets
#Eckardt point number distribution : $45^{68},\,$ $13^{2040},\,$ $5^{4080}$ 

#neighbors_of_0_with_4_removed.csv
#Row,C0,C1,C2,C3
#0,2,3,9,10
#1,1,3,7,8
#2,10,12,13,15
#3,1,5,10,11
#4,3,5,6,13
#5,8,9,11,12
#6,7,11,13,17
#7,7,10,14,16
#8,1,9,13,16
#9,2,8,13,14
#10,1,2,15,17
#11,6,8,10,17
#12,6,7,9,15
#13,2,6,11,16
#14,5,9,14,17
#15,5,8,15,16
#16,1,6,12,14
#17,2,5,7,12
#18,3,12,16,17
#19,3,11,14,15
#END



hyperoval_16_stab_0_disjoint_sets_graph:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -disjoint_sets_graph neighbors_of_0_with_4_removed.csv -end \
		-with G -do \
			-graph_theoretic_activity -find_cliques -target_size 4 -end -end \
		-print_symbols


# 5 cliques of size 4
#ROW,C0,C1,C2,C3
#0,0,6,15,16
#1,1,2,13,14
#2,3,9,12,18
#3,4,5,7,10
#4,8,11,17,19
#END

#clique 0:
#0,2,3,9,10
#6,7,11,13,17
#15,5,8,15,16
#16,1,6,12,14
# partition: (1,6,12,14|2,3,9,10|5,8,15,16|7,11,13,17)
# 4 is missing, it is the nucleus
# 0 is missing is the chosen point


hyperoval_16c:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
			-set_of_points $(HYPEROVAL_16_16320) \
		-end \
		-classification_prefix hyperoval_q16 \
		-report \
		-end \
		-end
	pdflatex hyperoval_q16_classification.tex
	open hyperoval_q16_classification.pdf

#			-set_of_points $(HYPEROVAL_16_144) \


# June 3 2020: 1 sec on Mac
 



# nonconical 6-arcs are used for classifying cubic surfaces:





nc_arcs_16:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label nc_arcs_q16_d2 -W -depth 6 \
					-report -end \
				-end \
				-target_size 6 \
				-d 2 \
				-conic_test \
			-end \
		-end
	pdflatex nc_arcs_q16_d2_poset.tex
	open nc_arcs_q16_d2_poset.pdf


#User time: 0:00



nc_arcs_32_E13:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 32 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label nc_arcs_q32_d2 -W -depth 6 \
					-report  -end \
				-end \
				-target_size 6 \
				-test_nb_Eckardt_points 13 \
				-d 2 \
				-conic_test \
			-end \
		-end
	pdflatex nc_arcs_q32_d2_poset.tex
	open nc_arcs_q32_d2_poset.pdf

#User time: 0:00



nc_arcs_32_canon: 
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 32 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input \
					-set_of_points "0,1,2,3,100,131" \
					-set_of_points "0,1,2,3,100,628" \
					-set_of_points "0,1,2,3,100,989" \
				-end \
				-classification_prefix nc_arcs_32 \
				-report \
			-end \
		-end
	pdflatex nc_arcs_32_classification.tex
	open nc_arcs_32_classification.pdf



nc_arcs_64_E13:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 64 -end \
		-define P -projective_space 2 F -use_projectivity_subgroup -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label nc_arcs_q64_d2 -W -depth 6 \
					-recognize "0,1,2,3,326,389" \
					-recognize "0,1,2,3,1106,1169" \
					-recognize "0,1,2,3,2599,2536" \
					-recognize "0,1,2,3,2796,2859" \
					-recognize "0,1,2,3,1756,1819" \
					-recognize "0,1,2,3,2469,2406" \
					-recognize "0,1,2,3,3056,3119" \
					-recognize "0,1,2,3,2339,2276" \
					-recognize "0,1,2,3,3836,3899" \
					-recognize "0,1,2,3,3966,4029" \
					-recognize "0,1,2,3,2666,2729" \
					-recognize "0,1,2,3,2016,2079" \
					-report  -end \
				-end \
				-target_size 6 \
				-test_nb_Eckardt_points 13 \
				-d 2 \
				-conic_test \
			-end \
		-end
	#pdflatex nc_arcs_q64_d2_poset.tex
	#open nc_arcs_q64_d2_poset.pdf


F64_work:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 64 -end \
		-define f -formula "f" "f" "" "a*a+a"    \
		-with F -do -finite_field_activity \
			-evaluate f "a=2" -end

F64_frob:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 64 -end \
		-define f -formula "f" "f" "" "a*a*a*a*a*a*a*a"    \
		-with F -do -finite_field_activity \
			-evaluate f "a=61" -end


# surfaces with 13 Eckardt points have OCN=0,98,99

surface_64_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 64 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S -q 64 -catalogue 0 \
			-end \
		-end \
		-with S -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q64_iso0_with_group.tex
	open surface_catalogue_q64_iso0_with_group.pdf




#makes it slow:
#				-test_nb_Eckardt_points 13 \
#					-report -select_orbits_by_level 6 -select_orbits_by_stabilizer_order_multiple_of 24 -end \
#User time: 0:3



nc_arcs_128:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 2 F -use_projectivity_subgroup -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label nc_arcs_q128_d2 -W -depth 6 \
					-report -select_orbits_by_level 6 -select_orbits_by_stabilizer_order_multiple_of 24 -end \
				-end \
				-target_size 6 \
				-d 2 \
				-conic_test \
			-end \
		-end
	pdflatex nc_arcs_q128_d2_poset.tex
	open nc_arcs_q128_d2_poset.pdf


#User time: 0:52




nc_arcs_256_E13:
	$(ORBITER_PATH)orbiter.out -v 8 \
		-define F -finite_field -q 256 -end \
		-define P -projective_space 2 F -use_projectivity_subgroup -end \
		-with P -do \
		-projective_space_activity \
			-classify_arcs \
				-poset_classification_control \
					-problem_label nc_arcs_q256_d2 -W -depth 6 \
					-report  -end \
				-end \
				-target_size 6 \
				-test_nb_Eckardt_points 13 \
				-d 2 \
				-conic_test \
			-end \
		-end
	pdflatex nc_arcs_q256_d2_poset.tex
	open nc_arcs_q256_d2_poset.pdf




##



Example_F64:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 64 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
			-projective_space_activity \
			-define_surface S64_abcd_52_8_8_52 -q 64 \
				-family_general_abcd 52 8 8 52 -end \
		-end \
		-with S64_abcd_52_8_8_52 -do \
			-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_family_general_abcd_q64_a52_b8_c8_d52_report.tex




six_arcs_4_nbE13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -problem_label sixarcs_q4 -end \
		-six_arcs_not_on_conic -filter_by_nb_Eckardt_points 13 -end


six_arcs_8_nbE13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -problem_label sixarcs_q8 -end \
		-six_arcs_not_on_conic -filter_by_nb_Eckardt_points 13 -end

six_arcs_16_nbE13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -problem_label sixarcs_q16 -end \
		-six_arcs_not_on_conic -filter_by_nb_Eckardt_points 13 -end

six_arcs_32_nbE13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 32 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -problem_label sixarcs_q32 -end \
		-six_arcs_not_on_conic -filter_by_nb_Eckardt_points 13 -end

six_arcs_64_nbE13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 64 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -problem_label sixarcs_q64 -end \
		-six_arcs_not_on_conic -filter_by_nb_Eckardt_points 13 -end


#User time: 0:7
# 9 arcs: ago: 4, 8, 24^5, 48^2



six_arcs_128_nbE13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -problem_label sixarcs_q128 -end \
		-six_arcs_not_on_conic -filter_by_nb_Eckardt_points 13 -end

# 1 min 39 sec
# 12 arcs, ago: 4^3, 24^9

six_arcs_256_nbE13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 256 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -problem_label sixarcs_q256 -end \
		-six_arcs_not_on_conic -filter_by_nb_Eckardt_points 13 -end

# 27 minutes on ripoff
#User time: 29:11 on ripoff 7/30/21









#############################################################################################################
# Section 6.8: Cubic Curves

SECTION_CUBIC_CURVES:


cubic_curves_PG_2_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-define F -finite_field -q 3 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_cubic_curves -q 4 -target_size 9 -n 3 -d 3 \
				-poset_classification_control -problem_label cc_4 -W -depth 9 \
					-draw_options -radius 200 -embedded -end \
					-draw_poset \
					-report -end \
				-end \
			-end
	pdflatex cc_4_poset.tex
	open cc_4_poset.pdf
	pdflatex cc_4_poset_lvl_9.tex
	#open cc_4_poset_lvl_9.pdf
	pdflatex Cubic_curves_q4.tex
	#open Cubic_curves_q4.pdf

cubic_curves_PG_2_4_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph cc_4_poset_lvl_9.layered_graph \
		-radius 300 -embedded -line_width 1.1  -y_stretch 0.9 -scale 0.25 \
		-paths_in_between 6 4 9 0 \
		-end
	pdflatex cc_4_poset_lvl_9_draw.tex
	open cc_4_poset_lvl_9_draw.pdf


cubic_curves_PG_2_8:
	$(ORBITER_PATH)orbiter.out -v 3 -define G \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-classify_cubic_curves -q 8 -target_size 9 -n 3 -d 3 \
			-poset_classification_control -problem_label cc_8 -W -depth 9 \
				-draw_options -radius 200 -embedded -end \
				-recognize "0,1,2,3,35,28" \
				-recognize "1,2,3,51,28,61,46,71,40" \
				-draw_poset \
				-Kramer_Mesner_matrix 6 9 \
			-end \
		-end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file cc_8_KM_6_9.csv \
		-box_width 50 -bit_depth 8   -end
	pdflatex Cubic_curves_q8.tex
	open Cubic_curves_q8.pdf
	#pdflatex cc_8_tree_lvl_9.tex
	#open cc_8_tree_lvl_9.pdf
	
# the 6-set is orbit 7
# the 9-set is orbit 1


cubic_curves_PG_2_8_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph cc_8_poset_lvl_9.layered_graph \
		-radius 2 -embedded -line_width 0.01  -y_stretch 1.3 -scale 0.5 \
		-paths_in_between 6 7 9 1 \
		-end
	pdflatex cc_8_poset_lvl_9_draw.tex
	open cc_8_poset_lvl_9_draw.pdf

#cc_8_poset_lvl_9.layered_graph
#cc_8_poset_detailed_lvl_9.layered_graph
	
cubic_curves_PG_2_8_canon:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -set_of_points "2,3,28,46,51,61,40,71" -end \
		-classification_prefix cc_8 \
		-report \
		-end \
		-end
	pdflatex cc_8_classification.tex
	open cc_8_classification.pdf

	







#############################################################################################################
# Chapter 7 - Cubic Surfaces
#############################################################################################################



#############################################################################################################
# Section 7.1: Cubic Surfaces Creation


SECTION_CUBIC_SURFACES_CREATION:



surface_4_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S4_0 -q 4 -catalogue 0 -end \
		-end \
		-with S4_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
			-all_quartic_curves \
		-end
	pdflatex surface_catalogue_q4_iso0_report.tex
	open surface_catalogue_q4_iso0_report.pdf
	pdflatex surface_catalogue_q4_iso0_with_group.tex
	open surface_catalogue_q4_iso0_with_group.pdf



surface_7_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S7_0 -q 7 -catalogue 0 -end \
		-end \
		-with S7_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
			-all_quartic_curves \
		-end
	pdflatex surface_catalogue_q7_iso0_report.tex
	open surface_catalogue_q7_iso0_report.pdf
	pdflatex surface_catalogue_q7_iso0_with_group.tex
	open surface_catalogue_q7_iso0_with_group.pdf




Family_general_F7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S7_abcd_2_3_3_4 -q 7 -family_general_abcd 2 3 3 4 -end \
		-end \
		-with S7_abcd_2_3_3_4 -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_family_general_abcd_q7_a2_b3_c3_d4_report.tex
	open surface_family_general_abcd_q7_a2_b3_c3_d4_report.pdf
		
# Fermat with 18 Eckardt points
# no automorphism group, so no -report_with_group and no -all_quartic_curves







surface_8_0_catalogue:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S8_0 -q 8 -catalogue 0 -end \
		-end \
		-with S8_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q8_iso0_report.tex
	open surface_catalogue_q8_iso0_report.pdf
	pdflatex surface_catalogue_q8_iso0_with_group.tex
	open surface_catalogue_q8_iso0_with_group.pdf



surface_8_0_clean:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S8_0 -q 8 -catalogue 0 \
			-select_double_six "15,11,22,19,24,5,16,10,23,20,25,4" \
			-select_double_six "3,2,1,0,5,4,9,8,7,6,11,10" \
			-transform_inverse "1,4,4,0,6,0,0,0,6,2,0,1,7,0,4,0,0" \
			-transform "4,4,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,1, 0" \
			-transform_inverse "2,0,0,0,0,2,0,0,0,0,2,0,1,1,2,3,0" \
		-end -end  \
		-with S8_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q8_iso0_report.tex
	open surface_catalogue_q8_iso0_report.pdf



# clean equation for Tekirdag-1:

surface_8_0b:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S8_0 -q 8 -catalogue 0 \
			-select_double_six "15,11,22,19,24,5,16,10,23,20,25,4" \
			-select_double_six "3,2,1,0,5,4,9,8,7,6,11,10" \
			-transform "1,0,0,0,0,1,0,6,0,0,1,6,0,0,0,1,0" \
			-transform_inverse "3,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0" \
			-transform_inverse "2,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0" \
			-end \
		-end \
		-with S8_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q8_iso0_with_group.tex
	open surface_catalogue_q8_iso0_with_group.pdf







Eckardt_13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S_q13 -q 13 -family_Eckardt 3 1 -end \
		-end \
		-with S_q13 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_family_Eckardt_q13_a3_b1_with_group.tex
	open surface_family_Eckardt_q13_a3_b1_with_group.pdf


# writes tangents.txt


# 13_0 has 4 Eckardt points
# 13_1 has 6 Eckardt points
# 13_2 has 9 Eckardt points
# 13_3 has 18 Eckardt points
#

surface_13_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-defi+ne F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S13_0 -q 13 -catalogue 0 -end \
		-end \
		-with S13_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q13_iso0_report.tex
	open surface_catalogue_q13_iso0_report.pdf


# clean equation for Tekirdag-2:


surface_16_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S16_0 -q 16 -catalogue 0 \
			-transform "1,0,0,0,0,1,0,12,0,0,1,12,0,0,0,1,0" \
			-transform "15,11,4,0,0,0,12,0,0,12,0,0,0,0,0,1,3" \
			-end \
		-end \
		-with S16_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q16_iso0_with_group.tex
	open surface_catalogue_q16_iso0_with_group.pdf


#		-transform_inverse "3,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0" \
#		-transform_inverse "13,12,1,0,12,13,1,0,0,0,1,0,0,0,0,1,0" \
#		-transform_inverse "1,0,0,0,0,1,0,0,12,12,1,0,0,0,0,1,0" \
#		-transform_inverse "12,0,0,0,0,12,0,0,0,0,1,0,0,0,0,1,0" 

# rank of lines ( 66591, 26737, 4093, 69904, 28376, 26470, 70160, 69855, 26208, 5847, 369, 32230, 529, 30293, 70068, 2178, 261, 28666, 8575, 105, 31694, 0, 51784, 25209, 22193, 49862, 274 )
# Rank of points on Klein quadric: ( 29181, 4677, 29950, 33, 62496, 429, 1, 9205, 37, 29964, 29364, 21501, 4656, 54735, 5425, 30105, 754, 6680, 13354, 758, 30106, 0, 29209, 48736, 25595, 33780, 4657 )


# ai: 29181, 4677, 29950, 33, 62496, 429
# bi: 1, 9205, 37, 29964, 29364, 21501

# Tekirdag-1:

G13_8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface T1 -family_G13 2 -q 8 -end \
		-end \
		-with T1 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_family_G13_q8_a2_with_group.tex
	open surface_family_G13_q8_a2_with_group.pdf


F13_8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface T1 -family_F13 2 -q 8 -end \
		-end \
		-with T1 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_family_F13_q8_a2_with_group.tex
	open surface_family_F13_q8_a2_with_group.pdf




# Tekirdag-2:

F13_16:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface T2 -family_F13 2 -q 16 -end \
		-end \
		-with T2 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_family_F13_q16_a2_with_group.tex
	open surface_family_F13_q16_a2_with_group.pdf


# Tekirdag-3:

F13_32:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 32 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface T3 -family_F13 2 -q 32 -end \
		-end \
		-with T3 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_family_F13_q32_a2_with_group.tex
	open surface_family_F13_q32_a2_with_group.pdf


# Kapadokya-1:

F13_64a:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 64 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface K1 -family_F13 2 -q 64 -end \
		-end \
		-with K1 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end


# Kapadokya-2:

F13_64b:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 64 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface K2 -family_F13 18 -q 64 -end \
		-end \
		-with K2 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end


Colorado1:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface CO-1 -q 128 -catalogue 0 \
			-transform_inverse "1,0,0,0,0,1,0,96,0,0,1,96,0,0,0,1,0" \
			-end \
		-end \
		-with CO-1 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end


# recognize the arcs from Colorado-1,2,3:



Colorado2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface CO-2 -q 128 -catalogue 926 \
			-transform_inverse "1,0,0,0,0,1,0,32,0,0,1,32,0,0,0,1,0" \
			-end \
		-end \
		-with CO-2 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end

Colorado3:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface CO-3 -q 128 -catalogue 928 \
			-transform_inverse "1,0,0,0,0,1,0,59,0,0,1,59,0,0,0,1,0" \
			-end \
		-end \
		-with CO-3 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end


# Colorado-1:

F13_128a:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface CO-1 -family_F13 2 -q 128 -end \
		-end \
		-with CO-1 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end

# Colorado-2:

F13_128b:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface CO-2 -family_F13 6 -q 128 -end \
		-end \
		-with CO-2 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end

# Colorado-3:

F13_128c:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 128 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface CO-3 -family_F13 14 -q 128 -end \
		-end \
		-with CO-3 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end


move_two_lines:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 8 -end \
		-with F -do -finite_field_activity \
		-move_two_lines_in_hyperplane_stabilizer 65 4680 72 657 -end




F_alpha_beta_gamma_delta:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-with F -do -finite_field_activity \
		-parse_and_evaluate \
		"F_alpha_beta_gamma_delta" "x0,x1,x2,x3" $(F_ALPHA_BETA_GAMMA_DELTA) \
		"alpha=2,beta=3,gamma=4,delta=5" -end
	dot -Tpng F_alpha_beta_gamma_delta.gv >F_alpha_beta_gamma_delta.png








F_abcd:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-with F -do -finite_field_activity \
			-parse_and_evaluate "Fabcd" "X0,X1,X2,X3" $(F_abcd_eqn) "a=4,b=2,c=2,d=4" -end
	#dot -Tpng F_alpha_beta_gamma_delta.gv >F_alpha_beta_gamma_delta.png



F_abcd_sweep_4_27_q7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-sweep_4_27 sweep_4_27_q7 -q 7 -by_equation "F_abcd" \
			"\DF_{a,b,c,d}\D" "X0,X1,X2,X3" \
			$(F_abcd_eqn) \
			"a=2,b=3,c=4,d=5" \
			"\Da=2,b=3,c=4,d=5\D" \
		-end \
		-end \







F_alpha_beta_gamma_delta_q7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-control_six_arcs -end \
		-define_surface -q 7 -by_equation "F_alpha_beta_gamma_delta" \
			"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
			$(F_ALPHA_BETA_GAMMA_DELTA) \
			"alpha=2,beta=3,gamma=4,delta=5" \
			"\D\alpha=2,\beta=3,\gamma=4,\delta=5\D" \
		-end \
		-end 
	pdflatex surface_equation_F_alpha_beta_gamma_delta_q7_report.tex
	open surface_equation_F_alpha_beta_gamma_delta_q7_report.pdf




surface_F7_15_lines_stab:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
			-set_of_points "$(SURFACE_F7_15LINES_POINTS)" \
		-end \
		-classification_prefix surface_15_lines_q7 \
		-report \
		-end \
		-end
	pdflatex surface_15_lines_q7_classification.tex
	open surface_15_lines_q7_classification.pdf


# group has order 6, 2 generators:
#1,5,0,0,3,6,0,0,1,1,3,0,5,5,0,3, 1,0,2,5,0,1,6,1,0,0,3,5,0,0,4,4,


F_alpha_beta_gamma_delta_q7_override_group:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface F_2345 -q 7 -by_equation "F_alpha_beta_gamma_delta" \
			"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
			$(F_ALPHA_BETA_GAMMA_DELTA) \
			"alpha=2,beta=3,gamma=4,delta=5" \
			"\D\alpha=2,\beta=3,\gamma=4,\delta=5\D" \
			-override_group 6 2 "1,5,0,0,3,6,0,0,1,1,3,0,5,5,0,3, 1,0,2,5,0,1,6,1,0,0,3,5,0,0,4,4" \
		-end \
		-end \
		-with F_2345 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_equation_F_alpha_beta_gamma_delta_q7_report.tex
	open surface_equation_F_alpha_beta_gamma_delta_q7_report.pdf
	pdflatex surface_equation_F_alpha_beta_gamma_delta_q7_with_group.tex
	open surface_equation_F_alpha_beta_gamma_delta_q7_with_group.pdf


F_alpha_beta_gamma_delta_sweep_4_q3:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 3 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-sweep_4 sweep_4_q3 -q 3 -by_equation "F_alpha_beta_gamma_delta" \
			"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
			$(F_ALPHA_BETA_GAMMA_DELTA) \
			"alpha=2,beta=3,gamma=4,delta=5" \
			"\D\alpha=2,\beta=3,\gamma=4,\delta=5\D" \
		-end \
		-end 



F_alpha_beta_gamma_delta_sweep_4_q7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-sweep_4 sweep_4_q7 -q 7 -by_equation "F_alpha_beta_gamma_delta" \
			"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
			$(F_ALPHA_BETA_GAMMA_DELTA) \
			"alpha=2,beta=3,gamma=4,delta=5" \
			"\D\alpha=2,\beta=3,\gamma=4,\delta=5\D" \
		-end \
		-end 


# creates F_alpha_beta_gamma_delta_q7_sweep4_15_data.csv


F_alpha_beta_gamma_delta_sweep_4_q7_split:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-csv_file_split_rows_modulo  F_alpha_beta_gamma_delta_q7_sweep4_15_data.csv 10

#F_alpha_beta_gamma_delta_q7_sweep4_15_data_split_0_mod_10.csv


F_alpha_beta_gamma_delta_sweep_4_q7_classify:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_quartic_curves_with_substructure \
				F_alpha_beta_gamma_delta_q7_sweep4_15_data.csv \
				1 4 3 F_alpha_beta_gamma_delta_sweep_4_q7_classify \
		-end \
		-print_symbols


F_alpha_beta_gamma_delta_sweep_4_classify_q7_nauty:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
			-file_of_points F_alpha_beta_gamma_delta_q7_points.txt \
		-end \
		-classification_prefix surface_15_lines_q7 \
		-report \
		-end \
		-end
	pdflatex surface_15_lines_q7_classification.tex
	open surface_15_lines_q7_classification.pdf


F_alpha_beta_gamma_delta_q7_recognize:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 49 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf -W -end \
		-end \
		-with Surf -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 49 \
				-by_equation "F_alpha_beta_gamma_delta" \
					"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
					$(F_ALPHA_BETA_GAMMA_DELTA) \
					"alpha=2,beta=1,gamma=1,delta=2" \
					"\D\alpha=2,\beta=1,\gamma=1,\delta=2\D" \
				-end \
			-end \
		-end \
		-with Surf -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 49 \
				-by_equation "F_alpha_beta_gamma_delta" \
					"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
					$(F_ALPHA_BETA_GAMMA_DELTA) \
					"alpha=2,beta=1,gamma=2,delta=2" \
					"\D\alpha=2,\beta=1,\gamma=2,\delta=2\D" \
				-end \
			-end \
		-end \
		-with Surf -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 49 \
				-by_equation "F_alpha_beta_gamma_delta" \
					"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
					$(F_ALPHA_BETA_GAMMA_DELTA) \
					"alpha=2,beta=1,gamma=5,delta=2" \
					"\D\alpha=2,\beta=1,\gamma=5,\delta=2\D" \
				-end \
			-end \
		-end \
		-with Surf -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 49 \
				-by_equation "F_alpha_beta_gamma_delta" \
					"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
					$(F_ALPHA_BETA_GAMMA_DELTA) \
					"alpha=2,beta=1,gamma=1,delta=3" \
					"\D\alpha=2,\beta=1,\gamma=1,\delta=3\D" \
				-end \
			-end \
		-end \
		-with Surf -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 49 \
				-by_equation "F_alpha_beta_gamma_delta" \
					"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
					$(F_ALPHA_BETA_GAMMA_DELTA) \
					"alpha=2,beta=1,gamma=2,delta=3" \
					"\D\alpha=2,\beta=1,\gamma=2,\delta=3\D" \
				-end \
			-end \
		-end \
		-with Surf -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 49 \
				-by_equation "F_alpha_beta_gamma_delta" \
					"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
					$(F_ALPHA_BETA_GAMMA_DELTA) \
					"alpha=2,beta=1,gamma=4,delta=3" \
					"\D\alpha=2,\beta=1,\gamma=4,\delta=3\D" \
				-end \
			-end \
		-end \



surf49_recognize:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options  -end \
		-define F -finite_field -q 49 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf27 -W -end \
		-end \
		-with Surf27 -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 49 \
				-by_coefficients "2,5,1,16,4,10,1,18,4,8,4,9,3,11,4,14" \
				-end \
			-end \
		-end \
		-print_symbols




McKean_15lines_q7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S -by_coefficients $(SURFACE_MCKEAN_15_LINES) -q 7 \
		-end \
		-end 
	#pdflatex surface_by_coefficients_q7_report.tex
	#open surface_by_coefficients_q7_report.pdf

# 2 Eckardt points

F_4_4_3_3_q7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface -q 7 -by_equation "F_alpha_beta_gamma_delta" \
			"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
			$(F_ALPHA_BETA_GAMMA_DELTA) \
			"alpha=4,beta=4,gamma=3,delta=3" \
			"\D\alpha=4,\beta=4,\gamma=3,\delta=3\D" \
		-end \
		-end
	#pdflatex surface_equation_F_alpha_beta_gamma_delta_q7_report.tex
	#open surface_equation_F_alpha_beta_gamma_delta_q7_report.pdf

# has 4 Eckardt points



F_alpha_beta_gamma_delta_points.txt:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-sweep nothing \
		-define_surface -q 7 -by_equation "F_alpha_beta_gamma_delta" \
			"\DF_{\alpha,\beta,\gamma,\delta}\D" "x0,x1,x2,x3" \
			$(F_ALPHA_BETA_GAMMA_DELTA) \
			"alpha=2,beta=3,gamma=4,delta=5" \
			"\D\alpha=2,\beta=3,\gamma=4,\delta=5\D" \
		-end \
		-end 











#############################################################################################################
# Section 7.2: Cubic Surfaces and Quartic Curves 


SECTION_CUBIC_SURFACES_AND_QUARTIC_CURVES:




surface_4_0_quartic_curves:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S4_0 -q 4 -catalogue 0 -end \
		-end \
		-with S4_0 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
			-all_quartic_curves \
		-end
	pdflatex surface_catalogue_q4_iso0_report.tex
	open surface_catalogue_q4_iso0_report.pdf
	pdflatex surface_catalogue_q4_iso0_with_group.tex
	open surface_catalogue_q4_iso0_with_group.pdf
	#pdflatex surface_catalogue_q4_iso0_quartics.tex
	#open surface_catalogue_q4_iso0_quartics.pdf



# full del Pezzo surfaces:

NB_CUBIC_SURFACES_Q13=4

quartic_curves_q13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-loop L 0 $(NB_CUBIC_SURFACES_Q13) 1 \
			-with P -do \
			-projective_space_activity \
				-define_surface S13_%L -q 13 -catalogue %L -end \
			-end \
		-end_loop \
		-print_symbols \
		-loop L 0 $(NB_CUBIC_SURFACES_Q13) 1 \
			-with S13_%L -do \
			-cubic_surface_activity \
				-export_all_quartic_curves \
			-end \
		-end_loop \
		-print_symbols
		#pdflatex surface_catalogue_q13_iso0_quartics.tex
		#open surface_catalogue_q13_iso0_quartics.pdf
		#pdflatex surface_catalogue_q13_iso1_quartics.tex
		#open surface_catalogue_q13_iso1_quartics.pdf
		#pdflatex surface_catalogue_q13_iso2_quartics.tex
		#open surface_catalogue_q13_iso2_quartics.pdf
		#pdflatex surface_catalogue_q13_iso3_quartics.tex
		#open surface_catalogue_q13_iso3_quartics.pdf


#surface_catalogue_q13_iso0_quartics.csv

quartic_curves_q13_classify:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_quartic_curves_with_substructure \
				surface_catalogue_q13_iso%d_quartics.csv \
				3 4 4 quartic_curves_q13 \
		-end \
		-print_symbols \

#q13
#The number of types of quartic curves is 2
#idx : ago
#0 : 24
#1 : 48


# problem, wrong group:


quartic_curves_13_classify_nauty: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-csv_file_extract_column_to_txt \
		surface_catalogue_q13_iso0_quartics.csv "pts" 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-csv_file_extract_column_to_txt \
		surface_catalogue_q13_iso1_quartics.csv "pts" 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
			-file_of_points surface_catalogue_q13_iso0_quartics_pts.txt \
			-file_of_points surface_catalogue_q13_iso1_quartics_pts.txt \
		-end \
		-classification_prefix quartic_q13 \
		-report \
		-end \
		-end
	pdflatex quartic_q13_classification.tex
	open quartic_q13_classification.pdf


NB_CUBIC_SURFACES_Q17=7

quartic_curves_q17:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 17 -end \
		-define P -projective_space 3 F -end \
		-loop L 0 $(NB_CUBIC_SURFACES_Q17) 1 \
			-with P -do \
			-projective_space_activity \
				-define_surface S17_%L -q 17 -catalogue %L -end \
			-end \
		-end_loop \
		-print_symbols \
		-loop L 0 $(NB_CUBIC_SURFACES_Q17) 1 \
			-with S17_%L -do \
			-cubic_surface_activity \
				-report \
				-export_all_quartic_curves \
			-end \
		-end_loop \
		-print_symbols
		pdflatex surface_catalogue_q17_iso0_quartics.tex
		open surface_catalogue_q17_iso0_quartics.pdf



quartic_curves_q17_classify:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 17 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_quartic_curves_with_substructure \
				surface_catalogue_q17_iso%d_quartics.csv \
				$(NB_CUBIC_SURFACES_Q17) \
		-end \
		-print_symbols \

#q17
#The number of types of quartic curves is 7
#idx : ago
#0 : 24
#1 : 24
#2 : 4
#3 : 96
#4 : 6
#5 : 8
#6 : 2



NB_CUBIC_SURFACES_Q19=10

quartic_curves_q19:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 19 -end \
		-define P -projective_space 3 F -end \
		-loop L 0 $(NB_CUBIC_SURFACES_Q19) 1 \
			-with P -do \
			-projective_space_activity \
				-define_surface S19_%L -q 19 -catalogue %L -end \
			-end \
		-end_loop \
		-print_symbols \
		-loop L 0 $(NB_CUBIC_SURFACES_Q19) 1 \
			-with S19_%L -do \
			-cubic_surface_activity \
				-report \
				-export_all_quartic_curves \
			-end \
		-end_loop \
		-print_symbols
		pdflatex surface_catalogue_q19_iso0_report.tex
		open surface_catalogue_q19_iso0_report.pdf


quartic_curves_q19_classify:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 19 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_quartic_curves_with_substructure \
				surface_catalogue_q19_iso%d_quartics.csv \
				$(NB_CUBIC_SURFACES_Q19) 4 4 quartic_curves_q19 \
		-end \
		-print_symbols

# writes:
# quartic_curves_q19_canonical_data.csv
# quartic_curves_q19_canonical.tex

# 14 isomorphism types:
# ago dist: 4^1, 9^1, 2^4, 6^2, 8^3, 24^3

quartic_curves_q19_set_stabilizer:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-list_arguments \
		-draw_options -end \
		-define F -finite_field -q 19 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-set_stabilizer 4 \
				surface_catalogue_q19_iso%d_quartics.csv \
				$(NB_CUBIC_SURFACES_Q19) "pts_on_curve" \
		-end \
		-print_symbols \




# needs work!!!

quartic_curves_17_classify_old: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 17 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
			-file_of_points surface_catalogue_q17_iso0_quartics.txt \
			-file_of_points surface_catalogue_q17_iso1_quartics.txt \
			-file_of_points surface_catalogue_q17_iso2_quartics.txt \
			-file_of_points surface_catalogue_q17_iso3_quartics.txt \
			-file_of_points surface_catalogue_q17_iso4_quartics.txt \
			-file_of_points surface_catalogue_q17_iso5_quartics.txt \
			-file_of_points surface_catalogue_q17_iso6_quartics.txt \
		-end \
		-classification_prefix quartic_q17 \
		-report \
		-end \
		-end
	pdflatex quartic_q17_classification.tex
	open quartic_q17_classification.pdf




quartic_curve_13_0_canonical: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -set_of_points "34,74,87,89,95,99,146,175" -end \
		-classification_prefix quartic_q13_0 \
		-report \
		-end \
		-end
	pdflatex quartic_q13_0_classification.tex
	open quartic_q13_0_classification.pdf

# group order 24

surface_13_0_quartics:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S13_0 -q 13 -catalogue 0 -end \
		-end \
		-with S13_0 -do \
		-cubic_surface_activity \
			-report \
			-export_all_quartic_curves \
		-end
	pdflatex surface_catalogue_q13_iso0_quartics.tex
	open surface_catalogue_q13_iso0_quartics.pdf


surface_13_1_quartics:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S13_1 -q 13 -catalogue 1 -end \
		-end \
		-with S13_1 -do \
		-cubic_surface_activity \
			-report \
			-export_all_quartic_curves \
		-end
	pdflatex surface_catalogue_q13_iso1_quartics.tex
	open surface_catalogue_q13_iso1_quartics.pdf






quartic_curve_13_2_group:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 3 13 \
			-subgroup_by_generators "quartic_13_2"  24336 5 \
			"1,0,0,0,1,0,1,2,12," \
			"1,0,0,1,7,0,11,0,7," \
			"1,0,0,8,12,1,4,11,1," \
			"1,0,0,11,4,1,2,10,0," \
			"0,1,0,11,3,0,4,9,1" \
			-restricted_action "29,86,97,154" \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-end
	pdflatex PGL_3_13_Subgroup_quartic_13_2_24336_report.tex
	open PGL_3_13_Subgroup_quartic_13_2_24336_report.pdf
	#pdflatex PGGL_3_4_report.tex
	#open PGGL_3_4_report.pdf


KNECHT_13_1="1,0,1,1,1,2,12,9"
KNECHT_13_2="1,0,1,1,1,2,8,9,8,10,8,11"

Knecht_13_1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object Knecht_13_1 \
				-projective_variety "Knecht_1" 4 $(KNECHT_13_1) \
				-monomial_type_PART \
			-end \
		-end \
		-with Knecht_13_1 -do \
		-combinatorial_object_activity \
			-save \
		-end \
		-with Knecht_13_1 -do \
		-combinatorial_object_activity \
			-conic_type 6 \
		-end \
		-print_symbols


quartic_curve_Knecht_1_canonical: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_points Knecht_1.txt -end \
		-classification_prefix quartic_Knecht_1 \
		-report \
		-end \
		-end
	pdflatex quartic_Knecht_1_classification.tex
	open quartic_Knecht_1_classification.pdf



Knecht_13_2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object Knecht_13_2 \
				-projective_variety "Knecht_2" 4 $(KNECHT_13_2) \
				-monomial_type_PART \
			-end \
		-end \
		-with Knecht_13_2 -do \
		-combinatorial_object_activity \
			-save \
		-end


quartic_curve_Knecht_2_canonical: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
		-file_of_points Knecht_2.txt -end \
		-classification_prefix quartic_Knecht_2 \
		-report \
		-end \
		-end
	pdflatex quartic_Knecht_2_classification.tex
	open quartic_Knecht_2_classification.pdf

# group of order 24

quartic_curve_13_canonical: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
		-file_of_points Knecht_1.txt \
		-file_of_points Knecht_2.txt \
		-set_of_points "34,74,87,89,95,99,146,175" \
		-set_of_points "29,86,97,154"  \
		-set_of_points "17,18,47,72,113,134,162,163"  \
		-end \
		-classification_prefix quartic_Knecht_2 \
		-report \
		-end \
		-end
	pdflatex quartic_Knecht_2_classification.tex
	open quartic_Knecht_2_classification.pdf



Nathan_19_1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 19 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
			-define_object N19_1 \
				-projective_variety "Nathan19_1" 4 \
				"16,0,9,3,10,4,11,9,14,12,9,5,4,14,3,7,16,1,4,6,17,11,17,8,6,2" \
				-monomial_type_PART \
			-end \
		-end \
		-with N19_1 -do -combinatorial_object_activity -save \
		-end



surface_25_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-define_surface S25_0 -q 25 -catalogue 0 -end \
		-end \
		-with S25_0 -do \
		-cubic_surface_activity \
			-report \
			-export_all_quartic_curves \
		-end
	pdflatex surface_catalogue_q25_iso0_quartics.tex
	open surface_catalogue_q25_iso0_quartics.pdf




quartic_curve_25_0_0_canonical: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
		-set_of_points "10,11,59,63,124,135,136,170,206,257,275,284,285,367,378,393,433,619,641,644"  \
		-set_of_points "9, 24, 62, 67, 77, 84, 87, 89, 125, 130, 158, 172, 197, 219, 266, 271, 325, 356, 391, 392, 400, 429, 454, 458, 470, 503, 531, 553, 605, 625, 627, 646"  \
		-set_of_points "9, 24, 55, 75, 79, 88, 93, 94, 112, 142, 186, 213, 230, 249, 316, 329, 337, 371, 381, 383, 392, 405, 417, 439, 464, 466, 470, 524, 530, 567, 583, 594"  \
		-set_of_points "10, 23, 34, 51, 65, 83, 117, 126, 144, 146, 147, 159, 172, 181, 185, 197, 198, 207, 240, 281, 283, 293, 300, 301, 305, 346, 357, 384, 409, 419, 444, 459, 463, 465, 468, 514, 533, 543, 547, 549, 586, 615, 618, 628"  \
		-set_of_points "2, 12, 48, 65, 87, 120, 189, 246, 305, 323, 354, 375, 434, 435, 455, 482, 496, 557, 586, 595"  \
		-end \
		-classification_prefix quartic_25_0_0 \
		-report \
		-end \
		-end
	pdflatex quartic_25_0_0_classification.tex
	open quartic_25_0_0_classification.pdf



quartic_curve_13_0_report: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-define_quartic_curve QC13_0 -q 13 -catalogue 0 -end \
			-end \
		-with QC13_0 -do \
			-quartic_curve_activity \
				-report \
			-end
	pdflatex quartic_curve_catalogue_q13_iso0_report.tex
	open quartic_curve_catalogue_q13_iso0_report.pdf


quartic_curve_13_1_report: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-define_quartic_curve QC13_1 -q 13 -catalogue 1 -end \
			-end \
		-with QC13_1 -do \
			-quartic_curve_activity \
				-report \
			-end
	pdflatex quartic_curve_catalogue_q13_iso1_report.tex
	open quartic_curve_catalogue_q13_iso1_report.pdf

quartic_curve_25_report: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 2 F -end \
		-loop L 0 18 1 \
			-with P -do \
				-projective_space_activity \
					-define_quartic_curve QC25_%L -q 25 -catalogue %L -end \
				-end \
		-end_loop \
		-print_symbols \
		-loop L 0 18 1 \
			-with QC25_%L -do \
				-quartic_curve_activity \
					-report \
				-end \
		-end_loop \
		-print_symbols
	pdflatex quartic_curve_catalogue_q25_iso0_report.tex
	pdflatex quartic_curve_catalogue_q25_iso1_report.tex
	pdflatex quartic_curve_catalogue_q25_iso2_report.tex
	pdflatex quartic_curve_catalogue_q25_iso3_report.tex
	pdflatex quartic_curve_catalogue_q25_iso4_report.tex
	pdflatex quartic_curve_catalogue_q25_iso5_report.tex
	pdflatex quartic_curve_catalogue_q25_iso6_report.tex
	pdflatex quartic_curve_catalogue_q25_iso7_report.tex
	pdflatex quartic_curve_catalogue_q25_iso8_report.tex
	pdflatex quartic_curve_catalogue_q25_iso9_report.tex
	pdflatex quartic_curve_catalogue_q25_iso10_report.tex
	pdflatex quartic_curve_catalogue_q25_iso11_report.tex
	pdflatex quartic_curve_catalogue_q25_iso12_report.tex
	pdflatex quartic_curve_catalogue_q25_iso13_report.tex
	pdflatex quartic_curve_catalogue_q25_iso14_report.tex
	pdflatex quartic_curve_catalogue_q25_iso15_report.tex
	pdflatex quartic_curve_catalogue_q25_iso16_report.tex
	pdflatex quartic_curve_catalogue_q25_iso17_report.tex
	gs -sDEVICE=pdfwrite -r120 -oquartic_curve_catalogue_q25.pdf \
		quartic_curve_catalogue_q25_iso0_report.pdf \
		quartic_curve_catalogue_q25_iso1_report.pdf \
		quartic_curve_catalogue_q25_iso2_report.pdf \
		quartic_curve_catalogue_q25_iso3_report.pdf \
		quartic_curve_catalogue_q25_iso4_report.pdf \
		quartic_curve_catalogue_q25_iso5_report.pdf \
		quartic_curve_catalogue_q25_iso6_report.pdf \
		quartic_curve_catalogue_q25_iso7_report.pdf \
		quartic_curve_catalogue_q25_iso8_report.pdf \
		quartic_curve_catalogue_q25_iso9_report.pdf \
		quartic_curve_catalogue_q25_iso10_report.pdf \
		quartic_curve_catalogue_q25_iso11_report.pdf \
		quartic_curve_catalogue_q25_iso12_report.pdf \
		quartic_curve_catalogue_q25_iso13_report.pdf \
		quartic_curve_catalogue_q25_iso14_report.pdf \
		quartic_curve_catalogue_q25_iso15_report.pdf \
		quartic_curve_catalogue_q25_iso16_report.pdf \
		quartic_curve_catalogue_q25_iso17_report.pdf
	
	#open quartic_curve_catalogue_q25_iso0_report.pdf


quartic_curve_13_table: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end


# quartic_curves_q13_info.csv

quartic_curve_tables: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 9 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 19 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 27 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 29 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 31 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end

quartic_curve_tables_latex: 
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 1 test.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 quartic_curves_q9_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 quartic_curves_q13_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 quartic_curves_q19_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 quartic_curves_q25_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 quartic_curves_q27_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 quartic_curves_q29_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 quartic_curves_q31_info.csv
	#$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 1 quartic_curves_q9_info.csv
	#pdflatex quartic_curves_q13_info.tex
	#open quartic_curves_q13_info.pdf
	#~/bin/tth quartic_curves_q13_info.tex
	#open quartic_curves_q13_info.html


quartic_curve_19_table: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 19 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end

quartic_curve_19_table_latex: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-csv_file_latex 1 quartic_curves_q19_info.csv
	~/bin/tth quartic_curves_q19_info.tex


quartic_curve_25_table: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end


# quartic_curves_q25_info.csv

quartic_curve_27_table: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 27 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end


# quartic_curves_q27_info.csv

quartic_curve_29_table: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 29 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end


# quartic_curves_q29_info.csv


quartic_curve_31_table: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 31 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-table_of_quartic_curves \
			-end


# quartic_curves_q31_info.csv




surface_25_12:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 -end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf


surface_25_12_t1:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf


surface_25_12_t2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf


surface_25_12_t3:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf



surface_25_12_t4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 0,0,1,0, 13,2,2,1, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf


surface_25_12_t5:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 0,0,1,0, 13,2,2,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 13,1,1,0, 0,0,0,1, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf


PG_2_25:
	$(ORBITER_PATH)orbiter.out \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 2 F -end \
		-with P -do -projective_space_activity -cheat_sheet -end
	pdflatex PG_2_25.tex 
	open PG_2_25.pdf



PG_2_25_lines:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGGL 3 25 -on_k_subspaces 2 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-problem_label PGGL_3_25 \
				-depth 3 -draw_poset -draw_options -radius 200 -end -report -end \
				-recognize "0,25,650" \
				-recognize "430,16,364" \
			-end \
			-orbits_on_subsets 3 \
			-report \
		-end
	pdflatex PGGL_3_25_poset.tex
	open PGGL_3_25_poset.pdf

surface_25_12_t6:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 0,0,1,0, 13,2,2,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 13,1,1,0, 0,0,0,1, 0" \
			-transform "3,8,8,0, 22,13,22,0, 14,19,15,0, 0,0,0,1, 1" \
			-transform_inverse "16,0,0,0, 0,16,0,0, 21,21,21,0, 0,0,0,1, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf



PG_2_25_stab_of_triangle:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-draw_options -end \
		-define G -linear_group -PGGL 3 25 \
			-subgroup_by_generators "triangle_stab" 6912 7 \
			"1,0,0,0,1,0,0,0,1,1" \
			"1,0,0,0,13,0,0,0,13,1" \
			"1,0,0,0,4,0,0,0,6,1" \
			"1,0,0,0,17,0,0,0,13,0" \
			"1,0,0,0,18,0,0,0,4,1" \
			"1,0,0,0,0,11,0,1,0,0" \
			"0,1,0,0,0,20,14,0,0,0" \
		-end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-problem_label PGGL_3_25 \
				-depth 3 -draw_poset -draw_options -radius 200 -end \
				-recognize "8,44,226" \
			-end \
			-orbits_on_subsets 3 \
		-end



surface_25_12_t7:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 0,0,1,0, 13,2,2,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 13,1,1,0, 0,0,0,1, 0" \
			-transform "3,8,8,0, 22,13,22,0, 14,19,15,0, 0,0,0,1, 1" \
			-transform_inverse "16,0,0,0, 0,16,0,0, 21,21,21,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 0,5,0,0, 0,0,17,0, 0,0,0,1, 1" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf



surface_25_12_t8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 0,0,1,0, 13,2,2,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 13,1,1,0, 0,0,0,1, 0" \
			-transform "3,8,8,0, 22,13,22,0, 14,19,15,0, 0,0,0,1, 1" \
			-transform_inverse "16,0,0,0, 0,16,0,0, 21,21,21,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 0,5,0,0, 0,0,17,0, 0,0,0,1, 1" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,24, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf


surface_25_12_t9:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 0,0,1,0, 13,2,2,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 13,1,1,0, 0,0,0,1, 0" \
			-transform "3,8,8,0, 22,13,22,0, 14,19,15,0, 0,0,0,1, 1" \
			-transform_inverse "16,0,0,0, 0,16,0,0, 21,21,21,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 0,5,0,0, 0,0,17,0, 0,0,0,1, 1" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,24, 0" \
			-transform "0,1,0,0, 0,0,1,0, 0,0,0,1, 1,0,0,0, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-report \
			-report_with_group \
		-end
	pdflatex surface_catalogue_q25_iso12_with_group.tex
	open surface_catalogue_q25_iso12_with_group.pdf





surface_25_12_t8_quartic_curves:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 25 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S25_12 -q 25 -catalogue 12 \
			-transform "1,0,0,16, 0,1,0,18, 0,0,1,8, 0,0,1,1, 0" \
			-transform_inverse "16,0,1,0, 3,5,1,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform "3,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 0,0,1,0, 13,2,2,1, 0" \
			-transform_inverse "1,0,0,0, 0,1,0,0, 13,1,1,0, 0,0,0,1, 0" \
			-transform "3,8,8,0, 22,13,22,0, 14,19,15,0, 0,0,0,1, 1" \
			-transform_inverse "16,0,0,0, 0,16,0,0, 21,21,21,0, 0,0,0,1, 0" \
			-transform "1,0,0,0, 0,5,0,0, 0,0,17,0, 0,0,0,1, 1" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,24, 0" \
			-end \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-all_quartic_curves \
		-end \
		-with S25_12 -do \
		-cubic_surface_activity \
			-export_all_quartic_curves \
		-end
	pdflatex surface_catalogue_q25_iso12_quartics.tex
	open surface_catalogue_q25_iso12_quartics.pdf


quartic_curve_13_0_surface: 
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 2 F -end \
		-with P -do \
			-projective_space_activity \
				-define_quartic_curve QC13_0 -q 13 -catalogue 0 -end \
			-end \
		-with QC13_0 -do \
			-quartic_curve_activity \
				-create_surface \
			-end

#surface equation: 0, 0, 9, 0, 3, 0, 1, 0, 12, 7, 0, 4, 4, 0, 7, 12, 6, 2, 5, 10
#9,2,3,4,1,6,12,8,7,9,4,11,4,12,7,14,12,15,6,16,2,17,5,18,10,19

quartic_curve_13_0_surface_create:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S -by_coefficients \
			"9,2,3,4,1,6,12,8,7,9,4,11,4,12,7,14,12,15,6,16,2,17,5,18,10,19" \
			-q 13 -end \
		-end \
		-with S -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_by_coefficients_q13_report.tex
	open surface_by_coefficients_q13_report.pdf

#			-report_with_group \

#############################################################################################################
# Section 7.3: Cubic Surfaces Classification 


SECTION_CUBIC_SURFACES_CLASSIFICATION:








surface_classify_q4:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-draw_options  -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf27 -W -end \
		-end \
		-with Surf27 -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-report -end \
		-end \
		-print_symbols
	pdflatex Surfaces_q4.tex
	open Surfaces_q4.pdf

# time: 0:00



surface_classify_q4_arc_lifting_two_lines:
	$(ORBITER_PATH)orbiter.out -v 10 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-control_six_arcs -problem_label sixarcs_q4 -end \
			-classify_surfaces_through_arcs_and_two_lines \
		-end
	pdflatex surfaces_arc_lifting_4.tex
	open surfaces_arc_lifting_4.pdf




surface_classify_q7:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-draw_options  -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf27 -W -end \
		-end \
		-with Surf27 -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-report -end \
		-end \
		-print_symbols
	pdflatex Surfaces_q7.tex
	open Surfaces_q7.pdf




#############################################################################################################
# Section 7.4: Cubic Surfaces - Isomorphism Testing and Recognition


SECTION_CUBIC_SURFACES_ISOMORPHISM_TESTING_AND_RECOGNITION:


surface_recognize_q7_abcd_2_3_3_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options  -end \
		-define F -finite_field -q 7 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf -W -end \
		-end \
		-with Surf -do \
			-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 7 \
				-family_general_abcd 2 3 3 4 \
				-end \
			-end \
		-end


surface_isomorph_16:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options  -end \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf27 -W -end \
		-end \
		-with Surf27 -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-isomorphism_testing \
				-q 16 -by_coefficients \
				"1,5,1,8,1,9,1,10,1,11,1,12,6,14,6,15,7,18,7,19" -end \
				-q 16 -by_coefficients \
				"13,6,3,8,3,11,13,13,1,19" -end \
			-end \
		-end \
		-print_symbols



# 1 min 8 sec on Mac from scratch (with all data files removed)


surface_recognize_8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options  -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf27 -W -end \
		-end \
		-with Surf27 -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-recognize \
				-q 8 \
				-by_coefficients "1,6,1,8,1,11,1,13,1,19" \
				-end \
			-end \
		-end \
		-print_symbols



surface_recognize_F13_q4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options  -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf27 -W -end \
		-end \
		-with Surf27 -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-identify_F13 \
		-end \
		-print_symbols


surface_sweep_Cayley_13:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options  -end \
		-define F -finite_field -q 13 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-classify_surfaces_with_double_sixes Surf27 -W -end \
		-end \
		-with Surf27 -do \
		-classification_of_cubic_surfaces_with_double_sixes_activity \
			-sweep \
		-end \
		-print_symbols









#############################################################################################################
# Section 7.5: Cubic Surfaces of Dickson type


SECTION_CUBIC_SURFACES_DICKSON:



D6_q2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D6_q2 -q 2 -by_coefficients $(D6) -end \
		-end \
		-with S_D6_q2 -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_by_coefficients_q2_report.tex
	open surface_by_coefficients_q2_report.pdf
	mv surface_by_coefficients_q2_points.txt surface_by_coefficients_q2_D6_points.txt


# 1 line over GF(2)

D3_q4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D3_q4 -q 4 -by_coefficients $(D3) -end \
		-end \
		-with S_D3_q4 -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_by_coefficients_q4_report.tex
	open surface_by_coefficients_q4_report.pdf
	mv surface_by_coefficients_q4_points.txt surface_by_coefficients_q4_D3_points.txt

#surface_by_coefficients_q4_points.txt


D3_q4_canonical: surface_by_coefficients_q4_points.txt
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_points surface_by_coefficients_q4_points.txt -end \
		-classification_prefix surface_by_coefficients_q4_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q4_points_classification.tex
	open surface_by_coefficients_q4_points_classification.pdf

# group order 24




D4_q8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D4_q8 -q 8 -by_coefficients $(D4) -end \
		-end \
		-with S_D4_q8 -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_by_coefficients_q8_report.tex
	open surface_by_coefficients_q8_report.pdf
	mv surface_by_coefficients_q8_points.txt surface_by_coefficients_q8_D4_points.txt





D6_q4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D6_q4 -q 4 -by_coefficients $(D6) -end \
		-end \
		-with S_D6_q4 -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_by_coefficients_q4_report.tex
	open surface_by_coefficients_q4_report.pdf
	mv surface_by_coefficients_q4_points.txt surface_by_coefficients_q4_D6_points.txt

# D6 has 7 lines over GF(4)



D6_q4_canonical: surface_by_coefficients_q4_D6_points.txt
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_points surface_by_coefficients_q4_D6_points.txt -end \
		-classification_prefix surface_by_coefficients_q4_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q4_points_classification.tex
	open surface_by_coefficients_q4_points_classification.pdf


# group order 384 = 192 * 2



D8_q4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D8_q4 -q 4 -by_coefficients $(D8) -end \
		-end \
		-with S_D8_q4 -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_by_coefficients_q4_report.tex
	open surface_by_coefficients_q4_report.pdf
	mv surface_by_coefficients_q4_points.txt surface_by_coefficients_q4_D8_points.txt


D3_D8_q4_canonical:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
			-file_of_points surface_by_coefficients_q4_D3_points.txt \
			-file_of_points surface_by_coefficients_q4_D8_points.txt \
		-end \
		-classification_prefix surface_by_coefficients_q4_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q4_points_classification.tex
	open surface_by_coefficients_q4_points_classification.pdf

# D3 and D8 are isomorphic over GF(4)


D1_q8:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D1_q8 -q 8 -by_coefficients $(D1) -end \
		-end \
		-with S_D1_q8 -do \
		-cubic_surface_activity \
			-report \
		-end
	pdflatex surface_by_coefficients_q8_report.tex
	open surface_by_coefficients_q8_report.pdf

#surface_by_coefficients_q8_points.txt


D1_q8_canonical:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_points surface_by_coefficients_q8_points.txt -end \
		-classification_prefix surface_by_coefficients_q8_points \
		-report \
		-end \
		-end
	pdflatex surface_by_coefficients_q8_points_classification.tex
	open surface_by_coefficients_q8_points_classification.pdf

# group order 2160

## cleaning D1 with 15 lines over F2 and 27 lines over F4:

D1_q4_with_select_double_six:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D1_q4 -q 4 -by_coefficients $(D1) \
			-select_double_six "3,9,15,19,22,26,4,10,14,18,21,25" \
			-end \
		-end \
		-with S_D1_q4 -do \
		-cubic_surface_activity \
			-report \
		-end
	mv surface_by_coefficients_q4_report.tex D1_q4.tex
	pdflatex D1_q4.tex
	open D1_q4.pdf


D1_q4_with_select_double_six_b:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D1_q4 -q 4 -by_coefficients $(D1) \
			-select_double_six "3,9,15,19,22,26,4,10,14,18,21,25" \
			-select_double_six "1,2,3,4,5,0,7,8,9,10,11,6" \
			-end \
		-end \
		-with S_D1_q4 -do \
		-cubic_surface_activity \
			-report \
		-end
	mv surface_by_coefficients_q4_report.tex D1_q4.tex
	pdflatex D1_q4.tex
	open D1_q4.pdf


D1_q4_trans:
	$(ORBITER_PATH)orbiter.out -v 5 -define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity \
		-move_two_lines_in_hyperplane_stabilizer_text  \
		"1,0,0,0, 0,0,1,1" "0,1,1,1, 1,0,1,0" \
		"1,0,0,0, 0,0,0,1" "0,1,0,1, 1,0,1,0" \
		-end
		
D1_q4_with_select_double_six_c:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-define_surface S_D1_q4 -q 4 -by_coefficients $(D1) \
			-select_double_six "3,9,15,19,22,26,4,10,14,18,21,25" \
			-select_double_six "1,2,3,4,5,0,7,8,9,10,11,6" \
			-transform "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
			-end \
		-end \
		-with S_D1_q4 -do \
		-cubic_surface_activity \
			-report \
		-end
	mv surface_by_coefficients_q4_report.tex D1_q4.tex
	pdflatex D1_q4.tex
	open D1_q4.pdf



#############################################################################################################
# 

DICKSON:


#

# Dickson nonsingular:

Dickson_q4_aut:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input \
					-file_of_points D1_points.txt \
					-file_of_points D16_points.txt \
				-end \
				-classification_prefix D \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling D_can_lab.csv \
				-save_cumulative_data D_data.csv  \
				-report \
			-end \
		-end 
	pdflatex D_classification.tex
	open D_classification.pdf






orbits_cubic_surfaces_q3:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G -linear_group -PGL 4 3 -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-end 
	pdflatex poly_orbits_d3_n3_q3.tex
	open poly_orbits_d3_n3_q3.pdf
	
# this takes 3 days and about 150 GB memory on ripoff

orbits_cubic_curves_q2_again:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 3 2 \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-end 
	pdflatex poly_orbits_d3_n2_q2.tex
	open poly_orbits_d3_n2_q2.pdf

orbits_cubic_curves_q3:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 3 3 \
		-end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-end 
	pdflatex poly_orbits_d3_n2_q3.tex
	open poly_orbits_d3_n2_q3.pdf



# compute and analyze properties over F2

poly_orbits_d3_n3_q2_F2.csv: poly_orbits_d3_n3_q2.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-table_of_cubic_surfaces_compute_properties \
			poly_orbits_d3_n3_q2.csv 2 0 \
		-end 


Dickson_q2_analyze: poly_orbits_d3_n3_q2_F2.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-cubic_surface_properties_analyze \
			poly_orbits_d3_n3_q2_F2.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q2_F2_report.tex
	open poly_orbits_d3_n3_q2_F2_report.pdf

# compute and analyze properties over F4

poly_orbits_d3_n3_q2_F4.csv: poly_orbits_d3_n3_q2.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-table_of_cubic_surfaces_compute_properties \
			poly_orbits_d3_n3_q2.csv 2 0 \
		-end 

Dickson_q4_analyze: poly_orbits_d3_n3_q2_F4.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-cubic_surface_properties_analyze \
			poly_orbits_d3_n3_q2_F4.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q2_F4_report.tex
	open poly_orbits_d3_n3_q2_F4_report.pdf

# compute and analyze properties over F8

poly_orbits_d3_n3_q2_F8.csv: poly_orbits_d3_n3_q2.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-table_of_cubic_surfaces_compute_properties \
			poly_orbits_d3_n3_q2.csv 2 0 \
		-end 

Dickson_q8_analyze: poly_orbits_d3_n3_q2_F8.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-cubic_surface_properties_analyze \
			poly_orbits_d3_n3_q2_F8.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q2_F8_report.tex
	open poly_orbits_d3_n3_q2_F8_report.pdf


# compute and analyze properties over F16

poly_orbits_d3_n3_q2_F16.csv: poly_orbits_d3_n3_q2.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-table_of_cubic_surfaces_compute_properties \
			poly_orbits_d3_n3_q2.csv 2 0 \
		-end 


Dickson_q16_analyze: poly_orbits_d3_n3_q2_F16.csv
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 16 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-cubic_surface_properties_analyze \
			poly_orbits_d3_n3_q2_F16.csv 2 \
		-end 
	pdflatex poly_orbits_d3_n3_q2_F16_report.tex
	open poly_orbits_d3_n3_q2_F16_report.pdf



Dickson_q8_surface_canonical:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 8 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input \
					-set_of_points "1,2,3,19,27,35,43,51,59,67,82,90,98,106,114,122,130,138,146,153,161,169,177,185,193,201,209,217,225,233,241,249,257,265,273,281,289,297,305,313,321,329,337,345,353,361,369,377,385,393,401,409,417,425,433,441,449,457,465,473,481,489,497,505,513,521,529,537,545,553,561,569,577" \
				-end \
				-classification_prefix design \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling D_can_lab.csv \
				-save_cumulative_data D_data.csv \
				-report \
			-end \
		-end 


# problem because the group order is too large for int


#############################################################################################################
# Section 7.6:  Cubic Surfaces - ATLAS and Tables


SECTION_CUBIC_SURFACES_ATLAS_AND_TABLES:


MAKE_TABLE_OF_CUBIC_SURFACES=-define P -projective_space 3 F  -end \
		-with P -do \
			-projective_space_activity \
				-table_of_cubic_surfaces \
			-end

cubic_surfaces_tables_up_to_19: 
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 4 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 7 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 8 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 9 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 11 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 13 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 16 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 17 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)


cubic_surfaces_tables_19_37: 
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 19 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 23 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 25 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 27 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 29 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 31 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 32 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 37 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)


cubic_surfaces_tables_41_and_up: 
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 41 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 43 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 47 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 49 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 53 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 59 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 61 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 64 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 67 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 71 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 73 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 79 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 81 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 83 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 89 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 97 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 101 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 103 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 107 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 109 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 113 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 121 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 127 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 128 -end $(MAKE_TABLE_OF_CUBIC_SURFACES)


cubic_surfaces_tables_latex: 
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 1 test.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q4_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q7_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q8_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q9_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q11_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q13_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q16_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q17_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q19_info.csv


cubic_surfaces_tables_latex_big:
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q23_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q25_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q27_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q29_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q31_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q32_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q37_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q41_info.csv
	$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 0 table_of_cubic_surfaces_q43_info.csv
	#$(ORBITER_PATH)orbiter.out -v 3 -csv_file_latex 1 quartic_curves_q9_info.csv
	#pdflatex quartic_curves_q13_info.tex
	#open quartic_curves_q13_info.pdf
	#~/bin/tth quartic_curves_q13_info.tex
	#open quartic_curves_q13_info.html




surface_table:
	$(ORBITER_PATH)orbiter.out -v 3 -make_table_of_surfaces
	pdflatex surfaces_report.tex
	open surfaces_report.pdf




surface_atlas:
	$(ORBITER_PATH)orbiter.out -v 3 -create_surface_atlas 97
	~/bin/tth surface_atlas.tex


surface_reports:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) -create_surface_reports 17

Surface_Orb0_q4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-linear_group -PGL 4 4 -wedge -end \
		-group_theoretic_activities \
		-control_six_arcs -end \
		-define_surface -label_txt "Orb0_q4" -label_tex "Rank-$(Orb0)" -by_rank $(Orb0) 2 -q 4 \
		-end \
		-end 

create_dickson_makefiles_q2:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q2_%03d \
		-N 141  \
		-line_numeric "Surface_Orb%d_q2:" \
		-line_numeric "\t\D(ORBITER_PATH)/orbiter.out -v 3 \\" \
		-line_numeric "\t\t-linear_group -PGL 4 2 -wedge -end \\" \
		-line_numeric "\t\t-group_theoretic_activities \\" \
		-line_numeric "\t\t-control_six_arcs -end \\" \
		-line_numeric "\t\t-define_surface -label_txt \"Orb%d_q2\" \
			-label_tex \"Rank-\D(Orb%d)\" \
			-label_for_summary \"Orb%d\" -by_rank \D(Orb%d) 2 -q 2 \\" \
		-line_numeric "\t\t-end \\" \
		-line_numeric "\t\t-end" \
		-line_numeric "" \
		-end
	cat dickson_q2_* >dickson_q2


create_dickson_makefiles_q4:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q4_%03d \
		-N 141  \
		-line_numeric "Surface_Orb%d_q4:" \
		-line_numeric "\t\D(ORBITER_PATH)/orbiter.out -v 3 \\" \
		-line_numeric "\t\t-linear_group -PGL 4 4 -wedge -end \\" \
		-line_numeric "\t\t-group_theoretic_activities \\" \
		-line_numeric "\t\t-control_six_arcs -end \\" \
		-line_numeric "\t\t-define_surface -label_txt \"Orb%d_q4\" \
			-label_tex \"Rank-\D(Orb%d)\" \
			-label_for_summary \"Orb%d\" -by_rank \D(Orb%d) 2 -q 4 \\" \
		-line_numeric "\t\t-end \\" \
		-line_numeric "\t\t-end" \
		-line_numeric "" \
		-end
	cat dickson_q4_* >dickson_q4

create_dickson_makefiles_q8:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q8_%03d \
		-N 141  \
		-line_numeric "Surface_Orb%d_q8:" \
		-line_numeric "\t\D(ORBITER_PATH)/orbiter.out -v 3 \\" \
		-line_numeric "\t\t-linear_group -PGL 4 8 -wedge -end \\" \
		-line_numeric "\t\t-group_theoretic_activities \\" \
		-line_numeric "\t\t-control_six_arcs -end \\" \
		-line_numeric "\t\t-define_surface -label_txt \"Orb%d_q8\" \
			-label_tex \"Rank-\D(Orb%d)\" -by_rank \D(Orb%d) 2 -q 8 \\" \
		-line_numeric "\t\t-end \\" \
		-line_numeric "\t\t-end" \
		-line_numeric "" \
		-end
	cat dickson_q8_* >q8

create_dickson_makefiles_q16:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q16_%03d \
		-N 141  \
		-line_numeric "Surface_Orb%d_q16:" \
		-line_numeric "\t\D(ORBITER_PATH)/orbiter.out -v 3 \\" \
		-line_numeric "\t\t-linear_group -PGL 4 16 -wedge -end \\" \
		-line_numeric "\t\t-group_theoretic_activities \\" \
		-line_numeric "\t\t-control_six_arcs -end \\" \
		-line_numeric "\t\t-define_surface -label_txt \"Orb%d_q16\" \
			-label_tex \"Rank-\D(Orb%d)\" -by_rank \D(Orb%d) 2 -q 16 \\" \
		-line_numeric "\t\t-end \\" \
		-line_numeric "\t\t-end" \
		-line_numeric "" \
		-end
	cat dickson_q16_* >q16

create_dickson_makefiles_q32:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q32_%03d \
		-N 141  \
		-line_numeric "Surface_Orb%d_q32:" \
		-line_numeric "\t\D(ORBITER_PATH)/orbiter.out -v 3 \\" \
		-line_numeric "\t\t-linear_group -PGL 4 32 -wedge -end \\" \
		-line_numeric "\t\t-group_theoretic_activities \\" \
		-line_numeric "\t\t-control_six_arcs -end \\" \
		-line_numeric "\t\t-define_surface -label_txt \"Orb%d_q32\" \
			-label_tex \"Rank-\D(Orb%d)\" -by_rank \D(Orb%d) 2 -q 32 \\" \
		-line_numeric "\t\t-end \\" \
		-line_numeric "\t\t-end" \
		-line_numeric "" \
		-end
	cat dickson_q32_* >q32

create_dickson_makefiles_q64:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q64_%03d \
		-N 141  \
		-line_numeric "Surface_Orb%d_q64:" \
		-line_numeric "\t\D(ORBITER_PATH)/orbiter.out -v 3 \\" \
		-line_numeric "\t\t-linear_group -PGL 4 64 -wedge -end \\" \
		-line_numeric "\t\t-group_theoretic_activities \\" \
		-line_numeric "\t\t-control_six_arcs -end \\" \
		-line_numeric "\t\t-define_surface -label_txt \"Orb%d_q64\" \
			-label_tex \"Rank-\D(Orb%d)\" -by_rank \D(Orb%d) 2 -q 64 \\" \
		-line_numeric "\t\t-end \\" \
		-line_numeric "\t\t-end" \
		-line_numeric "" \
		-end
	cat dickson_q64_* >q64


create_dickson_makefiles_top_line_q2:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q2_top \
		-N 1 -repeat 141 0 1  "Surface_q2: \\" -command "\tSurface_Orb%d_q2 \\" \
		-end


create_dickson_makefiles_top_line_q4:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q4_top \
		-N 1 -repeat 141 0 1  "Surface_q4: \\" -command "\tSurface_Orb%d_q4 \\" \
		-end

create_dickson_makefiles_top_line_q8:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q8_top \
		-N 1 -repeat 141 0 1  "Surface_q8: \\" -command "\tSurface_Orb%d_q8 \\" \
		-end

create_dickson_makefiles_top_line_q16:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q16_top \
		-N 1 -repeat 141 0 1  "Surface_q16: \\" -command "\tSurface_Orb%d_q16 \\" \
		-end

create_dickson_makefiles_top_line_q32:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q32_top \
		-N 1 -repeat 141 0 1  "Surface_q32: \\" -command "\tSurface_Orb%d_q32 \\" \
		-end

create_dickson_makefiles_top_line_q64:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q64_top \
		-N 1 -repeat 141 0 1  "Surface_q64: \\" -command "\tSurface_Orb%d_q64 \\" \
		-end


surfaces_q4_join:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-csv_file_join 3 \
			Orb0_q4_summary.csv "Surface" \
			Orb1_q4_summary.csv "Surface" \
			Orb2_q4_summary.csv "Surface"

create_dickson_makefiles_join_q2:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q2_join \
		-N 1 -repeat 141 0 1  "join: \n\t\D(ORBITER_PATH)/orbiter.out -v 3 \B" \
		-command "\t\t-csv_file_join Orb%d_q2_summary.csv \"Surface\" \B" \
		-end



create_dickson_makefiles_join_q4:
	$(ORBITER_PATH)orbiter.out -create_files -file_mask dickson_q4_join \
		-N 1 -repeat 141 0 1  "join: \n\t\D(ORBITER_PATH)/orbiter.out -v 3 \B" \
		-command "\t\t-csv_file_join Orb%d_q4_summary.csv \"Surface\" \B" \
		-end

dickson_q2_latex:
	$(ORBITER_PATH)orbiter.out -csv_file_latex q2.csv



#problem:
#orthogonal::lines_on_point_by_line_rank i=1366 / 4225 pt=15447347 pt2=15225451
#orthogonal::lines_on_point_by_line_rank before rank_line
#orthogonal::rank_line p1=15447347 p2=15225451

create_dickson_atlas:
	$(ORBITER_PATH)orbiter.out -create_dickson_atlas
	~/bin/tth dickson_surfaces.tex




#############################################################################################################
# Chapter 8 - Applications
#############################################################################################################


#############################################################################################################
# Section 8.1:  Number Theory


SECTION_NUMBER_THEORY:




inverse_mod_a:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod 18059241 58014043


jacobi_35_41:
	$(ORBITER_PATH)orbiter.out -v 5 -jacobi 35 41
	pdflatex jacobi_35_41.tex
	open jacobi_35_41.pdf


jacobi_33_41:
	$(ORBITER_PATH)orbiter.out -v 5 -jacobi 33 41
	pdflatex jacobi_33_41.tex
	open jacobi_33_41.pdf


jacobi_a:
	$(ORBITER_PATH)orbiter.out -v 5 -jacobi 2221 7817

jacobi_5_19:
	$(ORBITER_PATH)orbiter.out -v 5 -jacobi 5 19

sqrt_mod_7817:
		$(ORBITER_PATH)orbiter.out -v 2 -square_root_mod 2221 7817


equivalence_class_of_fractions:
		$(ORBITER_PATH)orbiter.out -v 2 -equivalence_class_of_fractions 10

#equivalence_class_of_fractions_draw:
#	$(ORBITER_PATH)orbiter.out -draw_matrix table_fractions_N10.csv 20 8





#############################################################################################################
# Section 8.2:  Combinatorics


SECTION_COMBINATORICS:
	

elementary_symmetric_functions_4:
	$(ORBITER_PATH)orbiter.out -make_elementary_symmetric_functions 4 4

elementary_symmetric_functions_8:
	$(ORBITER_PATH)orbiter.out -make_elementary_symmetric_functions 8 8



Char_Sym_4:
	$(ORBITER_PATH)orbiter.out -v 2 -character_table_symmetric_group 4

Char_Sym_5:
	$(ORBITER_PATH)orbiter.out -v 2 -character_table_symmetric_group 5

Char_Sym_6:
	$(ORBITER_PATH)orbiter.out -v 2 -character_table_symmetric_group 6
	

FILE_24_3_TFC_INC="24 24 72\
\n0 1 2 24 27 28 48 53 54 73 79 80 97 105 106 122 131 132 146 157 158 171 175 183 195 203 208 220 225 233 244 258 259 269 272 282 293 300 308 318 325 333 342 352 358 367 379 381 392 398 400 417 428 429 442 443 450 466 471 479 492 497 502 517 519 521 542 548 551 571 574 575              48\
\n0 1 2 24 27 28 48 53 54 73 79 80 97 105 106 122 131 132 146 157 158 171 175 183 195 203 208 220 225 233 244 258 259 269 272 281 293 301 308 318 324 327 342 354 357 367 373 378 392 400 403 417 419 430 442 446 447 466 472 479 492 500 503 518 525 526 545 549 551 571 572 574              48\
\n0 1 2 24 27 28 48 53 54 73 79 80 97 105 106 122 131 132 146 157 158 171 175 179 195 201 207 220 226 232 244 257 258 269 274 277 293 300 307 318 323 329 342 352 356 367 374 381 392 397 406 416 423 431 441 450 454 468 476 477 494 499 503 519 521 525 544 547 550 570 572 575             144\
\n-1 3"



design_test:
	echo $(FILE_24_3_TFC_INC) >24_3_TFC.inc
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define D -design  -end \
		-with D -do \
		-design_activity \
			-canonical_form \
			-input -file_of_incidence_geometries 24_3_TFC.inc 24 24 72 -end \
			-save_ago \
		-end


#############################################################################################################
# Section 8.3: Representation Theory

SECTION_REPRESENTATION_THEORY:



representation_on_polynomials_of_degree_3:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G -linear_group -PGL 4 3 -end \
		-with G -do \
		-group_theoretic_activities \
			-representation_on_polynomials 3 \
		-end 
	$(ORBITER_PATH)orbiter.out -v 2 \
		-loop L 0 9 1 -draw_matrix \
			-input_csv_file PGL_4_3_rep_3_%L.csv \
			-box_width 40 -bit_depth 24  -partition 3 20 20 -end \
		-end_loop




representation_tetrahedral_group_on_polynomials_of_degree_3:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G -linear_group -GL 3 3 \
		-subgroup_by_generators "tetra" "12" 2 \
		"0,1,0,0,0,1,1,0,0" \
		"0,0,1,2,0,0,0,2,0" \
		-end \
		-with G -do \
		-group_theoretic_activities \
			-representation_on_polynomials 3 \
		-end 
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 2 1 -draw_matrix \
		-input_csv_file GL_3_3_Subgroup_tetra_12_rep_3_%L.csv \
		-box_width 40 -bit_depth 24  -partition 3 10 10 -end -end_loop
	open GL_3_3_Subgroup_tetra_12_rep_3_0_draw.bmp
	open GL_3_3_Subgroup_tetra_12_rep_3_1_draw.bmp

# write GL_3_3_Subgroup_tetra_12_rep_3_0.csv


#############################################################################################################
# Section 8.4: Cryptography 


SECTION_CRYPTOGRAPHY:


EC_add:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-with F -do \
		-finite_field_activity \
		-EC_add 1 3 "1,4" "1,4" -end

EC_cyclic_subgroup:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-with F -do \
		-finite_field_activity \
		-EC_cyclic_subgroup 1 3 "1,4" -end


EC_points_13:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 13 -end \
		-with F -do \
		-finite_field_activity \
		-EC_points "EC_2_5_q13" 2 5 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file EC_2_5_q13_points_xy.csv \
		-box_width 20 -bit_depth 24 \
		-partition 2 "1,1,1,1,1,1,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,1,1,1,1,1,1" -end




EC_points_199:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 199 -end \
		-with F -do \
		-finite_field_activity \
		-EC_points "EC_5_7_q199" 5 7 -end
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_matrix -input_csv_file EC_5_7_q199_points_xy.csv \
		-box_width 10 -bit_depth 24 \
		-partition 2 199 199 -end

EC_Koblitz_encoding:
	$(ORBITER_PATH)orbiter.out -v 6 -seed 17 \
		-define F -finite_field -q 199 -end \
		-with F -do \
		-finite_field_activity \
		-EC_Koblitz_encoding 5 7 67 "147,164" "DEADBEEF" \
		-end

EC_bsgs:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 199 -end \
		-with F -do \
		-finite_field_activity \
		-EC_bsgs 5 7 "147,164" 212 \
		"172,158,45,195,50,22,10,103,55,33,50,22,145,105,31,74,73,155,67,60,25,6" \
		-end

EC_bsgs_decode:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 199 -end \
		-with F -do \
		-finite_field_activity \
		-EC_bsgs_decode 5 7 "129,176" 212 \
		"127,188,51,141,85,29,106,90,41,105,179,71,171,2,16,197,183,72,27,129,37,10" \
		"50,179,169,13,153,169,115,116,188,110,176" \
		-end





NTRU_N=7
NTRU_P=3
NTRU_Q=41
NTRU_D=2
NTRUE_XN1="-1,0,0,0,0,0,0,1,"
# D + 1 plus ones and D minus ones
ALICE_PRIVATE_F="-1,0,1,1,-1,0,1"
# D plus ones and D minus ones
ALICE_PRIVATE_G="0,-1,-1,0,1,0,1"

NTRU_Alice1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q $(NTRU_Q) -end \
		-with F -do \
		-finite_field_activity \
		-extended_gcd_for_polynomials $(NTRUE_XN1) $(ALICE_PRIVATE_F) -end

#F_q(x) = 8X^{6} + 26X^{5} + 31X^{4} + 21X^{3} + 40X^{2} + 2X + 37
ALICE_PRIVATE_FQ="37,2,40,21,31,26,8"


NTRU_Alice2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q $(NTRU_P) -end \
		-with F -do \
		-finite_field_activity \
		-extended_gcd_for_polynomials $(NTRUE_XN1) $(ALICE_PRIVATE_F) -end

#F_p(x) = X^{6} + 2X^{5} + X^{3} + X^{2} + X + 1
ALICE_PRIVATE_FP="1,1,1,1,0,2,1"

NTRU_Alice_public_key:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q $(NTRU_Q) -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod $(ALICE_PRIVATE_F) \
			$(ALICE_PRIVATE_G) $(NTRUE_XN1) \
		-end

#C(X)=20X^{6} + 40X^{5} + 2X^{4} + 38X^{3} + 8X^{2} + 26X + 30
ALICE_PUBLIC_KEY="30,26,8,38,2,40,20"

BOB_MESSAGE="1,-1,1,1,0,-1"
BOB_ONE_TIME_KEY="-1,1,0,0,0,-1,1"


NTRU_encrypt:
	$(ORBITER_PATH)orbiter.out -v 2  \
		-define F -finite_field -q $(NTRU_Q) -end \
		-with F -do \
		-finite_field_activity \
		-NTRU_encrypt $(NTRU_N) $(NTRU_P) $(ALICE_PUBLIC_KEY) \
		$(BOB_ONE_TIME_KEY) $(BOB_MESSAGE) -end

# E(X) = 31X^{6} + 19X^{5} + 4X^{4} + 2X^{3} + 40X^{2} + 3X + 25
BOB_ENCRYPT = "25,3,40,2,4,19,31"


NTRU_decrypt1:
	$(ORBITER_PATH)orbiter.out -v 2  \
		-define F -finite_field -q $(NTRU_Q) -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod $(ALICE_PRIVATE_F) \
			$(BOB_ENCRYPT) $(NTRUE_XN1) \
		-end

#C(X)=X^{6} + 10X^{5} + 33X^{4} + 40X^{3} + 40X^{2} + X + 40
ALICE_C1="40,1,40,40,33,10,1"

NTRU_decrypt2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q $(NTRU_Q) -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_center_lift $(ALICE_C1) -end

#A(X)=X^{6} + 10X^{5} - 8X^{4} - X^{3} - X^{2} + X - 1
ALICE_C2="-1,1,-1,-1,-8,10,1"

NTRU_decrypt3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q $(NTRU_P) -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_reduce_mod_p $(ALICE_C2) -end

#A(X)=X^{6} + X^{5} + X^{4} + 2X^{3} + 2X^{2} + X + 2
ALICE_C3="2,1,2,2,1,1,1"

NTRU_decrypt4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q $(NTRU_Q) -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod $(ALICE_PRIVATE_FP) \
			$(ALICE_C3) $(NTRUE_XN1) \
		-end

#C(X)=2X^{5} + X^{3} + X^{2} + 2X + 1
ALICE_C4="1,2,1,1,0,2"


NTRU_decrypt5:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q $(NTRU_P) -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_center_lift $(ALICE_C4) -end

#A(X)= - X^{5} + X^{3} + X^{2} - X + 1
#plaintext BOB_MESSAGE



######


inv_59_mod:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  59 10200

# the inverse of 59 mod 10200 is 2939



RSA_e:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 59 10403 2 "1921,1605,1804,2116,0518"


RSA_d:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 2939 10403 2 "902,3509,9833,3548,5181"


im1:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  869 1843488

#the inverse of 869 mod 1843488 is 386093

# FUNFACTOR:		
RSA_e1:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 386093 1846303 3 "62114,60103,201518"

RSA_d1:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 869 1846303 3 "1248407,345776,317846"





##############################
# 5503*4603 = 25330309
# 5502*4602 = 25320204


im1061:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  1061 25320204
	
# the inverse of 1061 mod 25320204 is 2076209



RSA_e2:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA_encrypt_text 2076209 25330309 3 creamcheese

#-RSA_encrypt_text 386093 1846303 creamcheese
#408918,1735142,239809,654636


RSA_d2:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 1061 25330309 3 "19019931,1619805,740498,2671344"


##############################
# 7253*8171 = 59264263
# 7252*8170 = 59248840


im3:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  2909 59248840
	
#the inverse of 2909 mod 59248840 is 4358629

RSA_e3:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA_encrypt_text 2909 59264263 3 encrypted

RSA_d3:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 4358629 59264263 3 "35270141,9642524,49091707"

#51403,182516,200504 = encrypted


####
# 7879 * 7901 = 62251979
# 7878 * 7900 = 62236200

# e = 

im4:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  583 62236200

# the inverse of 583 mod 62236200 is 32559247

RSA_e4:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA_encrypt_text 583 62251979 3 venividivici

#-RSA_encrypt_text 583 62251979 venividivici
#40513610,53979973,56449676,35068535

RSA_d4:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 32559247 62251979 "40513610,53979973,56449676,35068535"



######
# 7369 * 7127 = 52518863
# 7368 * 7126 = 52504368


im5:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  173 52504368

#the inverse of 173 mod 52504368 is 38543669
	
RSA_e5:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA_encrypt_text 38543669 52518863 3 fascinating

#-RSA_encrypt_text 38543669 52518863 fascinating
#31526751,8962078,51045732,51894467
		

RSA_d5:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 173 52518863 "31526751,8962078,51045732,51894467"


RSA_d6:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA 47177497 55040413 "28702119,48926559"


smooth:
	$(ORBITER_PATH)orbiter.out -v 2 -sift_smooth 100000 100 "2,3,5,7,11,13,17,19"


		
######
# 1999 * 7907 = 15806093
# 1998 * 7906 = 15796188

im7:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  3221 15796188

#the inverse of 3221 mod 15796188 is 10048553


RSA_e7:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA_encrypt_text 10048553 15806093 3 beachandfun
		
#
# 7853 * 7673 = 60256069
# 7852 * 7672 = 60240544

im8:
	$(ORBITER_PATH)orbiter.out -v 2 -inverse_mod  9017 60240544


#the inverse of 9017 mod 60240544 is 14430473

RSA_e8:
	$(ORBITER_PATH)orbiter.out -v 2 -RSA_encrypt_text 9017 60256069 3 strawberry


sqrt_big:
	$(ORBITER_PATH)orbiter.out -v 2 -square_root 1002001
	
sqrt_mod_33_41:
	$(ORBITER_PATH)orbiter.out -v 2 -square_root_mod 33 41
	
quadratic_sieve:
	$(ORBITER_PATH)orbiter.out -v 5 -quadratic_sieve 31 500 1



######

pseudoprime3:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 3 5 0 0 
	pdflatex pseudoprime_3.tex
	open pseudoprime_3.pdf

pseudoprime10:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 10 5 5 5
	pdflatex pseudoprime_10.tex
	open pseudoprime_10.pdf

# 4460190157


PR10_test1:
	$(ORBITER_PATH)orbiter.out -v 5 -power_mod 1293 2230095078 4460190157
	$(ORBITER_PATH)orbiter.out -v 5 -power_mod 9865 2230095078 4460190157
	$(ORBITER_PATH)orbiter.out -v 5 -power_mod 19645 2230095078 4460190157
	$(ORBITER_PATH)orbiter.out -v 5 -power_mod 974586571 2230095078 4460190157
	$(ORBITER_PATH)orbiter.out -v 5 -power_mod 974586571 1486730052 4460190157
	$(ORBITER_PATH)orbiter.out -v 5 -power_mod 974586571 15222492 4460190157
	$(ORBITER_PATH)orbiter.out -v 5 -power_mod 974586571 284796 4460190157



pseudoprime11:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 11 5 5 5 
	pdflatex pseudoprime_11.tex
	open pseudoprime_11.pdf

# 63814633367

# product is 284625399616057168619


pseudoprime20:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 20 5 5 5 
	pdflatex pseudoprime_20.tex
	open pseudoprime_20.pdf


PR10:
	$(ORBITER_PATH)orbiter.out -v 5 -primitive_root 4460190157


# mistake! long integer overflow
# a primitive root modulo 165222861 is 1293




pseudoprime50:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 50 5 0 0
	pdflatex pseudoprime_50.tex
	open pseudoprime_50.pdf


#91322792878581218181431392170986926262336688354473

pseudoprime51:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 51 5 5 5 
	pdflatex pseudoprime_51.tex
	open pseudoprime_51.pdf

#754600727746834470214089702490004944659715367045417

# product 68912245966050819606199994423264315732335295324400658436661744403244049572914094379904326661586100241

pseudoprime30:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 30 5 5 5 
	pdflatex pseudoprime_30.tex
	open pseudoprime_30.pdf

# 286525565474504516914595596387

pseudoprime31:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 31 5 5 5 
	pdflatex pseudoprime_31.tex
	open pseudoprime_31.pdf
#8777266765422645523724129853331

# 2514911323283298698837184692002835573476743643265896783515097

# maybe 2 seconds


pseudoprime33:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 33 5 5 5 
	pdflatex pseudoprime_33.tex
	open pseudoprime_33.pdf


#371674199498295345543363004459891


pseudoprime34:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 34 5 5 5 
	pdflatex pseudoprime_34.tex
	open pseudoprime_34.pdf

#9309708224110488378214945245346817

# 3460178351758962531912872979731874528849142238619677890786061016947
# 18 sec



pseudoprime35:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 35 5 5 5 
	pdflatex pseudoprime_35.tex
	open pseudoprime_35.pdf

#81329557792505271120435930267680203

pseudoprime36:
	$(ORBITER_PATH)orbiter.out -v 5 -seed 2531011 -find_pseudoprime 36 5 5 5 
	pdflatex pseudoprime_36.tex
	open pseudoprime_36.pdf

#162624680891993404333363207561599139

# 13226193383093105242537919350220354135219441323641636665484262532145217
# factoring takes 46 seconds


MATH360_hw2:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity \
		-parse_and_evaluate "test" "" "a+b" "a=8,b=14" -end
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity \
		-parse_and_evaluate "test" "" "a*b" "a=9,b=13" -end
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity \
		-parse_and_evaluate "test" "" "a*a*a*a*a" "a=9" -end
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity \
		-parse_and_evaluate "test" "" "(a+b)*(a+b)"  "a=5,b=7" -end
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity \
		-parse_and_evaluate "test" "" "a*a+b*b"  "a=5,b=7" -end


F_256_Rijndahl:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 256 -override_polynomial 283 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end




ntt:
	$(ORBITER_PATH)orbiter.out -v 2 -NTT 4 17 ntt4.cpp



#	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix -input_csv_file PG_2_4_singer_incma_subgroup_index_3.csv -box_width 20 -bit_depth 24  -partition 3 7,7,7 7,7,7 -end

# problem:

FGDTP:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_F_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_Fv_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_AAv_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_G_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_D_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_T_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_Tv_k4.csv 20 8
	
no:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_P_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_F_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_Gr_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_Dr_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_Tr_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_Pr_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_G_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_D_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_T_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_P_k3.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_G_k2.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_D_k2.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_T_k2.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_P_k2.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_the_P_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix ntt_L_k4.csv 20 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix Fourier_q17_N16.csv 40 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix Fourier_q17_N8.csv 40 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix Fourier_q17_N4.csv 40 8
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix Fourier_q17_N2.csv 40 8
	




ntt4.out:  ntt4.o ;
	$(DISCRETA_CPP)  ntt4.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

all_square_roots_mod_n_1549411:
	$(ORBITER_PATH)orbiter.out -v 3 -all_square_roots_mod_n 1075922 1549411
	


power_mod_211:
	$(ORBITER_PATH)orbiter.out -v 3 -power_mod_n 2 211
	$(ORBITER_PATH)orbiter.out -v 3 -plot_function power_mod_n_a2_n211.csv
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file power_mod_n_a2_n211_graph.csv \
		-box_width 10 -bit_depth 8  -partition 3 211 211 -end

power_mod_2_31:
	$(ORBITER_PATH)orbiter.out -v 3 -power_mod_n 2 31
	$(ORBITER_PATH)orbiter.out -v 3 -plot_function power_mod_n_a2_n31.csv
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file power_mod_n_a2_n31_graph.csv \
		-box_width 10 -bit_depth 8  -partition 3 31 31 -end

power_mod_3_31:
	$(ORBITER_PATH)orbiter.out -v 3 -power_mod_n 3 31
	$(ORBITER_PATH)orbiter.out -v 3 -plot_function power_mod_n_a3_n31.csv
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file power_mod_n_a3_n31_graph.csv \
		-box_width 10 -bit_depth 8  -partition 3 31 31 -end



#############################################################################################################
# Chapter 9 - Coding Theory
#############################################################################################################



#############################################################################################################
# Section 9.1: Coding Theory

SECTION_CODING_THEORY_INTRODUCTION:



Hamming_4_2:
	$(ORBITER_PATH)orbiter.out -Hamming_graph 4 2
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Hamming_n4_q2.csv \
		-box_width 20 -bit_depth 24  -partition 4 16 16 -end
	open Hamming_n4_q2_draw.bmp



code_5_2_3_diagram:
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "code_5_2_3" \
		$(CODE_5_2_3_CODEWORDS) 5 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file code_5_2_3_diagram_01_5_4.csv \
		-box_width 25 -bit_depth 24  -partition 4 8 4 -end



Hamming_5_2_graph:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -Hamming 5 2 -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Hamming_5_2.csv \
		-box_width 8 -bit_depth 24 -partition 4 32 32 -end
	dot -Tpng Hamming_5_2.gv >Hamming_5_2.png


Hamming_5_2_with_5_2_3_code:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -Hamming 5 2 \
			-subset "_code_5_2_3" "\\_code\\_5\\_2\\_3" $(CODE_5_2_3_CODEWORDS) -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end \
		-with G -do \
		-graph_theoretic_activity -automorphism_group -end
	pdflatex Hamming_5_2_code_5_2_3_report.tex
	open Hamming_5_2_code_5_2_3_report.pdf

# group has order 32



code_3_2_aut:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end  \
		-define P -projective_space 1 F -end \
		-with P -do \
		-projective_space_activity -canonical_form_of_code \
			"3_2" 2 3 "1,0,1, 0,1,1"  -end
	pdflatex classify_code_3_2_classification.tex
	open classify_code_3_2_classification.pdf





code_6:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-general_code_binary 6 "0,60,50,41,14,21,27,39"
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file code_matrix_8_8.csv \
		-box_width 20 -bit_depth 24 \
		-partition 2 "1,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,1" -end
	pdflatex code_6_8.tex
	open code_6_8.pdf
	open code_matrix_8_8_draw.bmp


# linear code with generator matrix
# 111100
# 110010
# 101001

H_6_diagram:
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "H6" \
		"" 6 






F_16_7_power_5:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 16 -end \
		-with F -do -finite_field_activity \
			-parse_and_evaluate "test" "" "(a*a)*(a*a)*a"  "a=7" -end
	dot -Tpng test.gv >test.png


# the answer is 11.





Queens_graph:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -non_attacking_queens_graph 8 -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end \
		-with G -do \
		-graph_theoretic_activity -automorphism_group -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques \
			-target_size 8 -output_file 8queens -end \
		-end 



	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix  -input_csv_file non_attacking_queens_graph_8.csv \
	#	-box_width 8 -bit_depth 24 -partition 4 64 64 -end
	#twopi -Tpng non_attacking_queens_graph_8.gv >non_attacking_queens_graph_8.png
	pdflatex non_attacking_queens_graph_8_report.tex
	open non_attacking_queens_graph_8_report.pdf



#############################################################################################################
# Section 9.2: Hamming codes

SECTION_CODING_THEORY_HAMMING_CODES:




Hamming_generator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F2 -finite_field -q 2 -end \
		-with F2 -do -finite_field_activity  \
		-nullspace 3 7 "1,0,1,0,1,0,1, 0,1,1,0,0,1,1, 0,0,0,1,1,1,1" \
		-normalize_from_the_right \
		-end
	pdflatex nullspace_3_7.tex
	open nullspace_3_7.pdf

# basis in binary:
# 67,37,22,15

Hamming_code_words:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-linear_code_through_basis 7 "67,37,22,15" 
	pdflatex code_n7_k4_q2.tex
	open code_n7_k4_q2.pdf


Hamming_code_diagram:
	- $(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4" \
		$(HAMMING_CODE_CODEWORDS) 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Hamming_7_4_diagram_01_7_16.csv \
		-box_width 25 -bit_depth 24  -partition 4 16 8 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Hamming_7_4_diagram_7_16.csv \
		-box_width 25 -bit_depth 14  -partition 4 16 8 -end
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_0" "0" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_1" "67" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_2" "37" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_3" "102" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_4" "22" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_5" "85" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_6" "51" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_7" "112" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_8" "15" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_9" "76" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_10" "42" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_11" "105" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_12" "25" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_13" "90" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_14" "60" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_15" "127" 7 -metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 16 1 -draw_matrix \
		-input_csv_file Hamming_7_4_word_%L_diagram_7_1.csv \
		-box_width 25 -bit_depth 8  -partition 4 16 8 -end \
		-end_loop

	
	
	
code_Hamming_systematic:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-linear_code_through_basis 7  "67,37,22,15" 
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file code_matrix_16_8.csv \
		-box_width 25 -bit_depth 8  -partition 2 16 8 -end
	open code_matrix_16_8_draw.bmp
	pdflatex code_7_16.tex
	open code_7_16.pdf


#	$(DRAW_PATH)/draw_matrix.out -read code_matrix_256_256.csv -box_width 10 

# 1 0 0 0 0 1 1 = 67
# 0 1 0 0 1 0 1 = 37
# 0 0 1 0 1 1 0 = 22
# 0 0 0 1 1 1 1 = 15



Hamming_RREF:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
			-RREF 4 7 $(CODE_HAMMING) -end
	pdflatex RREF_example_q2_4_7.tex
	gs -sDEVICE=png16 -dFIXEDMEDIA -dDEVICEWIDTHPOINTS=500 -dDEVICEHEIGHTPOINTS=450 \
		-r240 -oHamming_dual_page%02d.png \
		RREF_example_q2_4_7.pdf



Hamming_nullspace:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F2 -finite_field -q 2 -end \
		-with F2 -do -finite_field_activity  \
		-nullspace 4 7 $(CODE_HAMMING) \
		-normalize_from_the_right \
		-end
	pdflatex nullspace_4_7.tex
	open nullspace_4_7.pdf


#check equations of the Hamming code:
# a4+a5+a6+a7 =1+0+1+0=0 mod2 OK. 
# a2+a3+a6+a7 =0+1+1+0=0 mod2 OK. 
# a1+a3+a5+a7 =1+1+0+0=0 mod2 OK.

#1010101
#0110011
#0001111

#[23,12,8]
#0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 132, 913, 1460, 1750, 1898, 2518, 2787, 2874, 3320, 3357, 3662

#[24,12,8]
#0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 132, 913, 1460, 1750, 1898, 2518, 2787, 2874, 3320, 3357, 3662, 4004




Hamming_long:
	$(ORBITER_PATH)orbiter.out -v 2 -long_code 7 4 \
		"0,5,6" \
		"1,4,6" \
		"2,4,5" \
		"3,4,5,6" 
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 16 1 -draw_matrix \
		-input_csv_file long_code_genma_n7_k4_codeword_%L.csv \
		-box_width 25 -bit_depth 8  -partition 3 4 2 -end \
		-end_loop


#long_code_genma_n7_k4_codeword_0.csv
#long_code_genma_n7_k4_codeword_15.csv
#Weight distribution:( 0, 3^7, 4^7, 7 )

Hamming_code_macwilliams:
	$(ORBITER_PATH)orbiter.out -v 2 -make_macwilliams_system 7 4 2
	pdflatex MacWilliams_n7_k4_q2.tex
	open MacWilliams_n7_k4_q2.pdf

Hamming_singer:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -radius 200 -end \
		-define G -linear_group -PGL 3 2 -singer 1 -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-orbits_on_points \
		-end
	pdflatex PGL_3_2_Singer_3_2_1_report.tex
	open PGL_3_2_Singer_3_2_1_report.pdf

# cycle is 0,1,2,5,3,4,6

#1001110
#0100111
#0011101



Hamming_cyclic_generator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F2 -finite_field -q 2 -end \
		-with F2 -do -finite_field_activity  \
		-nullspace 3 7 "1,0,0,1,1,1,0, 0,1,0,0,1,1,1, 0,0,1,1,1,0,1" \
		-normalize_from_the_right \
		-end
	pdflatex nullspace_3_7.tex
	open nullspace_3_7.pdf

Hamming_cyclic_long:
	$(ORBITER_PATH)orbiter.out -v 2 -long_code 7 4 \
		"0,4,6" \
		"1,4,5,6" \
		"2,4,5" \
		"3,5,6" 
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 16 1 -draw_matrix \
		-input_csv_file long_code_genma_n7_k4_codeword_%L.csv \
		-box_width 25 -bit_depth 8  -partition 3 4 2 -end \
		-end_loop


Hamming_cyclic:
	$(ORBITER_PATH)orbiter.out -v 2 -linear_code_through_basis 7  "69,39,22,11"
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file code_matrix_16_8.csv \
		-box_width 25 -bit_depth 8  -partition 2 16 8 -end
	open code_matrix_16_8_draw.bmp
	pdflatex code_7_16.tex
	open code_7_16.pdf

Hamming_cyclic_clean_ns:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F2 -finite_field -q 2 -end \
		-with F2 -do -finite_field_activity  \
		-nullspace 4 7 "1,0,1,1,0,0,0, 0,1,0,1,1,0,0, 0,0,1,0,1,1,0, 0,0,0,1,0,1,1" \
		-normalize_from_the_right \
		-end
	pdflatex nullspace_4_7.tex
	open nullspace_4_7.pdf

Hamming_cyclic_clean:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-linear_code_through_basis 7  "88,44,22,11"
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file code_matrix_16_8.csv \
		-box_width 25 -bit_depth 8  -partition 2 16 8 -end
	open code_matrix_16_8_draw.bmp
	pdflatex code_7_16.tex
	open code_7_16.pdf


Hamming_cyclic_clean_long:
	$(ORBITER_PATH)orbiter.out -v 2 -long_code 7 4 \
		"0,2,3" \
		"1,3,4" \
		"2,4,5" \
		"3,5,6" 
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 16 1 -draw_matrix \
		-input_csv_file long_code_genma_n7_k4_codeword_%L.csv \
		-box_width 25 -bit_depth 8  -partition 3 4 2 -end -end_loop


	


#11111111 = 255
#01010101 = 170
#00110011 = 204
#00001111 = 240


#############################################################################################################
# Section 9.3: Coding Theory - Golay codes

SECTION_CODING_THEORY_GOLAY_CODES:



Golay23_code_words:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-linear_code_through_columns_of_parity_check_projectively 12 \
		$(GOLAY_23_COLUMN_RANKS_PROJECTIVELY)
	pdflatex code_n23_k12_q2.tex
	open code_n23_k12_q2.pdf



Golay23_code_diagram:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-code_diagram_from_file "Golay_23" \
		codewords_n23_k12_q2.csv 23 -enhance 4
		
		#-metric_balls 3


Golay23_code_diagram_draw:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Golay_23_diagram_01_23_4096.csv \
		-box_width 4 -bit_depth 8  -partition 20 4096 2048 -end





#############################################################################################################
# Section 9.4: Coding Theory - CRC codes

SECTION_CODING_THEORY_CRC_CODES:






encode_text_5bits:
	$(ORBITER_PATH)orbiter.out -encode_text_5bits \
		"Hithere" "text.csv"
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
			-polynomial_division_from_file \
			text.csv 13 -end
	pdflatex polynomial_division_file_13.tex
	open polynomial_division_file_13.pdf

#\usepackage[a1paper]{geometry}

encode_text_5bits_check:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division_from_file text_with_1error.csv 13 -end
	pdflatex polynomial_division_file_13.tex
	open polynomial_division_file_13.pdf

encode_constitution_field_induction:
	$(ORBITER_PATH)orbiter.out -encode_text_5bits  "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defense, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America." "constitution.csv"
	$(ORBITER_PATH)orbiter.out -reformat "constitution.csv" "constitution_80.csv" 80
	$(ORBITER_PATH)orbiter.out -field_induction  "constitution.csv"  "constitution_32.csv" 5
	$(ORBITER_PATH)orbiter.out -reformat "constitution_32.csv" "constitution_32_16.csv" 16
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file constitution_80.csv \
		-box_width 20 -bit_depth 24  -partition 5 "21" "80"  -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file constitution_32_16.csv \
		-box_width 20 -bit_depth 24  -partition 5 "21" "16"  -end



encode_text_5bits_1error:
	$(ORBITER_PATH)orbiter.out -encode_text_5bits \
		"Hithere" "text.csv"
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division_from_file_all_k_bit_error_patterns \
			text.csv 13 1 -end
	pdflatex polynomial_division_file_all_1_error_patterns_13.tex
	open polynomial_division_file_all_1_error_patterns_13.pdf

encode_text_zero_5bits_1error:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division_from_file_all_k_bit_error_patterns text_zero.csv 13 1 -end
	pdflatex polynomial_division_file_all_1_error_patterns_13.tex
	open polynomial_division_file_all_1_error_patterns_13.pdf

encode_text_zero_5bits_2error:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division_from_file_all_k_bit_error_patterns text_zero.csv 13 2 -end
	pdflatex polynomial_division_file_all_2_error_patterns_13.tex
	open polynomial_division_file_all_2_error_patterns_13.pdf

encode_text_5bits_2error:
	$(ORBITER_PATH)orbiter.out -encode_text_5bits  "Hithere" "text.csv"
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division_from_file_all_k_bit_error_patterns text.csv 13 2 -end
	pdflatex polynomial_division_file_all_2_error_patterns_13.tex
	open polynomial_division_file_all_2_error_patterns_13.pdf

encode_text_5bits_error_0_7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -polynomial_division_from_file text_error_0_7.csv 13 -end
	pdflatex polynomial_division_file_13.tex
	open polynomial_division_file_13.pdf






CRC_2_4_3:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 4 3 -end

#algebra_global::find_CRC_polynomials info=4 check=3 nb_sol=3
#0 : 1101
#1 : 1011
#2 : 1111


CRC_2_4_4:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 4 4 -end

CRC_2_5_4:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 5 4 -end

CRC_2_6_4:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 6 4 -end

CRC_2_7_4:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 7 4 -end

CRC_2_8_4:
	$(ORBITER_PATH)orbiter.out -v 2 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 8 4 -end

CRC_2_9_4:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 9 4 -end

CRC_2_9_5:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 9 5 -end

CRC_2_10_5:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 10 5 -end

CRC_2_32_6:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 32 6 -end

CRC_2_64_8:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 64 8 -end

CRC_2_128_8:
	$(ORBITER_PATH)orbiter.out -v 1 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 128 8 -end

CRC_2_128_16:
	$(ORBITER_PATH)orbiter.out -v 1 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 128 16 -end

CRC_3_128_8:
	$(ORBITER_PATH)orbiter.out -v 1 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 128 8 -end

CRC_3_128_10:
	$(ORBITER_PATH)orbiter.out -v 1 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 128 10 -end



CRC_2_256_8:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 256 8 -end

# no solution

CRC_2_256_10:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 256 10 -end


#algebra_global::find_CRC_polynomials info=256 check=10 nb_sol=368
# 0:13 on Mac


CRC_2_256_12:
	$(ORBITER_PATH)orbiter.out -v 1 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 2 256 12 -end


#algebra_global::find_CRC_polynomials info=256 check=12 nb_sol=2654
#User time: 2:0

CRC_3_256_10:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 256 10 -end


#algebra_global::find_CRC_polynomials info=256 check=10 nb_sol=140
#Orbiter session finished.
#User time: 8:7


CRC_3_256_16:
	$(ORBITER_PATH)orbiter.out -v 1 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 256 16 -end



CRC_3_8_4:
	$(ORBITER_PATH)orbiter.out -v 2 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 8 4 -end



CRC_3_64_8:
	$(ORBITER_PATH)orbiter.out -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 64 8 -end




CRC_poly_3_128_10:
	$(ORBITER_PATH)orbiter.out -v 1 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity -find_CRC_polynomials 3 128 10 -end


#  12/26/2020: 243 polynomials in 0:56 minutes on Mac






#############################################################################################################
# Section 9.5: Coding Theory - Reed-Muller codes

SECTION_CODING_THEORY_REED_MULLER_CODES:









RM_3_1_code_words:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-linear_code_through_basis 8 "255,170,204,240" 
	pdflatex code_n8_k4_q2.tex
	open code_n8_k4_q2.pdf

#Codewords: (0,255,170,85,204,51,102,153,240,15,90,165,60,195,150,105)
	

RM_3_group:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end  \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_of_code "RM_3_1" 4 8 $(CODE_RM_3_1_GENMA)  -end
	pdflatex classify_code_RM_3_1_classification.tex
	open classify_code_RM_3_1_classification.pdf

# group order 1344

RM_3_1_diagram:
	$(ORBITER_PATH)orbiter.out -v 2 -code_diagram "RM_3_1" \
		"0,255,170,85,204,51,102,153,240,15,90,165,60,195,150,105" 8 \
		-metric_balls 1
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file RM_3_1_diagram_01_8_16.csv \
		-box_width 25 -bit_depth 8  -partition 4 16 16 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file RM_3_1_diagram_8_16.csv \
		-box_width 25 -bit_depth 8  -partition 4 16 16 -end
	#- $(ORBITER_PATH)orbiter.out -v 2 -code_diagram "Hamming_7_4_word_0" "0" 7 -metric_balls 1


RM_3_1_split:
	$(ORBITER_PATH)orbiter.out -split_by_values RM_3_1_holes_8_16.csv 

RM_3_1_holes_draw:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-loop L 0 3 1 -draw_matrix \
		-input_csv_file RM_3_1_holes_8_16_value%L.csv \
		-box_width 25 -bit_depth 8  -partition 5 16 16 -end \
		-end_loop

RM_3_1_hole0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-algebraic_normal_form RM_3_1_holes_8_16_value0.csv 8 -end

# E_0 + E_1 + E_2 + E_3 + E_4


RM_3_1_hole1:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-algebraic_normal_form RM_3_1_holes_8_16_value1.csv 8 -end

#E_1 = X_0X_8^7 + X_1X_8^7 + X_2X_8^7 + X_3X_8^7 + X_4X_8^7 + X_5X_8^7 + X_6X_8^7 + X_7X_8^7

RM_3_1_hole2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-algebraic_normal_form RM_3_1_holes_8_16_value2.csv 8 -end

#X0*X1*X8^6 + X0*X2*X8^6 + X0*X3*X8^6 + X0*X4*X8^6 + X0*X5*X8^6 + X0*X6*X8^6 + X0*X7*X8^6 + X1*X2*X8^6 + X1*X3*X8^6 + X1*X4*X8^6 + X1*X5*X8^6 + X1*X6*X8^6 + X1*X7*X8^6 + X2*X3*X8^6 + X2*X4*X8^6 + X2*X5*X8^6 + X2*X6*X8^6 + X2*X7*X8^6 + X3*X4*X8^6 + X3*X5*X8^6 + X3*X6*X8^6 + X3*X7*X8^6 + X4*X5*X8^6 + X4*X6*X8^6 + X4*X7*X8^6 + X5*X6*X8^6 + X5*X7*X8^6 + X6*X7*X8^6 + X0*X1*X2*X8^5 + X0*X1*X3*X8^5 + X0*X1*X4*X8^5 + X0*X1*X5*X8^5 + X0*X1*X6*X8^5 + X0*X1*X7*X8^5 + X0*X2*X3*X8^5 + X0*X2*X4*X8^5 + X0*X2*X5*X8^5 + X0*X2*X6*X8^5 + X0*X2*X7*X8^5 + X0*X3*X4*X8^5 + X0*X3*X5*X8^5 + X0*X3*X6*X8^5 + X0*X3*X7*X8^5 + X0*X4*X5*X8^5 + X0*X4*X6*X8^5 + X0*X4*X7*X8^5 + X0*X5*X6*X8^5 + X0*X5*X7*X8^5 + X0*X6*X7*X8^5 + X1*X2*X3*X8^5 + X1*X2*X4*X8^5 + X1*X2*X5*X8^5 + X1*X2*X6*X8^5 + X1*X2*X7*X8^5 + X1*X3*X4*X8^5 + X1*X3*X5*X8^5 + X1*X3*X6*X8^5 + X1*X3*X7*X8^5 + X1*X4*X5*X8^5 + X1*X4*X6*X8^5 + X1*X4*X7*X8^5 + X1*X5*X6*X8^5 + X1*X5*X7*X8^5 + X1*X6*X7*X8^5 + X2*X3*X4*X8^5 + X2*X3*X5*X8^5 + X2*X3*X6*X8^5 + X2*X3*X7*X8^5 + X2*X4*X5*X8^5 + X2*X4*X6*X8^5 + X2*X4*X7*X8^5 + X2*X5*X6*X8^5 + X2*X5*X7*X8^5 + X2*X6*X7*X8^5 + X3*X4*X5*X8^5 + X3*X4*X6*X8^5 + X3*X4*X7*X8^5 + X3*X5*X6*X8^5 + X3*X5*X7*X8^5 + X3*X6*X7*X8^5 + X4*X5*X6*X8^5 + X4*X5*X7*X8^5 + X4*X6*X7*X8^5 + X5*X6*X7*X8^5 + X0*X1*X2*X4*X8^4 + X0*X1*X2*X5*X8^4 + X0*X1*X2*X6*X8^4 + X0*X1*X2*X7*X8^4 + X0*X1*X3*X4*X8^4 + X0*X1*X3*X5*X8^4 + X0*X1*X3*X6*X8^4 + X0*X1*X3*X7*X8^4 + X0*X1*X4*X6*X8^4 + X0*X1*X4*X7*X8^4 + X0*X1*X5*X6*X8^4 + X0*X1*X5*X7*X8^4 + X0*X2*X3*X4*X8^4 + X0*X2*X3*X5*X8^4 + X0*X2*X3*X6*X8^4 + X0*X2*X3*X7*X8^4 + X0*X2*X4*X5*X8^4 + X0*X2*X4*X7*X8^4 + X0*X2*X5*X6*X8^4 + X0*X2*X6*X7*X8^4 + X0*X3*X4*X5*X8^4 + X0*X3*X4*X6*X8^4 + X0*X3*X5*X7*X8^4 + X0*X3*X6*X7*X8^4 + X0*X4*X5*X6*X8^4 + X0*X4*X5*X7*X8^4 + X0*X4*X6*X7*X8^4 + X0*X5*X6*X7*X8^4 + X1*X2*X3*X4*X8^4 + X1*X2*X3*X5*X8^4 + X1*X2*X3*X6*X8^4 + X1*X2*X3*X7*X8^4 + X1*X2*X4*X5*X8^4 + X1*X2*X4*X6*X8^4 + X1*X2*X5*X7*X8^4 + X1*X2*X6*X7*X8^4 + X1*X3*X4*X5*X8^4 + X1*X3*X4*X7*X8^4 + X1*X3*X5*X6*X8^4 + X1*X3*X6*X7*X8^4 + X1*X4*X5*X6*X8^4 + X1*X4*X5*X7*X8^4 + X1*X4*X6*X7*X8^4 + X1*X5*X6*X7*X8^4 + X2*X3*X4*X6*X8^4 + X2*X3*X4*X7*X8^4 + X2*X3*X5*X6*X8^4 + X2*X3*X5*X7*X8^4 + X2*X4*X5*X6*X8^4 + X2*X4*X5*X7*X8^4 + X2*X4*X6*X7*X8^4 + X2*X5*X6*X7*X8^4 + X3*X4*X5*X6*X8^4 + X3*X4*X5*X7*X8^4 + X3*X4*X6*X7*X8^4 + X3*X5*X6*X7*X8^4

# E_2 + E_3 + E_4

RM_4_1:
	- $(ORBITER_PATH)orbiter.out -v 2 \
		-linear_code_through_columns_of_parity_check 5 \
		"1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31" 
	pdflatex code_n16_k5_q2.tex
	open code_n16_k5_q2.pdf


# codewords_n16_k5_q2.csv

RM_4_group:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end  \
		-define P -projective_space 4 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_of_code "RM_4_1" \
			5 16 $(CODE_RM_4_1_GENMA)  -end
	pdflatex classify_code_RM_4_1_classification.tex
	open classify_code_RM_4_1_classification.pdf


# group order 322560 = 24*30*28*16


RM_4_1_diagram:
	- $(ORBITER_PATH)orbiter.out -v 2 \
		-code_diagram_from_file "RM_4_1" \
		codewords_n16_k5_q2.csv 16 
		#-enhance 4 
		#-metric_balls 3

RM_4_1_diagram_draw:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file RM_4_1_diagram_01_16_32.csv \
		-box_width 25 -bit_depth 8  -partition 10 256 256 -end


RM_4_1_split:
	$(ORBITER_PATH)orbiter.out -split_by_values RM_4_1_holes_16_32.csv 

RM_4_1_diagram_draw_holes:
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file RM_4_1_holes_16_32.csv \
		-box_width 25 -bit_depth 8  -partition 10 256 256 -end
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 7 1 -draw_matrix \
		-input_csv_file RM_4_1_holes_16_32_value%L.csv \
		-box_width 25 -bit_depth 8  -partition 10 256 256 -end -end_loop



RM_4_1_diagram_metric_balls:
	- $(ORBITER_PATH)orbiter.out -v 2 -code_diagram_from_file "RM_4_1" \
		codewords_n16_k5_q2.csv 16 -metric_balls 3
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file RM_4_1_diagram_16_32.csv \
		-box_width 25 -bit_depth 8  -partition 10 256 256 -end



RM_4_1_hole0:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-algebraic_normal_form RM_4_1_holes_16_32_value0.csv 16 -end







Reed_Muller_6:
	$(ORBITER_PATH)orbiter.out -v 2 -long_code 64 7 \
		-set_builder -loop 0 64 1 -end \
		-set_builder -loop 0 32 1 -affine_function 2 1 -end \
		-set_builder -loop 0 16 1 -affine_function 4 2 -clone_with_affine_function 4 3 -end \
		-set_builder -set_builder -set_builder -loop 0 4 1 -affine_function 1 4 \
			-clone_with_affine_function 1 12 -end -clone_with_affine_function 1 16 \
			-end -clone_with_affine_function 1 32 -end\
		-set_builder -set_builder -loop 0 8 1 -affine_function 1 8 \
			-clone_with_affine_function 1 24 -end -clone_with_affine_function 1 32 -end\
		-set_builder -loop 0 16 1 -affine_function 1 16 -clone_with_affine_function 1 48 -end \
		-set_builder -loop 0 32 1 -affine_function 1 32 -end 
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file long_code_genma_n64_k7.csv \
		-box_width 25 -bit_depth 8  -partition 3 7 64 -end
	open long_code_genma_n64_k7_draw.bmp




#-nearest_codeword "8,16,32,24,40,48,56,1,2,4,3,5,6,7,9,18,36,27,45,54,63,33,42,20,11,53,62,31"

RM_6_GENERATOR_1="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,\
22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,\
46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63"
RM_6_GENERATOR_2="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,\
41,43,45,47,49,51,53,55,57,59,61,63"
RM_6_GENERATOR_3="2,3,6,7,18,19,22,23,10,11,14,15,26,27,30,31,34,35,38,\
39,42,43,46,47,50,51,54,55,58,59,62,63"
RM_6_GENERATOR_4="4,6,12,14,36,38,52,54,5,7,13,15,37,39,53,55,20,22,28,\
30,44,46,60,62,21,23,29,31,45,47,61,63"
RM_6_GENERATOR_5="8,9,12,13,24,25,28,29,10,11,14,15,26,27,30,31,40,41,\
44,45,56,57,60,61,42,43,46,47,58,59,62,63"
RM_6_GENERATOR_6="16,18,24,26,48,50,56,58,17,19,25,27,49,51,57,59,20,22,\
28,30,52,54,60,62,21,23,29,31,53,55,61,63"
RM_6_GENERATOR_7="32,34,48,50,33,35,49,51,36,38,52,54,37,39,53,55,40,42,\
56,58,41,43,57,59,44,46,60,62,45,47,61,63"

RM_6:
	$(ORBITER_PATH)orbiter.out -v 2 -long_code 64 7 \
		$(RM_6_GENERATOR_1) \
		$(RM_6_GENERATOR_2) \
		$(RM_6_GENERATOR_3) \
		$(RM_6_GENERATOR_4) \
		$(RM_6_GENERATOR_5) \
		$(RM_6_GENERATOR_6) \
		$(RM_6_GENERATOR_7) \
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file long_code_genma_n64_k7.csv \
		-box_width 25 -bit_depth 8  -partition 3 7 64 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file long_code_genma_n64_k7_codeword_0.csv \
		-box_width 25 -bit_depth 8  -partition 3 8 8 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file long_code_genma_n64_k7_codeword_1.csv \
		-box_width 25 -bit_depth 8  -partition 3 8 8 -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file long_code_genma_n64_k7_codeword_2.csv \
		-box_width 25 -bit_depth 8  -partition 3 8 8 -end




RM6words:
	- mkdir RM6
	#$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix -input_csv_file long_code_genma_n64_k7_codeword_0.csv -box_width 25 -bit_depth 8  -partition 4 8 8 -end
	$(ORBITER_PATH)orbiter.out -v 2 -loop L 0 128 1 \
		-draw_matrix -input_csv_file long_code_genma_n64_k7_codeword_%L.csv \
		-box_width 25 -bit_depth 8 -partition 4 8 8 -end \
		-mv long_code_genma_n64_k7_codeword_%L_draw.bmp RM6/RM_6_1_codeword_%L.bmp \
	-end_loop



RM6_convert:
	- mkdir RM6_PNG
	convert RM6/RM_6_1_codeword_0.bmp -frame 8 RM6_PNG/000.png
	convert RM6/RM_6_1_codeword_1.bmp -frame 8 RM6_PNG/001.png
	convert RM6/RM_6_1_codeword_2.bmp -frame 8 RM6_PNG/002.png
	convert RM6/RM_6_1_codeword_3.bmp -frame 8 RM6_PNG/003.png
	convert RM6/RM_6_1_codeword_4.bmp -frame 8 RM6_PNG/004.png
	convert RM6/RM_6_1_codeword_5.bmp -frame 8 RM6_PNG/005.png
	convert RM6/RM_6_1_codeword_6.bmp -frame 8 RM6_PNG/006.png
	convert RM6/RM_6_1_codeword_7.bmp -frame 8 RM6_PNG/007.png
	convert RM6/RM_6_1_codeword_8.bmp -frame 8 RM6_PNG/008.png
	convert RM6/RM_6_1_codeword_9.bmp -frame 8 RM6_PNG/009.png
	convert RM6/RM_6_1_codeword_10.bmp -frame 8 RM6_PNG/010.png
	convert RM6/RM_6_1_codeword_11.bmp -frame 8 RM6_PNG/011.png
	convert RM6/RM_6_1_codeword_12.bmp -frame 8 RM6_PNG/012.png
	convert RM6/RM_6_1_codeword_13.bmp -frame 8 RM6_PNG/013.png
	convert RM6/RM_6_1_codeword_14.bmp -frame 8 RM6_PNG/014.png
	convert RM6/RM_6_1_codeword_15.bmp -frame 8 RM6_PNG/015.png
	convert RM6/RM_6_1_codeword_16.bmp -frame 8 RM6_PNG/016.png
	convert RM6/RM_6_1_codeword_17.bmp -frame 8 RM6_PNG/017.png
	convert RM6/RM_6_1_codeword_18.bmp -frame 8 RM6_PNG/018.png
	convert RM6/RM_6_1_codeword_19.bmp -frame 8 RM6_PNG/019.png
	convert RM6/RM_6_1_codeword_20.bmp -frame 8 RM6_PNG/020.png
	convert RM6/RM_6_1_codeword_21.bmp -frame 8 RM6_PNG/021.png
	convert RM6/RM_6_1_codeword_22.bmp -frame 8 RM6_PNG/022.png
	convert RM6/RM_6_1_codeword_23.bmp -frame 8 RM6_PNG/023.png
	convert RM6/RM_6_1_codeword_24.bmp -frame 8 RM6_PNG/024.png
	convert RM6/RM_6_1_codeword_25.bmp -frame 8 RM6_PNG/025.png
	convert RM6/RM_6_1_codeword_26.bmp -frame 8 RM6_PNG/026.png
	convert RM6/RM_6_1_codeword_27.bmp -frame 8 RM6_PNG/027.png
	convert RM6/RM_6_1_codeword_28.bmp -frame 8 RM6_PNG/028.png
	convert RM6/RM_6_1_codeword_29.bmp -frame 8 RM6_PNG/029.png
	convert RM6/RM_6_1_codeword_30.bmp -frame 8 RM6_PNG/030.png
	convert RM6/RM_6_1_codeword_31.bmp -frame 8 RM6_PNG/031.png
	convert RM6/RM_6_1_codeword_32.bmp -frame 8 RM6_PNG/032.png
	convert RM6/RM_6_1_codeword_33.bmp -frame 8 RM6_PNG/033.png
	convert RM6/RM_6_1_codeword_34.bmp -frame 8 RM6_PNG/034.png
	convert RM6/RM_6_1_codeword_35.bmp -frame 8 RM6_PNG/035.png
	convert RM6/RM_6_1_codeword_36.bmp -frame 8 RM6_PNG/036.png
	convert RM6/RM_6_1_codeword_37.bmp -frame 8 RM6_PNG/037.png
	convert RM6/RM_6_1_codeword_38.bmp -frame 8 RM6_PNG/038.png
	convert RM6/RM_6_1_codeword_39.bmp -frame 8 RM6_PNG/039.png
	convert RM6/RM_6_1_codeword_40.bmp -frame 8 RM6_PNG/040.png
	convert RM6/RM_6_1_codeword_41.bmp -frame 8 RM6_PNG/041.png
	convert RM6/RM_6_1_codeword_42.bmp -frame 8 RM6_PNG/042.png
	convert RM6/RM_6_1_codeword_43.bmp -frame 8 RM6_PNG/043.png
	convert RM6/RM_6_1_codeword_44.bmp -frame 8 RM6_PNG/044.png
	convert RM6/RM_6_1_codeword_45.bmp -frame 8 RM6_PNG/045.png
	convert RM6/RM_6_1_codeword_46.bmp -frame 8 RM6_PNG/046.png
	convert RM6/RM_6_1_codeword_47.bmp -frame 8 RM6_PNG/047.png
	convert RM6/RM_6_1_codeword_48.bmp -frame 8 RM6_PNG/048.png
	convert RM6/RM_6_1_codeword_49.bmp -frame 8 RM6_PNG/049.png
	convert RM6/RM_6_1_codeword_50.bmp -frame 8 RM6_PNG/050.png
	convert RM6/RM_6_1_codeword_51.bmp -frame 8 RM6_PNG/051.png
	convert RM6/RM_6_1_codeword_52.bmp -frame 8 RM6_PNG/052.png
	convert RM6/RM_6_1_codeword_53.bmp -frame 8 RM6_PNG/053.png
	convert RM6/RM_6_1_codeword_54.bmp -frame 8 RM6_PNG/054.png
	convert RM6/RM_6_1_codeword_55.bmp -frame 8 RM6_PNG/055.png
	convert RM6/RM_6_1_codeword_56.bmp -frame 8 RM6_PNG/056.png
	convert RM6/RM_6_1_codeword_57.bmp -frame 8 RM6_PNG/057.png
	convert RM6/RM_6_1_codeword_58.bmp -frame 8 RM6_PNG/058.png
	convert RM6/RM_6_1_codeword_59.bmp -frame 8 RM6_PNG/059.png
	convert RM6/RM_6_1_codeword_60.bmp -frame 8 RM6_PNG/060.png
	convert RM6/RM_6_1_codeword_61.bmp -frame 8 RM6_PNG/061.png
	convert RM6/RM_6_1_codeword_62.bmp -frame 8 RM6_PNG/062.png
	convert RM6/RM_6_1_codeword_63.bmp -frame 8 RM6_PNG/063.png
	convert RM6/RM_6_1_codeword_64.bmp -frame 8 RM6_PNG/064.png
	convert RM6/RM_6_1_codeword_65.bmp -frame 8 RM6_PNG/065.png
	convert RM6/RM_6_1_codeword_66.bmp -frame 8 RM6_PNG/066.png
	convert RM6/RM_6_1_codeword_67.bmp -frame 8 RM6_PNG/067.png
	convert RM6/RM_6_1_codeword_68.bmp -frame 8 RM6_PNG/068.png
	convert RM6/RM_6_1_codeword_69.bmp -frame 8 RM6_PNG/069.png
	convert RM6/RM_6_1_codeword_70.bmp -frame 8 RM6_PNG/070.png
	convert RM6/RM_6_1_codeword_71.bmp -frame 8 RM6_PNG/071.png
	convert RM6/RM_6_1_codeword_72.bmp -frame 8 RM6_PNG/072.png
	convert RM6/RM_6_1_codeword_73.bmp -frame 8 RM6_PNG/073.png
	convert RM6/RM_6_1_codeword_74.bmp -frame 8 RM6_PNG/074.png
	convert RM6/RM_6_1_codeword_75.bmp -frame 8 RM6_PNG/075.png
	convert RM6/RM_6_1_codeword_76.bmp -frame 8 RM6_PNG/076.png
	convert RM6/RM_6_1_codeword_77.bmp -frame 8 RM6_PNG/077.png
	convert RM6/RM_6_1_codeword_78.bmp -frame 8 RM6_PNG/078.png
	convert RM6/RM_6_1_codeword_79.bmp -frame 8 RM6_PNG/079.png
	convert RM6/RM_6_1_codeword_80.bmp -frame 8 RM6_PNG/080.png
	convert RM6/RM_6_1_codeword_81.bmp -frame 8 RM6_PNG/081.png
	convert RM6/RM_6_1_codeword_82.bmp -frame 8 RM6_PNG/082.png
	convert RM6/RM_6_1_codeword_83.bmp -frame 8 RM6_PNG/083.png
	convert RM6/RM_6_1_codeword_84.bmp -frame 8 RM6_PNG/084.png
	convert RM6/RM_6_1_codeword_85.bmp -frame 8 RM6_PNG/085.png
	convert RM6/RM_6_1_codeword_86.bmp -frame 8 RM6_PNG/086.png
	convert RM6/RM_6_1_codeword_87.bmp -frame 8 RM6_PNG/087.png
	convert RM6/RM_6_1_codeword_88.bmp -frame 8 RM6_PNG/088.png
	convert RM6/RM_6_1_codeword_89.bmp -frame 8 RM6_PNG/089.png
	convert RM6/RM_6_1_codeword_90.bmp -frame 8 RM6_PNG/090.png
	convert RM6/RM_6_1_codeword_91.bmp -frame 8 RM6_PNG/091.png
	convert RM6/RM_6_1_codeword_92.bmp -frame 8 RM6_PNG/092.png
	convert RM6/RM_6_1_codeword_93.bmp -frame 8 RM6_PNG/093.png
	convert RM6/RM_6_1_codeword_94.bmp -frame 8 RM6_PNG/094.png
	convert RM6/RM_6_1_codeword_95.bmp -frame 8 RM6_PNG/095.png
	convert RM6/RM_6_1_codeword_96.bmp -frame 8 RM6_PNG/096.png
	convert RM6/RM_6_1_codeword_97.bmp -frame 8 RM6_PNG/097.png
	convert RM6/RM_6_1_codeword_98.bmp -frame 8 RM6_PNG/098.png
	convert RM6/RM_6_1_codeword_99.bmp -frame 8 RM6_PNG/099.png
	convert RM6/RM_6_1_codeword_100.bmp -frame 8 RM6_PNG/100.png
	convert RM6/RM_6_1_codeword_101.bmp -frame 8 RM6_PNG/101.png
	convert RM6/RM_6_1_codeword_102.bmp -frame 8 RM6_PNG/102.png
	convert RM6/RM_6_1_codeword_103.bmp -frame 8 RM6_PNG/103.png
	convert RM6/RM_6_1_codeword_104.bmp -frame 8 RM6_PNG/104.png
	convert RM6/RM_6_1_codeword_105.bmp -frame 8 RM6_PNG/105.png
	convert RM6/RM_6_1_codeword_106.bmp -frame 8 RM6_PNG/106.png
	convert RM6/RM_6_1_codeword_107.bmp -frame 8 RM6_PNG/107.png
	convert RM6/RM_6_1_codeword_108.bmp -frame 8 RM6_PNG/108.png
	convert RM6/RM_6_1_codeword_109.bmp -frame 8 RM6_PNG/109.png
	convert RM6/RM_6_1_codeword_110.bmp -frame 8 RM6_PNG/110.png
	convert RM6/RM_6_1_codeword_111.bmp -frame 8 RM6_PNG/111.png
	convert RM6/RM_6_1_codeword_112.bmp -frame 8 RM6_PNG/112.png
	convert RM6/RM_6_1_codeword_113.bmp -frame 8 RM6_PNG/113.png
	convert RM6/RM_6_1_codeword_114.bmp -frame 8 RM6_PNG/114.png
	convert RM6/RM_6_1_codeword_115.bmp -frame 8 RM6_PNG/115.png
	convert RM6/RM_6_1_codeword_116.bmp -frame 8 RM6_PNG/116.png
	convert RM6/RM_6_1_codeword_117.bmp -frame 8 RM6_PNG/117.png
	convert RM6/RM_6_1_codeword_118.bmp -frame 8 RM6_PNG/118.png
	convert RM6/RM_6_1_codeword_119.bmp -frame 8 RM6_PNG/119.png
	convert RM6/RM_6_1_codeword_120.bmp -frame 8 RM6_PNG/120.png
	convert RM6/RM_6_1_codeword_121.bmp -frame 8 RM6_PNG/121.png
	convert RM6/RM_6_1_codeword_122.bmp -frame 8 RM6_PNG/122.png
	convert RM6/RM_6_1_codeword_123.bmp -frame 8 RM6_PNG/123.png
	convert RM6/RM_6_1_codeword_124.bmp -frame 8 RM6_PNG/124.png
	convert RM6/RM_6_1_codeword_125.bmp -frame 8 RM6_PNG/125.png
	convert RM6/RM_6_1_codeword_126.bmp -frame 8 RM6_PNG/126.png
	convert RM6/RM_6_1_codeword_127.bmp -frame 8 RM6_PNG/127.png

RM_6_poster:
	convert RM6_PNG/000.png RM6_PNG/001.png RM6_PNG/002.png RM6_PNG/003.png RM6_PNG/004.png RM6_PNG/005.png RM6_PNG/006.png RM6_PNG/007.png RM6_PNG/008.png RM6_PNG/009.png RM6_PNG/010.png +append a0
	convert RM6_PNG/011.png RM6_PNG/012.png RM6_PNG/013.png RM6_PNG/014.png RM6_PNG/015.png RM6_PNG/016.png RM6_PNG/017.png RM6_PNG/018.png RM6_PNG/019.png RM6_PNG/020.png RM6_PNG/021.png +append a1
	convert RM6_PNG/022.png RM6_PNG/023.png RM6_PNG/024.png RM6_PNG/025.png RM6_PNG/026.png RM6_PNG/027.png RM6_PNG/028.png RM6_PNG/029.png RM6_PNG/030.png RM6_PNG/031.png RM6_PNG/032.png +append a2
	convert RM6_PNG/033.png RM6_PNG/034.png RM6_PNG/035.png RM6_PNG/036.png RM6_PNG/037.png RM6_PNG/038.png RM6_PNG/039.png RM6_PNG/040.png RM6_PNG/041.png RM6_PNG/042.png RM6_PNG/043.png +append a3
	convert RM6_PNG/044.png RM6_PNG/045.png RM6_PNG/046.png RM6_PNG/047.png RM6_PNG/048.png RM6_PNG/049.png RM6_PNG/050.png RM6_PNG/051.png RM6_PNG/052.png RM6_PNG/053.png RM6_PNG/054.png +append a4
	convert RM6_PNG/055.png RM6_PNG/056.png RM6_PNG/057.png RM6_PNG/058.png RM6_PNG/059.png RM6_PNG/060.png RM6_PNG/061.png RM6_PNG/062.png RM6_PNG/063.png RM6_PNG/064.png RM6_PNG/065.png +append a5
	convert RM6_PNG/066.png RM6_PNG/067.png RM6_PNG/068.png RM6_PNG/069.png RM6_PNG/070.png RM6_PNG/071.png RM6_PNG/072.png RM6_PNG/073.png RM6_PNG/074.png RM6_PNG/075.png RM6_PNG/076.png +append a6
	convert RM6_PNG/077.png RM6_PNG/078.png RM6_PNG/079.png RM6_PNG/080.png RM6_PNG/081.png RM6_PNG/082.png RM6_PNG/083.png RM6_PNG/084.png RM6_PNG/085.png RM6_PNG/086.png RM6_PNG/087.png +append a7
	convert RM6_PNG/088.png RM6_PNG/089.png RM6_PNG/090.png RM6_PNG/091.png RM6_PNG/092.png RM6_PNG/093.png RM6_PNG/094.png RM6_PNG/095.png RM6_PNG/096.png RM6_PNG/097.png RM6_PNG/098.png +append a8
	convert RM6_PNG/099.png RM6_PNG/100.png RM6_PNG/101.png RM6_PNG/102.png RM6_PNG/103.png RM6_PNG/104.png RM6_PNG/105.png RM6_PNG/106.png RM6_PNG/107.png RM6_PNG/108.png RM6_PNG/109.png +append a9
	convert RM6_PNG/110.png RM6_PNG/111.png RM6_PNG/112.png RM6_PNG/113.png RM6_PNG/114.png RM6_PNG/115.png RM6_PNG/116.png RM6_PNG/117.png RM6_PNG/118.png RM6_PNG/119.png RM6_PNG/120.png +append a10
	convert RM6_PNG/121.png RM6_PNG/122.png RM6_PNG/123.png RM6_PNG/124.png RM6_PNG/125.png RM6_PNG/126.png RM6_PNG/127.png RM6_PNG/000.png RM6_PNG/000.png RM6_PNG/000.png RM6_PNG/000.png +append a11
	convert a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 -append poster_RM_1_6.png




#############################################################################################################
# Section 9.6: Coding Theory - Reed Solomon codes

SECTION_CODING_THEORY_REED_SOLOMON_CODES:


RS_6_4_7_group:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end  \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_of_code "RS_6" 4 6 $(CODE_RS_6_4_7)  -end
	pdflatex code__classification.tex
	open code__classification.pdf
	
#group of order 12
#set of points:  0, 9, 51, 344, 253, 3
#1,0,0,0,2,0,0,1,5,0,6,0,5,1,0,0, 0,0,0,1,6,0,0,2,0,6,0,5,0,0,6,5,

RREF_RS_6_4_7_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 7 -end \
		-with F -do \
		-finite_field_activity \
		-weight_enumerator 4 6 $(CODE_RS_6_4_7)  -end




Code_RS_11:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-with F -do \
		-finite_field_activity -RREF 8 10 $(CODE_RS_10_8_11) -end
	pdflatex RREF_example_q11_8_10.tex
	#gs -sDEVICE=png16 -dFIXEDMEDIA -dDEVICEWIDTHPOINTS=500 -dDEVICEHEIGHTPOINTS=450 \
	#	-r240 -oRREF_example_q11_8_10_page%02d.png \
	#	RREF_example_q11_8_10.pdf
	open RREF_example_q11_8_10.pdf


Code_RS_11_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-with F -do \
		-finite_field_activity \
			-weight_enumerator 8 10 $(CODE_RS_11_RREF) -end


#1*y^(10) + 1200*x^3*y^7 + 16800*x^4*y^6 + 209160*x^5*y^5 + 1734600*x^6*y^4 + 9918000*x^7*y^3 + 37189800*x^8*y^2 + 82644700*x^9*y + 82644620*x^(10)


RREF_RS_8_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -end \
		-with F -do \
		-finite_field_activity \
			-weight_enumerator 5 7 $(CODE_RS_8) -end


RS_8_group:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -end  \
		-define P -projective_space 4 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_of_code "RS_8" 5 7 $(CODE_RS_8)  -end
	pdflatex code__classification.tex
	open code__classification.pdf

#set=( 0, 10, 59, 451, 3593, 1609, 4 )
# too big


RS_8_field_reduction:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 8 -end \
		-with F -do \
		-finite_field_activity \
		-field_reduction "RS_8_red_2" 2 5 7 $(CODE_RS_8) -end
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_matrix -input_csv_file RS_8_red_2.csv \
		-box_width 40 -bit_depth 24 \
		-partition  4 "3,3,3,3,3" "3,3,3,3,3,3,3"  -end
	pdflatex field_reduction_Q8_q2_5_7.tex
	open field_reduction_Q8_q2_5_7.pdf


RREF_RS_8_reduced_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -weight_enumerator 15 21 $(RS_8_reduced) -end



CODE_21_15_4_store:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-store_as_csv_file "code_21_15_4.csv" 15 21 $(CODE_21_15_4) 
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file code_21_15_4.csv \
		-box_width 40 -bit_depth 24  -partition  4 "15" "21"  -end

CODE_21_15_4_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -weight_enumerator 15 21 $(CODE_21_15_4) -end





	
Reed_solomon_F8_work:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 8 -end \
		-with F -do -finite_field_activity \
		-parse_and_evaluate "test" "" "(t-a)*(t-a*a)" "a=2" -end



#############################################################################################################
# Section 9.7: Coding Theory - BCH Codes

SECTION_CODING_THEORY_BCH_CODES:



BCH_255_5_evaluate_elementary_symmetric_functions_1:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 256 -end \
		-define e1 -formula "e1" "e1" "" $(ELEMENTARY_SYMMETRIC_8_1)    \
		-define e2 -formula "e2" "e2" "" $(ELEMENTARY_SYMMETRIC_8_2)    \
		-define e3 -formula "e3" "e3" "" $(ELEMENTARY_SYMMETRIC_8_3)    \
		-define e4 -formula "e4" "e4" "" $(ELEMENTARY_SYMMETRIC_8_4)    \
		-define e5 -formula "e5" "e5" "" $(ELEMENTARY_SYMMETRIC_8_5)    \
		-define e6 -formula "e6" "e6" "" $(ELEMENTARY_SYMMETRIC_8_6)    \
		-define e7 -formula "e7" "e7" "" $(ELEMENTARY_SYMMETRIC_8_7)    \
		-define e8 -formula "e8" "e8" "" $(ELEMENTARY_SYMMETRIC_8_8)    \
		-define E8 -collection "e1,e2,e3,e4,e5,e6,e7,e8" \
		-with F -do -finite_field_activity \
			-evaluate E8 "x0=2,x1=4,x2=16,x3=29,x4=76,x5=157,x6=95,x7=133" -end

BCH_255_5_evaluate_elementary_symmetric_functions_2:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 256 -end \
		-define e1 -formula "e1" "e1" "" $(ELEMENTARY_SYMMETRIC_8_1)    \
		-define e2 -formula "e2" "e2" "" $(ELEMENTARY_SYMMETRIC_8_2)    \
		-define e3 -formula "e3" "e3" "" $(ELEMENTARY_SYMMETRIC_8_3)    \
		-define e4 -formula "e4" "e4" "" $(ELEMENTARY_SYMMETRIC_8_4)    \
		-define e5 -formula "e5" "e5" "" $(ELEMENTARY_SYMMETRIC_8_5)    \
		-define e6 -formula "e6" "e6" "" $(ELEMENTARY_SYMMETRIC_8_6)    \
		-define e7 -formula "e7" "e7" "" $(ELEMENTARY_SYMMETRIC_8_7)    \
		-define e8 -formula "e8" "e8" "" $(ELEMENTARY_SYMMETRIC_8_8)    \
		-define E8 -collection "e1,e2,e3,e4,e5,e6,e7,e8" \
		-with F -do -finite_field_activity \
			-evaluate E8 "x0=8,x1=64,x2=205,x3=143,x4=70,x5=217,x6=130,x7=23" -end


BCH15:
	#$(ORBITER_PATH)orbiter.out -BCH 15 2 3 
	$(ORBITER_PATH)orbiter.out -BCH 15 2 5 
	#$(ORBITER_PATH)orbiter.out -BCH 15 2 7 
	#$(ORBITER_PATH)orbiter.out -BCH 15 2 9 


BCH11:
	$(ORBITER_PATH)orbiter.out -BCH 11 2 3 
	$(ORBITER_PATH)orbiter.out -BCH 11 2 5 


BCH13:
	$(ORBITER_PATH)orbiter.out -BCH 13 2 3 
	$(ORBITER_PATH)orbiter.out -BCH 13 2 5 


BCH7:
	$(ORBITER_PATH)orbiter.out -BCH 7 2 3 
	$(ORBITER_PATH)orbiter.out -BCH 7 2 5 


BCH21:
	$(ORBITER_PATH)orbiter.out -BCH 21 2 3 
	$(ORBITER_PATH)orbiter.out -BCH 21 2 5 
	$(ORBITER_PATH)orbiter.out -BCH 21 2 7 
	$(ORBITER_PATH)orbiter.out -BCH_dual 21 2 7 


BCH93:
	$(ORBITER_PATH)orbiter.out -BCH 93 2 3

BCH255:
	$(ORBITER_PATH)orbiter.out -BCH 255 2 4
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file BCH_255_4.csv \
		-box_width 40 -bit_depth 24  -partition 10 "239" "255"  -end

#BCH_255_4.csv


BCH273:
	$(ORBITER_PATH)orbiter.out -BCH 273 2 4





draw_mod_31:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -embedded -end \
		-draw_mod_n 31 mod_31 -draw_mod_n_power_cycle 2
	pdflatex mod_31_draw.tex
	open mod_31_draw.pdf


PR127:
	$(ORBITER_PATH)orbiter.out -v 5 -primitive_root 127


draw_mod_127_power:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -scale 0.4 -embedded -end \
		-draw_mod_n 127 mod_127 -draw_mod_n_power_cycle 3 
	pdflatex mod_127_draw.tex
	open mod_127_draw.pdf

draw_mod_251:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -nodes_empty -radius 10 -embedded -end \
		-draw_mod_n 251 mod_251 
	pdflatex mod_251_draw.tex
	open mod_251_draw.pdf

#-draw_mod_n_inverse


draw_mod_255_cyclotomic_1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -nodes_empty -radius 10 \
			-line_width 0.4 -embedded -end \
		-draw_mod_n -n 255 -file mod_255_cyclotomic_1 \
		-cyclotomic_sets 2 "1" -end
	pdflatex mod_255_cyclotomic_1_draw.tex
	open mod_255_cyclotomic_1_draw.pdf

draw_mod_255_cyclotomic_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -nodes_empty -radius 10 \
			-line_width 0.4 -embedded -end \
		-draw_mod_n -n 255 -file mod_255_cyclotomic_3 \
		-cyclotomic_sets 2 "3" -end 
	pdflatex mod_255_cyclotomic_3_draw.tex
	open mod_255_cyclotomic_3_draw.pdf

draw_mod_255_cyclotomic_1_and_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -nodes_empty -radius 10 \
			-line_width 0.4 -embedded -end \
		-draw_mod_n -n 255 -file mod_255_cyclotomic_1_and_3 \
		-cyclotomic_sets 2 "1,3" -end
	pdflatex mod_255_cyclotomic_1_and_3_draw.tex
	open mod_255_cyclotomic_1_and_3_draw.pdf

draw_mod_63_4_cyclotomic_3_6:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -radius 20 \
			-line_width 0.1 -embedded -end \
		-draw_mod_n -n 63 -file mod_63_4_cyclotomic_3_6 \
		-cyclotomic_sets 4 "3,6" \
		-cyclotomic_sets_thickness 50 \
		-end
	pdflatex mod_63_4_cyclotomic_3_6_draw.tex
	open mod_63_4_cyclotomic_3_6_draw.pdf

BCH_F_64:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 64 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	pdflatex GF_64.tex


BCH_elementary_symmetric_functions_3:
	$(ORBITER_PATH)orbiter.out -make_elementary_symmetric_functions 3 3


ELEMENTARY_SYMMETRIC_3_1="x0 + x1 + x2"
ELEMENTARY_SYMMETRIC_3_2="x0*x1 + x0*x2 + x1*x2"
ELEMENTARY_SYMMETRIC_3_3="x0*x1*x2"


ELEMENTARY_SYMMETRIC_4_1="x0 + x1 + x2 + x3"
ELEMENTARY_SYMMETRIC_4_2="x0*x1 + x0*x2 + x0*x3 + x1*x2 + x1*x3 + x2*x3"
ELEMENTARY_SYMMETRIC_4_3="x0*x1*x2 + x0*x1*x3 + x0*x2*x3 + x1*x2*x3"
ELEMENTARY_SYMMETRIC_4_4="x0*x1*x2*x3"

BCH_63_4_evaluate_elementary_symmetric_functions_1:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 64 -end \
		-define e1 -formula "e1" "e1" "" $(ELEMENTARY_SYMMETRIC_3_1)    \
		-define e2 -formula "e2" "e2" "" $(ELEMENTARY_SYMMETRIC_3_2)    \
		-define e3 -formula "e3" "e3" "" $(ELEMENTARY_SYMMETRIC_3_3)    \
		-define E3 -collection "e1,e2,e3" \
		-with F -do -finite_field_activity \
			-evaluate E3 "x0=8,x1=62,x2=15" -end


#The values of the formulae are:
#0 : 57
#1 : 0
#2 : 1


# poly: 1,0,2,1

BCH_63_4_evaluate_elementary_symmetric_functions_2:
	$(ORBITER_PATH)orbiter.out -v 3 -define F -finite_field -q 64 -end \
		-define e1 -formula "e1" "e1" "" $(ELEMENTARY_SYMMETRIC_3_1)    \
		-define e2 -formula "e2" "e2" "" $(ELEMENTARY_SYMMETRIC_3_2)    \
		-define e3 -formula "e3" "e3" "" $(ELEMENTARY_SYMMETRIC_3_3)    \
		-define E3 -collection "e1,e2,e3" \
		-with F -do -finite_field_activity \
			-evaluate E3 "x0=33,x1=45,x2=52" -end

#The values of the formulae are:
#0 : 56
#1 : 0
#2 : 1


# poly: 1,0,3,1



BCH_21_poly_mult_mod_F4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_mult_mod "1,0,2,1" "1,0,3,1" "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" -end

#C(X)=X^{6} + X^{5} + X^{4} + X^{2} + 1

# poly 1,0,1,0,1,1,1


BCH_21_poly_division_a:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" "1,0,2,1" -end

BCH_21_poly_division_b:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" "1,0,3,1" -end


BCH_21_poly_division_ab:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" "1,0,1,0,1,1,1" -end

BCH_21_generator_matrix:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-generator_matrix_cyclic_code 21 "1,0,1,0,1,1,1" -end


BCH_21_15_PROJ=" 0, 1, 19, 37, 113, 420, 1651, 6577, 26284, 105115, 420442, 1681753, 6727000, 26907991, 107631958, 27874647, 111498582, 43341143, 173364566, 156587350, 14 "


BCH_21_15="1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1"


BCH_21_15_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity -weight_enumerator 15 21 $(BCH_21_15) -end

# too slow!

BCH_21_15_dual:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do -finite_field_activity  \
		-nullspace 15 21 $(BCH_21_15)  \
		-normalize_from_the_right \
		-end


BCH_21_6=" 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1 "

BCH_21_6_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity -weight_enumerator 6 21 $(BCH_21_6) -end

# 1y^{21} + 63x^8y^{13} + 294x^{12}y^9 + 756x^{14}y^7 + 1890x^{16}y^5 + 1092x^{18}y^3

#( 1, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 294, 0, 756, 0, 1890, 0, 1092, 0, 0, 0 )



BCH_21_6_4_macwilliams:
	$(ORBITER_PATH)orbiter.out -v 2 -make_macwilliams_system 21 6 4
	pdflatex MacWilliams_n21_k6_q4.tex
	open MacWilliams_n21_k6_q4.pdf


#[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[63,59,55,51,47,43,39,35,31,27,23,19,15,11,7,3,-1,-5,-9,-13,-17,-21],[1890,1650,1426,1218,1026,850,690,546,418,306,210,130,66,18,-14,-30,-30,-14,18,66,130,210],[35910,29070,23142,18062,13766,10190,7270,4942,3142,1806,870,270,-58,-178,-154,-50,70,142,102,-114,-570,-1330],[484785,361665,263169,185841,126481,82145,50145,28049,13681,5121,705,-975,-1071,-479,161,465,305,-191,-639,-399,1425,5985],[4944807,3375027,2223855,1403163,837879,464963,232383,98091,28999,-45,-7281,-5061,-873,1635,1631,75,-1305,-1037,1071,2907,-969,-20349],[39558456,24488568,14441976,8008632,4091832,1859064,696952,170296,-14792,-43656,-21768,696,8376,4344,-2184,-4040,-200,4216,2040,-7752,-7752,54264],[254304360,141280200,73465704,34997832,14713704,5044680,1094760,-113080,-239000,-93240,15720,35400,9576,-11832,-9624,4680,9320,-3128,-13464,7752,38760,-116280],[1335097890,656952930,295275618,116816418,37677474,7829730,-499230,-1254750,-424670,94050,140130,18210,-45918,-19998,20706,16290,-16350,-16286,27234,17442,-106590,203490],[5785424190,2479467510,936687726,290962854,61114014,-52650,-6384690,-2121210,407550,550070,35630,-159130,-39586,66326,24206,-45370,-12610,52598,-12818,-92378,209950,-293930],[20827527084,7603700364,2314169676,504593388,30288492,-30667572,-11460852,1287468,2186028,128268,-528692,-86932,190956,31564,-107380,4524,87724,-57460,-71604,218348,-319124,352716],[62482581252,18843953076,4297743684,469793844,-125665020,-63950796,1099332,8697780,852228,-1718604,-260796,528692,42756,-242892,47684,141492,-126204,-41548,230724,-352716,386308,-352716],[156206453130,37192012650,5454828522,-252296694,-345095478,-27578070,33074730,5882058,-5372406,-1068822,1432170,106890,-550070,135850,235690,-236470,650,251498,-399126,428298,-377910,293930],[324428787270,56646296190,3089797974,-1608140466,-320557338,107272350,35626230,-15094674,-4859514,3719358,491670,-1261170,282150,424670,-418250,55470,289990,-465154,480726,-405042,300390,-203490],[556163635320,61795959480,-4119730632,-2385107208,184705272,183445560,-30705480,-21047688,8625528,2326968,-2867400,424440,839160,-717000,113080,364920,-560520,544952,-432072,302328,-193800,116280],[778629089448,37077575688,-12359191896,-1084139640,746851752,11809800,-79519320,14329224,9500328,-6106104,169128,1763208,-1178712,133128,510888,-696952,619688,-454648,296616,-178296,100776,-54264],[875957725629,-13904090883,-13904090883,1707519933,551104317,-231176835,-4428675,32102973,-10727235,-1909251,3689469,-1769283,3645,782973,-882819,697149,-464963,279293,-155907,82365,-41667,20349],[772903875555,-61341577425,-5725213893,3056317191,-304515693,-162089505,82373355,-9506889,-9428157,7026831,-2132325,-513945,1244403,-1108161,757323,-451305,246435,-126481,61947,-29241,13395,-5985],[515269250370,-73609892910,4907326194,1463588514,-679181598,111602610,26572050,-27280638,10510722,-1141614,-1771470,1902690,-1316574,763506,-400302,196290,-91710,41298,-18062,7714,-3230,1330],[244074908070,-50364663570,8523250758,-774840978,-200884698,143489070,-47829690,7440174,3188646,-3897234,2558790,-1377810,669222,-304722,132678,-55890,22950,-9234,3654,-1426,550,-210],[73222472421,-19758444939,5036466357,-1162261467,215233605,-14348907,-14348907,11160261,-5845851,2657205,-1121931,452709,-177147,67797,-25515,9477,-3483,1269,-459,165,-59,21],[10460353203,-3486784401,1162261467,-387420489,129140163,-43046721,14348907,-4782969,1594323,-531441,177147,-59049,19683,-6561,2187,-729,243,-81,27,-9,3,-1]]

#ww := [1, 0, 0, 84, 252, 1575, 10080, 58032, 319662, 1411116, 5133744, 15282792, 37951620, 79336530, 135622080, 190615824, 213273081, 188911548, 125744304, 59721732, 17767512, 2580255]


BCH_21_15_4_field_reduction:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 4 -end \
		-with F -do \
		-finite_field_activity \
		-field_reduction "BCH_21_15_4" 2 15 21 $(BCH_21_15) -end
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_matrix -input_csv_file BCH_21_15_4.csv \
		-box_width 20 -bit_depth 24 \
		-partition  4 "30" "42"  -end
	pdflatex field_reduction_Q4_q2_15_21.tex
	open BCH_21_15_4_draw.bmp
	#open field_reduction_Q4_q2_15_21.pdf

# poly of degree 12: 1,0,1,0,1,0,0,0,1,0,0,0,1

BCH_21_poly_division_c:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity \
		-polynomial_division "1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1" "1,0,1,0,1,0,0,0,1,0,0,0,1" -end



#############################################################################################################
# Section 9.8: Coding Theory - Bounds

SECTION_CODING_THEORY_BOUNDS:

bounds_for_d_given_n6_k4_q7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-make_bounds_for_d_given_n_and_k_and_q 6 4 7
	
bounds_for_d_given_n15_k6_q2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-make_bounds_for_d_given_n_and_k_and_q 15 6 2

#n = 15 k=6 q=2
#d_GV = 5
#d_singleton = 10
#d_hamming = 6
#d_plotkin = 7
#d_griesmer = 6


coding_theory_bounds_q2:
	$(ORBITER_PATH)orbiter.out -v 2 -table_of_bounds 20 2

# produces table_of_bounds_n20_q2.csv

coding_theory_bounds_q8:
	$(ORBITER_PATH)orbiter.out -v 2 -table_of_bounds 20 8

make_GV_n15_k6_d5:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end  \
		-define P -projective_space 8 F -end \
		-with P -do \
		-projective_space_activity -make_gilbert_varshamov_code 15 5 -end


# [15,6] code created

GV_n15_k6_d5_weight_enumerator:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -weight_enumerator 6 15 $(CODE_GV_N15_K6) -end

#1y^{15} + 27x^6y^9 + 24x^8y^7 + 9x^{10}y^5 + 3x^{12}y^3
# surprise: d = 6

GV_n15_k6_d5_group:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 5 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_of_code "GV_n15_k6_d5" 6 15 $(CODE_GV_N15_K6) \
		-end
	pdflatex classify_code_GV_n15_k6_d5_classification.tex
	open classify_code_GV_n15_k6_d5_classification.pdf

#ago=12



code_n15_k6_d6_a_group:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 2 -end \
		-define P -projective_space 5 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_of_code "n15_k6_d6_a" 6 15 $(CODE_15_6_6_A) \
		-end
	pdflatex classify_code_n15_k6_d6_a_classification.tex
	open classify_code_n15_k6_d6_a_classification.pdf

code_n15_k6_d6_a_we:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -weight_enumerator 6 15 $(CODE_15_6_6_A) -end

#1y^{15} + 27x^6y^9 + 24x^8y^7 + 9x^{10}y^5 + 3x^{12}y^3

code_n15_k6_d6_b_we:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do \
		-finite_field_activity -weight_enumerator 6 15 $(CODE_15_6_6_B) -end

#1y^{15} + 28x^6y^9 + 21x^8y^7 + 12x^{10}y^5 + 2x^{12}y^3


code_n15_k6_d6_RREF:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-RREF 6 15 $(CODE_GV_N15_K6)  -normalize_from_the_right \
		-end
	pdflatex RREF_example_q2_6_15.tex
	open RREF_example_q2_6_15.pdf

code_n15_k6_d6_check_RREF:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 2 -end \
		-with F -do -finite_field_activity \
		-RREF 9 15 $(CODE_GV_N15_K6_CHECK)  -normalize_from_the_right \
		-end
	pdflatex RREF_example_q2_9_15.tex
	open RREF_example_q2_9_15.pdf





#############################################################################################################
# Section 9.9: Coding Theory - Classification

SECTION_CODING_THEORY_CLASSIFICATION:



# code classification:

codes_8_4_4:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 4 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control \
			-problem_label codes_8_4_4 \
			-draw_poset \
			-draw_options -embedded -radius 250 \
			-line_width 1.0 -spanning_tree -end \
			-report -end \
		-end \
		-linear_codes 3 8 \
		-end
	pdflatex codes_8_4_4_poset_lvl_8.tex
	open codes_8_4_4_poset_lvl_8.pdf
	pdflatex codes_8_4_4_poset.tex
	open codes_8_4_4_poset.pdf

codes_8_4_4_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph codes_8_4_4_poset_lvl_8.layered_graph \
		-radius 250 -embedded -line_width 1.0 \
		-y_stretch 1.0 -scale 0.5 \
		-end
	pdflatex codes_8_4_4_poset_lvl_8_draw.tex
	open codes_8_4_4_poset_lvl_8_draw.pdf



codes_14_4_9_3:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G \
		-linear_group -PGL 10 3 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control \
			-problem_label codes_14_4_9_3 \
			-draw_poset -draw_options -embedded -radius 250 \
			-end \
		-end \
		-linear_codes 9 14 \
		-end
	pdflatex codes_14_4_9_3_poset_lvl_13.tex
	open codes_14_4_9_3_poset_lvl_13.pdf 


codes_15_6_6_2:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G \
		-linear_group -PGL 9 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control \
			-problem_label codes_15_6_6_2 \
			-draw_poset -draw_options -embedded -radius 250 \
			-end \
		-end \
		-linear_codes 6 15 \
		-end
	pdflatex codes_15_6_6_2_poset_lvl_15.tex
	open codes_15_6_6_2_poset_lvl_15.pdf 



codes_16_5_9_3:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-draw_options -end \
		-codes_classify -n 16 -k 5 -q 3 -d 9 -w -W -lex \
		-draw_poset \
		-end
		

# 5/31/2020: 28 min 22 sec on Mac

#0 : 1 orbits
#1 : 1 orbits
#2 : 1 orbits
#3 : 1 orbits
#4 : 1 orbits
#5 : 1 orbits
#6 : 1 orbits
#7 : 1 orbits
#8 : 1 orbits
#9 : 2 orbits
#10 : 3 orbits
#11 : 4 orbits
#12 : 5 orbits
#13 : 5 orbits
#14 : 4 orbits
#15 : 3 orbits
#16 : 1 orbits
#total: 36


codes_d4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_options -end \
		-define G -linear_group -PGL 4 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -W \
		-problem_label codes_r4_d4 -draw_poset \
		-embedded -end -linear_codes 4 100 \
		-end \
		-end


codes_24_12_8:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 12 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label codes_24_12_8 \
			-draw_poset \
			-draw_options -embedded -radius 250 \
			-line_width 1.0 -spanning_tree -end \
			-report -end \
		-end \
		-linear_codes 8 24 \
		-end
	pdflatex codes_24_12_8_poset.tex
	open codes_24_12_8_poset.pdf

#codes_24_12_8_poset_lvl_24.layered_graph

codes_24_12_8_draw:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-draw_layered_graph codes_24_12_8_poset_lvl_24.layered_graph \
		-radius 100 -spanning_tree -embedded \
		-line_width 0.5  -x_stretch 1.4 -scale 0.25 -nodes_empty \
		-end
	pdflatex codes_24_12_8_poset_lvl_24_draw.tex
	open codes_24_12_8_poset_lvl_24_draw.pdf

glynn_arc:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGGL 5 9 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label glynn_arc \
			-draw_options -embedded -radius 250 \
			-line_width 1.0 -spanning_tree -end \
			-draw_poset \
			-report -end \
		-end \
		-linear_codes 6 10 \
		-end
	pdflatex glynn_arc_poset.tex
	open glynn_arc_poset.pdf





five_points_in_general:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 4 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control \
			-problem_label five_points_in_general \
			-draw_options -embedded -radius 250 \
			-line_width 1.0 -spanning_tree -end \
			-draw_poset \
			-report -end \
		-end \
		-linear_codes 4 5 \
		-end
	pdflatex five_points_in_general_poset.tex
	open five_points_in_general_poset.pdf




codes_q13_12_4:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 4 13 -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control \
			-problem_label codes_q13 \
			-report -end \
		-end \
		-linear_codes 6 12 \
		-end
	pdflatex codes_q13_poset.tex
	open codes_q13_poset.pdf




#############################################################################################################
# Chapter 10 - Incidence Geometry
#############################################################################################################



########################################################################################################
# Section 10.1: Diophantine Systems

SECTION_DIOPHANT:



part10:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define D -diophant -label part10  \
			-coefficient_matrix 1 10 "10,9,8,7,6,5,4,3,2,1" \
			-RHS "10,10,1" -x_min_global 0 -x_max_global 10 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end
		


	
# Finds 42 solutions with 67 backtrack steps


octic_monomials:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define D -diophant -label octic_monomials  \
		-coefficient_matrix 1 4 "1,1,1,1" \
		-RHS "8,8,1" -x_min_global 0 -x_max_global 8 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end
	sort -r octic_monomials.sol >octic_monomials_sorted.txt

#Found 165 solutions with 210 backtrack steps
# 165=binomial(11,3)






########################################################################################################
# Section 10.2: Combinatorial Linear Spaces

SECTION_COMBINATORIAL_LINEAR_SPACES:
	

linsp7:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define D -diophant -label linsp7  \
		-coefficient_matrix 1 6 "21,15,10,6,3,1" \
		-RHS "21,21,1" \
		-x_min_global 0 \
		-x_max_global 21 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end
	

# 32 solutions in 45 backtrack steps


linsp6:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define D -diophant -label linsp6  \
		-coefficient_matrix 1 5 "15,10,6,3,1" \
		-RHS "15,15,1" \
		-x_min_global 0 \
		-x_max_global 15 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end
	
# Found 15 solutions with 22 backtrack steps





linsp30_pt_types:
	$(ORBITER_PATH)orbiter.out -v 4 -define D \
		-diophant -label linsp30_pt_types \
		-coefficient_matrix 1 3 "6,4,3" \
		-RHS "29,29,1" -x_bounds "0,1,0,27,0,24" \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end

linsp30_pt_distribution:
	$(ORBITER_PATH)orbiter.out -v 4 -define D \
		-diophant -label linsp30_pt_distribution  \
		-coefficient_matrix 6 4 \
			"1,1,1,1,1,1,0,0,5,2,5,2,1,5,3,7,10,1,10,1,0,10,3,21" \
			-RHS "30,30,1,7,7,1,135,135,1,96,96,1,0,351,2,0,276,2" \
			-x_min_global 0 -x_max_global 30 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end \
		-with D -do \
			-diophant_activity -draw_as_bitmap 20 8 \
		-end


#############################################################################################################
# Section 10.3: Design Theory

SECTION_DESIGN_THEORY:


design_PG_2_3:
	$(ORBITER_PATH)orbiter.out -v 8  \
		-create_design -q 3 -family PG_2_q  -end

design_PG_2_4:
	$(ORBITER_PATH)orbiter.out -v 8  \
		-create_design -q 4 -family PG_2_q  -end




ORDER_SYM_13="6227020800"
GENERATORS_SYM_13="1,0,2,3,4,5,6,7,8,9,10,11,12, 0,2,1,3,4,5,6,7,8,9,10,11,12, 0,1,3,2,4,5,6,7,8,9,10,11,12, 0,1,2,4,3,5,6,7,8,9,10,11,12, 0,1,2,3,5,4,6,7,8,9,10,11,12, 0,1,2,3,4,6,5,7,8,9,10,11,12, 0,1,2,3,4,5,7,6,8,9,10,11,12, 0,1,2,3,4,5,6,8,7,9,10,11,12, 0,1,2,3,4,5,6,7,9,8,10,11,12, 0,1,2,3,4,5,6,7,8,10,9,11,12, 0,1,2,3,4,5,6,7,8,9,11,10,12, 0,1,2,3,4,5,6,7,8,9,10,12,11"
GENERATORS_H5="1,2,3,4,0,6,7,8,9,5,10,11,12"
GENERATORS_N5="0,1,2,3,4,9,5,6,7,8,10,11,12,0,1,2,3,4,5,6,7,8,9,10,12,11,0,4,3,2,1,5,9,8,7,6,10,11,12,0,2,4,1,3,5,7,9,6,8,10,11,12,0,1,2,3,4,5,6,7,8,9,11,10,12,1,2,3,4,0,6,7,8,9,5,10,11,12,5,9,8,7,6,0,4,3,2,1,10,11,12"



design_PG_2_3_table_create_old:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define D -design -q 3 -family PG_2_q  -end \
		-with D -do \
		-design_activity \
			-create_table "PG_2_13" $(ORDER_SYM_13) $(GENERATORS_SYM_13) \
		-end
				 
	
# written file PG_2_13_design_table.csv
# 1108800 designs
#User time: 7:30


design_PG_2_3_table_create:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define D -design -q 3 -family PG_2_q  -end \
		-define T -design_table D "PG_2_13" $(ORDER_SYM_13) $(GENERATORS_SYM_13) 

# written file PG_2_13_design_table.csv
# 1108800 designs
#User time: 7:30

design_PG_2_3_group_5:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define D -design -q 3 -family PG_2_q  -end \
		-define T -design_table D "PG_2_13" $(ORDER_SYM_13) $(GENERATORS_SYM_13) \
		-define LSW -large_set_with_symmetry_assumption T \
				-H "5" $(GENERATORS_H5) \
				-N "1200" $(GENERATORS_N5) \
				-prefix "H5" \
				-selected_orbit_length 5 \
			-end \
		-with LSW -do \
			-large_set_with_symmetry_assumption_activity \
				-normalizer_on_orbits_of_a_given_length 5 \
			-end

#H5_N_orbit_reps.csv
# 678 orbits
#User time: 2:39


design_PG_2_3_group_5_sol_0:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define D -design -q 3 -family PG_2_q  -end \
		-define T -design_table D "PG_2_13" $(ORDER_SYM_13) $(GENERATORS_SYM_13) \
		-define LSW -large_set_with_symmetry_assumption T \
				-H "5" $(GENERATORS_H5) \
				-N "1200" $(GENERATORS_N5) \
				-prefix "H5" \
				-selected_orbit_length 5 \
			-end \
		-with LSW -do \
			-large_set_with_symmetry_assumption_activity \
				-read_solution_file 5 case_0_sol.txt \
			-end


# to create simple 7-designs on 33 points with block size 8 and lambda = 10 invariant under PGGL(2,32):

KM_PGGL_2_32:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define G -linear_group -PGGL 2 32 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-problem_label KM_PGGL_2_32 -W -depth 8 \
				-Kramer_Mesner_matrix 7 8 \
				-draw_poset  \
				-draw_options -embedded -sideways -radius 50 \
					-scale 0.5 -line_width 0.3 -end \
			-end \
			-orbits_on_subsets 8 \
		-end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file KM_PGGL_2_32_KM_7_8.csv \
		-box_width 20 -bit_depth 24 \
		-partition  3 32 97 -end
	pdflatex KM_PGGL_2_32_poset_lvl_8.tex
	open KM_PGGL_2_32_poset_lvl_8.pdf
	open KM_PGGL_2_32_KM_7_8_draw.bmp
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define D -diophant \
		-label "KM_PGGL_2_32_KM_7_8_system" \
		-coefficient_matrix_csv KM_PGGL_2_32_KM_7_8.csv  \
		-RHS_constant "10,10,1" \
		-x_min_global 0 -x_max_global 1 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end
		
		


KM_PSL_3_5:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define G -linear_group -PSL 3 5 -end \
		-with G -do \
			-group_theoretic_activities \
			-poset_classification_control \
				-problem_label KM_PSL_3_5 -W -depth 10 \
				-Kramer_Mesner_matrix 8 10 \
				-draw_poset \
				-draw_options -embedded -sideways \
					-radius 50 -scale 0.5 -line_width 0.3 -end \
			-end \
			-orbits_on_subsets 10
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file KM_PSL_3_5_KM_8_10.csv \
		-box_width 10 -bit_depth 8  -partition  3 42 174  -end
	$(ORBITER_PATH)orbiter.out -v 4 -define D -diophant \
		-label "KM_PSL_3_5_KM_8_10_system" \
			-coefficient_matrix_csv KM_PSL_3_5_KM_8_10.csv  \
		-RHS_constant "93,93,1" \
		-x_min_global 0 -x_max_global 1 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end





conf_10_3:
	$(ORBITER_PATH)orbiter.out -v 2 -regular_linear_space_classify \
		-m 10 -n 10 -r 3 -k 3 -target_size 10  \
		-control -problem_label 10_3 -W -end

conf_11_3:
	$(ORBITER_PATH)orbiter.out -v 2 -regular_linear_space_classify \
		-m 11 -n 11 -r 3 -k 3 -target_size 11  \
		-control -problem_label 11_3 -w -end

conf_12_3:
	$(ORBITER_PATH)orbiter.out -v 2 -regular_linear_space_classify \
		-m 12 -n 12 -r 3 -k 3 -target_size 12  \
		-control -problem_label 12_3 -w -end

conf_18_3:
	$(ORBITER_PATH)orbiter.out -v 2 -regular_linear_space_classify \
		-m 18 -n 18 -r 3 -k 3 -target_size 18  \
		-control -problem_label 18_3 -W -depth 8 -end


STS13:
	$(ORBITER_PATH)orbiter.out -v 2 -regular_linear_space_classify \
		-m 13 -n 26 -r 6 -k 3 -target_size 26  \
		-control -problem_label STS13 -w -depth 7 -end
	
PLANE_1="0,3,6,8,0,4,5,10,3,7,10,12,2,5,7,8,1,8,9,10,0,7,9,11,1,3,5,11,5,6,9,12,4,8,11,12,1,4,6,7,0,1,2,12,2,6,10,11,2,3,4,9"
PLANE_2="1,5,7,12,2,4,5,11,3,6,7,11,0,3,5,9,5,6,8,10,1,3,4,10,1,2,6,9,9,10,11,12,0,1,8,11,0,2,7,10,0,4,6,12,4,7,8,9,2,3,8,12"
PLANE_3="0,5,9,10,0,2,3,7,2,4,9,11,4,6,7,10,3,4,5,12,2,8,10,12,1,2,5,6,5,7,8,11,1,7,9,12,1,3,10,11,3,6,8,9,0,1,4,8,0,6,11,12"
PLANE_4="1,3,9,12,1,4,5,8,5,7,9,10,2,4,10,12,6,7,8,12,0,3,4,7,0,5,11,12,0,1,6,10,2,3,5,6,1,2,7,11,3,8,10,11,4,6,9,11,0,2,8,9"


#the sorted ranks of all subsets are: ( 9, 116, 140, 205, 278, 308, 375, 389, 455, 482, 577, 640, 656 )
#the sorted ranks of all subsets are: ( 47, 87, 111, 148, 246, 270, 339, 414, 426, 557, 625, 651, 714 )
#the sorted ranks of all subsets are: ( 22, 58, 179, 199, 237, 298, 371, 444, 499, 511, 559, 612, 662 )
#the sorted ranks of all subsets are: ( 37, 90, 102, 184, 253, 297, 303, 393, 447, 582, 620, 664, 683 )

#191704,9,116,140,205,278,308,375,389,455,482,577,640,656
#964499,47,87,111,148,246,270,339,414,426,557,625,651,714
#445252,22,58,179,199,237,298,371,444,499,511,559,612,662
#761894,37,90,102,184,253,297,303,393,447,582,620,664,683

#191704,445252,761894,964499


design_large_set_rank_k_subsets_a:
	$(ORBITER_PATH)orbiter.out -v 2 -rank_k_subset 13 4 $(PLANE_1)

design_large_set_rank_k_subsets_b:
	$(ORBITER_PATH)orbiter.out -v 2 -rank_k_subset 13 4 $(PLANE_2)

design_large_set_rank_k_subsets_c:
	$(ORBITER_PATH)orbiter.out -v 2 -rank_k_subset 13 4 $(PLANE_3)

design_large_set_rank_k_subsets_d:
	$(ORBITER_PATH)orbiter.out -v 2 -rank_k_subset 13 4 $(PLANE_4)

design_test:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define D -design  -end \
		-with D -do \
		-design_activity \
			-canonical_form \
			-input -file_of_incidence_geometries 24_3_TFC.inc 24 24 72 -end \
			-save_ago \
		-end


#############################################################################################################
# Section 10.4: Design Theory - Delandtsheer Doyen

SECTION_DESIGN_THEORY_DELANDTHSHEER_DOYEN:

PP4 = -d1 1 -q1 3 -d2 1 -q2 7 -K 5 -search_control -W -end -problem_label PP4
PP4_GROUP1=-subgroup "1,1,1,1, " "21" -group_label "cyclic21"
PP4_MASK1=\
	-nb_orbits_on_blocks 1 \
	-depth 5 \
	-mask_label "no_mask"

DD_PP4:
	$(ORBITER_PATH)orbiter.out -v 6  \
		-Delandtsheer_Doyen  $(PP4) $(PP4_GROUP1) $(PP4_MASK1) \
			-end \

DD_PP4_system:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define D -diophant -label PP4  \
		-problem_of_Steiner_type 10 PP4_pair_covering.csv \
		-has_sum 1 \
		-end \
		-with D -do \
			-diophant_activity -solve_mckay \
		-end





DD_CC:
	$(ORBITER_PATH)orbiter.out -v 6  \
		-Delandtsheer_Doyen  -search_wrt_subgroup \
			$(DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13) \
			$(DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13_GROUP1) \
			$(DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13_MASK1) \
		-end 

#target level: 6
#k2: 15
#number of k-orbits at target level: 1774964

# creates DD_CC_7_13_pair_covering.csv


DD_CC_system:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define D -diophant -label DD_CC_7_13  \
		-problem_of_Steiner_type 45 DD_CC_7_13_pair_covering.csv \
		-has_sum 3 \
		-end \
		-with D -do \
		-diophant_activity -solve_mckay \
		-end
	


# no solution



# 18603 = 27 * 53 * 13

DD_M1_G1:
	$(ORBITER_PATH)orbiter.out -v 4  \
		-Delandtsheer_Doyen  \
			$(DELANDTSHEER_DOYEN_PROBLEM_27_53) \
			$(DELANDTSHEER_DOYEN_PROBLEM_27_53_GROUP1) \
			$(DELANDTSHEER_DOYEN_PROBLEM_27_53_MASK1) \
			-end 

DD_M1_G1_S:
	$(ORBITER_PATH)orbiter.out -v 4  \
		-Delandtsheer_Doyen \
			$(DELANDTSHEER_DOYEN_PROBLEM_27_53) \
			$(DELANDTSHEER_DOYEN_PROBLEM_27_53_GROUP1) \
			$(DELANDTSHEER_DOYEN_PROBLEM_27_53_MASK1) \
			-singletons \
			-end 


DD_PG_2_4_M1_G1:
	$(ORBITER_PATH)orbiter.out -v 4  \
		-Delandtsheer_Doyen  \
			$(DELANDTSHEER_DOYEN_PROBLEM_3_7) \
			$(DELANDTSHEER_DOYEN_PROBLEM_3_7_GROUP1) \
			$(DELANDTSHEER_DOYEN_PROBLEM_3_7_MASK1) \
			-end 

PG_2_27_special:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 27 -override_polynomial 46 -end 
		-with F -do \
		-finite_field_activity \
		-cheat_sheet_PG 2 -end
	pdflatex PG_2_27.tex 
	open PG_2_27.pdf



design_27c:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define F -finite_field -q 27 -override_polynomial 46  -end \
		-define P -projective_space 2 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input \
		-set_of_points "2,56,30,112,253,90,440,508" \
		-end \
		-classification_prefix design \
		-report \
		-end
	pdflatex design_classification.tex
	open design_classification.pdf








#############################################################################################################
# Section 10.5: Tactical Decompositions

SECTION_TACTICAL_DECOMPOSITIONS:

max_arc_16_4_start:
	$(ORBITER_PATH)orbiter.out -v 4 -maximal_arc_parameters 16 4


max_arc_16_4_convert_stack_tdo:
	$(ORBITER_PATH)orbiter.out -v 4 -convert_stack_to_tdo max_arc_q16_r4.stack

max_arc_16_4_refine:
	$(ORBITER_PATH)orbiter.out -v 4 -tdo_refinement \
		-input_file max_arc_q16_r4.tdo -dual_is_linear_space -end
	
max_arc_16_4r_print:
	$(ORBITER_PATH)orbiter.out -v 4 -tdo_print max_arc_q16_r4r.tdo 



#############################################################################################################
# Section 10.6: Spreads

SECTION_SPREADS:
	
desarguesian_spread_in_PG_3_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define FQ -finite_field -q 4 -end \
		-define Fq -finite_field -q 2 -end \
		-with FQ -and Fq -do -finite_field_activity \
			-cheat_sheet_desarguesian_spread 2 -end
	pdflatex Desarguesian_Spread_3_2.tex
	open Desarguesian_Spread_3_2.pdf

desarguesian_spread_in_PG_5_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define FQ -finite_field -q 8 -end \
		-define Fq -finite_field -q 2 -end \
		-with FQ -and Fq -do -finite_field_activity \
			-cheat_sheet_desarguesian_spread 2 -end
	pdflatex Desarguesian_Spread_5_2.tex
	open Desarguesian_Spread_5_2.pdf

desarguesian_spread_in_PG_3_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define FQ -finite_field -q 16 -end \
		-define Fq -finite_field -q 4 -end \
		-with FQ -and Fq -do -finite_field_activity \
			-cheat_sheet_desarguesian_spread 2 -end
	pdflatex Desarguesian_Spread_3_4.tex
	open Desarguesian_Spread_3_4.pdf

desarguesian_spread_in_PG_3_5:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define FQ -finite_field -q 25 -end \
		-define Fq -finite_field -q 5 -end \
		-with FQ -and Fq -do -finite_field_activity \
			-cheat_sheet_desarguesian_spread 2 -end
	pdflatex Desarguesian_Spread_3_5.tex
	open Desarguesian_Spread_3_5.pdf


spreads4:
	- mkdir SPREADS_4
	- rm live_points.txt
	$(ORBITER_PATH)orbiter.out -v 10 \
		-define G \
		-linear_group -PGL 4 2 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_4 \
			-end \
			-spread_classify 2 \
		-end

spreads16_4:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-draw_options -radius 20 \
					-nodes_empty -line_width 0.2 -end \
				-draw_poset \
				-W \
				-problem_label spreads_16_4 \
				-report -end \
			-end \
			-spread_classify 2 \
		-end
	pdflatex spreads_16_4_poset.tex
	open spreads_16_4_poset.pdf

#17 : 3 orbits
#total: 1129
#(81600, 1200, 576) average is 27792 + 0 / 3

spreads25:
	$(ORBITER_PATH)orbiter.out -v 6 \
		-orbiter_path $(ORBITER_PATH) \
		-define G \
		-linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-draw_options -radius 20 \
					-nodes_empty -line_width 0.2 -end \
				-draw_poset \
				-W \
				-problem_label spreads_25 \
				-report -end \
			-end \
			-spread_classify 2 \
		-end
	pdflatex spreads_25_poset.tex
	open spreads_25_poset.pdf



spreads64_4:
	- mkdir SPREADS_64_4
	- rm live_points.txt
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G \
		-linear_group -PGGL 6 4 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_64_4 \
			-end \
			-spread_classify 3 \
		-end


# 853 orbits at level 5, each with about 60k live points

spreads64_8:
	- mkdir SPREADS_64_8
	- rm live_points.txt
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define G \
		-linear_group -PGGL 4 8 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_64_8 \
			-end \
			-spread_classify 2 \
		-end


# 16766 orbits at level 6, each with about 2k live points


spreads16:
	- mkdir SPREADS_16
	- rm live_points.txt
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G \
		-linear_group -PGL 8 2 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_16 \
			-end \
			-spread_classify 4 \
		-end

# 8 spreads of order 16
# May 14, 2020: 52 sec on Mac


line_spreads_PG5_2:
	- rm live_points.txt
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define G \
		-linear_group -PGL 6 2 -end \
		-with G -do \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label line_spreads_PG_5_2 \
				-depth 21 \
			-end \
			-spread_classify 2 \
		-end


# 16167 nodes at level 7
# 357991 nodes at levl 8 with about 150 live points each; 52371304 extension nodes


#############################################################################################################
# Section 10.7: Translation planes


SECTION_TRANSLATION_PLANES:



TP_9_0:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 3 -end \
		-define PGL4 -linear_group -PGL 4 F -end \
		-define PGL5 -linear_group -PGL 5 F -end \
		-with PGL4 -and PGL5 -do \
		-group_theoretic_activities \
		-Andre_Bruck_Bose_construction 0 "TP9-0" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file TP9-0_incma.csv \
		-box_width 6 -bit_depth 8  -partition 6 91 91  -end
	#open TP9-0_incma_draw.bmp

TP_9_1:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 3 -end \
		-define PGL4 -linear_group -PGL 4 F -end \
		-define PGL5 -linear_group -PGL 5 F -end \
		-with PGL4 -and PGL5 -do \
		-group_theoretic_activities \
		-Andre_Bruck_Bose_construction 0 "TP9-1" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file TP9-1_incma.csv \
		-box_width 6 -bit_depth 8  -partition 6 91 91  -end
	open TP9-1_incma_draw.bmp
	pdflatex TP9-1_report.tex
	open TP9-1_report.pdf


TP_16_4:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 4 -end \
		-define PGGL4 -linear_group -PGGL 4 F -end \
		-define PGGL5 -linear_group -PGGL 5 F -end \
		-with PGGL4 -and PGGL5 -do \
		-group_theoretic_activities \
		-Andre_Bruck_Bose_construction 0 "TP16-4-HALL" -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file TP16-4-HALL_incma.csv \
		-box_width 6 -bit_depth 8  -partition 6 273 273 -end
	open TP16-4-HALL_incma_draw.bmp
	pdflatex TP16-4-HALL_report.tex
	open TP16-4-HALL_report.pdf


#"1200", // Hall spread
#"81600", // Desarguesian spread
#"576", // Semifield spread



TP_16_2:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 2 -end \
		-define PGL8 -linear_group -PGL 8 F -end \
		-define PGL9 -linear_group -PGL 9 F -end \
		-with PGL8 -and PGL9 -do \
		-group_theoretic_activities \
			-Andre_Bruck_Bose_construction 0 "TP16_0_1008" -end


#"1008",
#"1008",
#"1728",
#"216",
#"360",
#"288",
#"3600",
#"244800",



#############################################################################################################
# Section 10.8: Packings


SECTION_PACKINGS:

spread_table_PG_3_4:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "0,1,2" "SPREAD_TABLES_4/" \


# 5096448 spreads
# 1020 self dual spreads
# User time: 56:38 on Mac

spread_table_PG_3_5_regular:
	- mkdir SPREAD_TABLES_5_REG
	$(ORBITER_PATH)orbiter.out -v 6 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-print_symbols


# 12/9/2020: 34 sec on Mac
# 12 is the index of the regular spread in the classification of spreads
# 155000 spreads


PG_3_5_desarguesian_spread:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define FQ -finite_field -q 25 -end \
		-define Fq -finite_field -q 5 -end \
		-with FQ -and Fq -do \
		-finite_field_activity \
			-cheat_sheet_desarguesian_spread 2 \
		-end
	pdflatex Desarguesian_Spread_3_5.tex
	open Desarguesian_Spread_3_5.pdf

#Spread elements by rank: (0, 805, 36, 108, 72, 144, 581, 509, 686, 415, 639, 758, 285, 722, 332, 343, 202, 592, 473, 238, 675, 379, 166, 545, 249, 451 )


PG_3_5_element_of_order_31:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
			-linear_group -GL 3 5 -end \
			-with G -do \
			-group_theoretic_activities \
			-raise_to_the_power "0,1,0, 0,0,1, 3,0,4" 31
		pdflatex GL_3_5_power.tex
		open GL_3_5_power.pdf

PG_3_5_element_of_order_31_normalizer:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "31" \
		"2,0,0,0, 0,0,1,0, 0,0,0,1, 0,3,0,4" -end
	mv normalizer_of_31_in_PGL_4_5.tex normalizer_of_31_AB_in_PGL_4_5.tex
	pdflatex normalizer_of_31_AB_in_PGL_4_5.tex
	open normalizer_of_31_AB_in_PGL_4_5.pdf



PG_3_5_element_of_order_31_GL_normalizer:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -GL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "124" \
		"2,0,0,0, 0,0,1,0, 0,0,0,1, 0,3,0,4" -end
	#mv normalizer_of_31_in_PGL_4_5.tex normalizer_of_31_AB_in_PGL_4_5.tex
	pdflatex normalizer_of_124_in_GL_4_5.tex
	open normalizer_of_124_in_GL_4_5.pdf



PG_3_5_element_of_order_31_ME_normalizer:
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGL 4 5 -end \
		-with G -do \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "31" \
		"1,0,0,0, 0,3,4,3, 0,3,3,4, 0,3,2,3" -end
	mv normalizer_of_31_in_PGL_4_5.tex normalizer_of_31_ME_in_PGL_4_5.tex
	pdflatex normalizer_of_31_ME_in_PGL_4_5.tex
	open normalizer_of_31_ME_in_PGL_4_5.pdf



# normalizer has order 1488=4*372=4*4*3*31



PG_3_5_assume_31_graph:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-define PW -packing_with_symmetry_assumption T \
				-H "H31" $(PGL_4_5_SUBGROUP_31_ME) -end \
				-N "N31" $(PGL_4_5_SUBGROUP_31_ME_NORMALIZER) -end \
			-end \
		-define PWF -packing_choose_fixed_points PW 0 -end \
		-define L -packing_long_orbits PWF \
			-orbit_length 31 -clique_size 1 -create_graphs -end \
		-print_symbols
	#pdflatex H31_spread_orbits_orbits_report.tex
	#open H31_spread_orbits_orbits_report.pdf
	pdflatex H31_reduced_spread_orbits_orbits_report.tex
	open H31_reduced_spread_orbits_orbits_report.pdf
	pdflatex H31_line_orbits_orbits_report.tex
	open H31_line_orbits_orbits_report.pdf
	pdflatex H31_line_orbits_orbits_report.tex
	open H31_line_orbits_orbits_report.pdf
	pdflatex N31_line_orbits_orbits_report.tex
	open N31_line_orbits_orbits_report.pdf
	pdflatex H31_point_orbits_orbits_report.tex
	open H31_point_orbits_orbits_report.pdf
	pdflatex N31_point_orbits_orbits_report.tex
	open N31_point_orbits_orbits_report.pdf

#H31_line_orbits_orbits.bin
#H31_line_orbits_orbits_report.tex
#H_spread_orbits_orbit_types_report.tex
#H31_spread_orbits_orbits.bin
#H31_good_orbits
#H31_spread_types_reduced_orbit_types_report.tex
#H31_reduced_spread_orbits_orbits.bin
#H31_fpc0_lo.graph


PG_3_5_assume_31_fpc0_lo_cliques:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -load_from_file H31_fpc0_lo.graph -end \
		-with G -do \
			-graph_theoretic_activity \
			-find_cliques  -target_size 1 -end -end \
		-print_symbols

#H31_fpc0_lo_sol.txt
#H31_fpc0_lo_sol.csv



# ToDo: problem when computing the orbits of the normalizer:

PG_3_5_assume_31_read:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-define PW -packing_with_symmetry_assumption T \
				-H "H31" $(PGL_4_5_SUBGROUP_31_ME) -end \
				-N "H31" $(PGL_4_5_SUBGROUP_31_ME) -end \
			-end \
		-define PWF -packing_choose_fixed_points PW 0 -end \
		-define L -packing_long_orbits PWF \
			-orbit_length 31 -clique_size 1 \
			-read_solutions \
		-end \

# writes H31_packings.csv


PG_3_5_assume_31_classify:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input -file_of_packings_through_spread_table \
					H31_packings.csv \
					SPREAD_TABLES_5_REG/spread_25_spreads.csv \
				-end \
			-classification_prefix H31_packings \
			-report H31_packings \
			-end
	pdflatex H31_packings_classification.tex
	open H31_packings_classification.pdf

# the two packings (ignore the first number)
#0,444,43313,154402,46682,108254,75363,27729,32139,5244,60442,142811,111115,94209,120678,89533,13798,103994,129953,82168,136838,19253,23017,145985,134996,54705,36267,55066,117542,96699,69154,72460
#1,616,42728,152655,48576,105431,79607,28634,32817,9799,62356,141176,110085,92557,122136,86312,13975,101942,126869,81478,139352,18028,24325,147284,130370,52074,36843,55602,118454,95973,69642,74036

PENTTILA_WILLIAMS_PRINCE_REG_PACKING_0="444,43313,154402,46682,108254,75363,27729,32139,5244,60442,142811,111115,94209,120678,89533,13798,103994,129953,82168,136838,19253,23017,145985,134996,54705,36267,55066,117542,96699,69154,72460"
PENTTILA_WILLIAMS_PRINCE_REG_PACKING_1="616,42728,152655,48576,105431,79607,28634,32817,9799,62356,141176,110085,92557,122136,86312,13975,101942,126869,81478,139352,18028,24325,147284,130370,52074,36843,55602,118454,95973,69642,74036"

PENTTILA_WILLIAMS_PRINCE_REG_PACKING_0_DUAL="3938, 66740, 56555, 93538, 107785, 64917, 47567, 54483, 141012, 138602, 18308, 6880, 131351, 88788, 125484, 102075, 21234, 99392, 119149, 80640, 124839, 148843, 71862, 11468, 35950, 27050, 75338, 113337, 40002, 154102, 30567"

PG_3_5_packing_0_dualize:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-with T -do \
			-spread_table_activity \
			-dualize_packing $(PENTTILA_WILLIAMS_PRINCE_REG_PACKING_0) \
		-end 



PG_3_5_assume_3:
	- rm N3B_ME_fixp_cliques.csv
	$(ORBITER_PATH)orbiter.out -v 20 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-define PW -packing_with_symmetry_assumption T \
				-H "H3B_ME" $(PGL_4_5_SUBGROUP_3B_ME) -end \
				-N "N3B_ME" $(PGL_4_5_SUBGROUP_3B_ME_NORMALIZER) -end \
			-end \
		-define PWF -packing_choose_fixed_points PW 1 \
			-W -problem_label N3B_ME_fixp_cliques \
			-preferred_choice 0 0 2 -end \
		-print_symbols

PG_3_5_3B_create_graph_on_long_orbits:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-define PW -packing_with_symmetry_assumption T \
				-H "H3B_ME" $(PGL_4_5_SUBGROUP_3B_ME) -end \
				-N "N3B_ME" $(PGL_4_5_SUBGROUP_3B_ME_NORMALIZER) -end \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			1 -W -problem_label N3B_ME_fixp_cliques -end \
		-define L -packing_long_orbits PWF \
			-orbit_length 3 \
			-clique_size 10 \
			-list_of_cases_from_file N3B_ME_fixp_cliques.csv \
			-create_graphs \
		-end \
		-print_symbols

# 16120 vertices
#creates H3B_ME_fpc0_lo.graph

PG_3_5_assume_3B_fpc0_lo_cliques:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -load_from_file H3B_ME_fpc0_lo.graph -end \
		-with G -do \
			-graph_theoretic_activity -find_cliques -target_size 10 -end -end \
		-print_symbols

# 768 solutions 
# User time: 8:16

PG_3_5_assume_3B_long_read:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-define PW -packing_with_symmetry_assumption T \
				-H "H3B_ME" $(PGL_4_5_SUBGROUP_3B_ME) -end \
				-N "N3B_ME" $(PGL_4_5_SUBGROUP_3B_ME_NORMALIZER) -end \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			1 -W -problem_label N3B_ME_fixp_cliques -end \
		-define L -packing_long_orbits PWF \
			-orbit_length 3 -clique_size 10 \
			-list_of_cases_from_file N3B_ME_fixp_cliques.csv \
			-read_solutions \
		-end \
		-print_symbols


#total number of packings = 768
#written file N3B_ME_fixp_cliques_count.csv of size 38
#packing_long_orbits::list_of_cases_from_file before save_packings_by_case
#packing_long_orbits::save_packings_by_case
#written file H3B_ME_packings.csv of size 150540



PG_3_5_assume_3B_classify:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
			-canonical_form_PG \
				-input -file_of_packings_through_spread_table \
					H3B_ME_packings.csv \
					SPREAD_TABLES_5_REG/spread_25_spreads.csv \
				-end \
				-classification_prefix H3B_packings \
				-save_classification cl_class \
				-save_canonical_labeling \
				-save_ago \
				-save_cumulative_canonical_labeling cl_cl \
				-save_cumulative_ago cl_ago \
				-save_cumulative_data cl_data \
				-save_fibration cl_fib \
				-report H3B_packings \
			-end
	pdflatex H3B_packings_classification.tex
	open H3B_packings_classification.pdf

PG_3_5_PACKING_0_WITH_AGO3="0,5201,60427,86602,11453,121452,46663,19716,32921,108680,23456,91963,68386,26921,74601,57067,36188,42312,78780,53117,118488,114700,83960,99669,104791,126662,130960,145179,137230,150626,140216"


PG_3_5_PACKING_0_WITH_AGO3_FIXP444="444,5001,12957,18194,23485,26817,34667,38299,41249,47472,50450,56601,62638,68986,71833,75369,80805,87025,92577,95676,104509,109718,114948,116333,124391,127498,133240,137711,144777,148059,150175"


PG_3_5_packing0_print:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-define PW -packing_with_symmetry_assumption T \
				-H "H31_ME" $(PGL_4_5_SUBGROUP_31_ME) -end \
				-N "N31_ME" $(PGL_4_5_SUBGROUP_31_ME_NORMALIZER) -end \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			0 -W -problem_label N3B_ME_fixp_cliques -end \
		-with PWF -do -packing_fixed_points_activity \
			-print_packing $(PG_3_5_PACKING_0_WITH_AGO3_FIXP444) \
		-end




# H in the action on point has 6 orbits of length 1 and 50 orbits of length 3
# the fixed points make up a line. It is the first of the two special lines.
# line orbits 2 and 3 make up the second of the special line.

# H in the action on lines has 270 orbits.
# There are 2 orbits of length 1 and 268 orbits of length 3
# the two orbits of length one are the special lines.
# the first line is fixed pointwise.


#spread orbits of length 1:\\
#Orbit 0:
#$$
#0=( 0, 36, 72, 108, 144, 157, 193, 229, 265, 301, 314, 350, 386, 422, 458, 466, 502, 538, 574, 610, 623, 659, 695, 731, 767, 805 )

ORBIT_OF_LENGTH_1="0, 36, 72, 108, 144, 157, 193, 229, 265, 301, 314, 350, 386, 422, 458, 466, 502, 538, 574, 610, 623, 659, 695, 731, 767, 805"

#		-define L -packing_long_orbits PWF -orbit_length 31 -clique_size 1 -create_graphs -end \



PG_3_5_H3_orbit_of_length_1:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-with T -do \
			-spread_table_activity \
			-find_spread_and_dualize $(ORBIT_OF_LENGTH_1) \
		-end 

#The given spread has index 0 in the spread table
#The dual spread has index 1 in the spread table


H31_ORBIT_OF_SPREAD_0="0, 44137, 153432, 45323, 109781, 77407, 29412, 32522, 6582, 61911, 144009, 112494, 91257, 123677, 88268, 13372, 100509, 125783, 80312, 135206, 17508, 22283, 146811, 132608, 53487, 36011, 55803, 116998, 99446, 69752, 73292"



#H31_packings.csv
#H3B_ME_packings.csv


PG_3_5_packings_compare:
	$(ORBITER_PATH)orbiter.out -v 5 \
		-define F -finite_field -q 5 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "12" "SPREAD_TABLES_5_REG/" \
		-define PW -packing_with_symmetry_assumption T \
				-H "H31_ME" $(PGL_4_5_SUBGROUP_31_ME) -end \
				-N "N31_ME" $(PGL_4_5_SUBGROUP_31_ME_NORMALIZER) -end \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			0 -W -problem_label N3B_ME_fixp_cliques -end \
		-with PWF -do -packing_fixed_points_activity \
			-compare_files_of_packings H31_packings.csv H3B_ME_packings.csv \
		-end

PG_3_5_packings_sort_each_row:
	$(ORBITER_PATH)orbiter.out -v 5 -csv_file_sort_each_row H31_packings.csv
	$(ORBITER_PATH)orbiter.out -v 5 -csv_file_sort_each_row H3B_ME_packings.csv

# H31_packings_sorted.csv



#############################################################################################################
# Section 10.9: BLT-sets

SECTION_BLT_SETS:

BLT_5_1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 5 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -catalogue 1 -end \
		-end
	pdflatex catalogue_q5_iso1.tex
	open catalogue_q5_iso1.pdf

BLT_5_Linear:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 5 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -family "Linear" -end \
		-end
	pdflatex BLT_Linear_q5.tex
	open BLT_Linear_q5.pdf

BLT_9_K1:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 9 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -family "K1" -end \
		-end
	pdflatex BLT_K1_q9.tex
	open BLT_K1_q9.pdf




BLT_11_0:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 11 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -catalogue 0 -end \
		-end
	#pdflatex O_1_6_2_report.tex 
	#open O_1_6_2_report.pdf


BLT_11_Fisher:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 11 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -family "Fisher" -end \
		-end
	pdflatex BLT_Fisher_q11.tex
	open BLT_Fisher_q11.pdf

BLT_11_Mondello:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define F -finite_field -q 11 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -family "Mondello" -end \
		-end
	pdflatex BLT_Mondello_q11.tex
	open BLT_Mondello_q11.pdf


BLT_13_FTWKB:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 11 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -family "FTWKB" -end \
		-end
	pdflatex BLT_FTWKB_q11.tex
	open BLT_FTWKB_q11.pdf


# for K2, q must be congruent to 2 or 3 mod 5
BLT_13_K2:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-create_BLT_set -family "Kantor2" -end \
		-end
	pdflatex BLT_K2_q13.tex
	open BLT_K2_q13.pdf


BLT_13_starter_5:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-BLT_set_starter 5 -problem_label BLT_q13 -W -depth 5 -end \
		-end

# BLT_13_lvl_5


BLT_13_deep_14:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-BLT_set_starter 14 -problem_label BLT_q13 -W -depth 14 -end \
		-end


BLT_13_graphs:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 13 -end \
		-define O -orthogonal_space 0 5 F -end \
		-with O -do -orthogonal_space_activity \
			-BLT_set_graphs 5 0 1 \
		-end


BLT_13_cliques:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-loop L 0 38 1 \
			-define G -graph -load_from_file BLT_q13_graph_5_%L.bin -end \
			-with G -do \
			-graph_theoretic_activity -find_cliques  -rainbow -target_size 9 -end -end \
		-end_loop




# old stuff, does not work any longer:

BLT_5_classify:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-orbiter_path $(ORBITER_PATH) \
		-define F -finite_field -q 5 -end \
		-define G -linear_group -PGO 5 F -end \
		-with G -do \
		-group_theoretic_activities \
		-poset_classification_control -problem_label BLT_5 -W -depth 6 \
			-draw_options -radius 250 -end \
			-draw_poset -report -end \
		-end \
		-BLT_starter 6 \
		-end
	#pdflatex PGO_5_5_report.tex
	#open PGO_5_5_report.pdf
	pdflatex BLT_5_poset.tex
	open BLT_5_poset.pdf



#############################################################################################################
# Chapter 11 - Graph Theory
#############################################################################################################



############################################################################################################
# Section 11.1: Creating Graphs


SECTION_CREATING_GRAPHS:
	


Cycle_13:
	$(ORBITER_PATH)orbiter.out -v 2 -define Gamma -graph -cycle 13 -end \
		-with Gamma -do \
		-graph_theoretic_activity -export_csv  -end \
		-with Gamma -do \
		-graph_theoretic_activity -automorphism_group -end \
		-with Gamma -do \
		-graph_theoretic_activity -export_graphviz -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Cycle_13.csv \
		-box_width 20 -bit_depth 8 -partition 4 13 13 -end
	#dot -Tpng Cycle_13.gv >Cycle_13.png
	#twopi -Tpng Cycle_13.gv >Cycle_13.png
	#open Cycle_13_draw.bmp
	#pdflatex Cycle_13_report.tex
	#open Cycle_13_report.pdf

draw_cycle_13:
		$(ORBITER_PATH)orbiter.out -v 2 -draw_options -embedded -end \
			-draw_mod_n 13 mod_13 
			pdflatex mod_13_draw.tex
			open mod_13_draw.pdf



Paley_13:
	$(ORBITER_PATH)orbiter.out -v 2 -define Gamma -graph -Paley 13 -end \
		-with Gamma -do \
		-graph_theoretic_activity -export_csv  -end \
		-with Gamma -do \
		-graph_theoretic_activity -automorphism_group -end \
		-with Gamma -do \
		-graph_theoretic_activity -export_graphviz -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Paley_13.csv \
		-box_width 20 -bit_depth 8 -partition 4 13 13 -end
	dot -Tpng Paley_13.gv >Paley_13.png
	open Paley_13_draw.bmp





	
trihedal_pair_graph:
	$(ORBITER_PATH)orbiter.out -v 2 -define Gamma \
		-graph -trihedral_pair_disjointness_graph  -end \
		-with Gamma -do \
		-graph_theoretic_activity -export_csv  -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file trihedral_pair_disjointness.csv \
		-box_width 20 -depth 8  -end
	open trihedral_pair_disjointness_draw.bmp





small_graph:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -edges_as_pairs 5 "0,1,0,2,0,3,0,4,1,3,1,4,2,4" -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file graph_v5_e7.csv \
		-box_width 40 -bit_depth 24 -partition 4 "1,1,1,1,1" "1,1,1,1,1" -end
	dot -Tpng graph_v5_e7.gv >graph_v5_e7.png

# creates graph_v5_e7.csv
# creates graph_v5_e7.colored_graph

small_graph_cliques:
	$(ORBITER_PATH)orbiter.out -v 10 \
		-define G -graph -load_from_file graph_v5_e7.colored_graph -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques -target_size 3 -end

# nb_sol = 3

small_graph_cliques_Sajeeb:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -load_from_file graph_v5_e7.colored_graph -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques -Sajeeb  -target_size 3 -end

# nb_sol = 3

petersen:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -Johnson 5 2 0 -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Johnson_5_2_0.csv \
		-box_width 40 -bit_depth 24 -partition 4 "10" "10" -end
	dot -Tpng Johnson_5_2_0.gv >Johnson_5_2_0.png

Johnson_6_2_0:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -Johnson 6 2 0 -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Johnson_6_2_0.csv \
		-box_width 40 -bit_depth 24 -partition 4 "10" "10" -end
	dot -Tpng Johnson_6_2_0.gv >Johnson_6_2_0.png

Hamming_graph_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -Hamming 3 2 -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Hamming_3_2.csv \
		-box_width 40 -bit_depth 24 \
		-partition 4 "1,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,1" -end
	dot -Tpng Hamming_3_2.gv >Hamming_3_2.png


Hamming_graph_7:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -Hamming 7 2 -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end
	$(ORBITER_PATH)orbiter.out -v 2 -draw_matrix \
		-input_csv_file Hamming_7_2.csv \
		-box_width 8 -bit_depth 24 -partition 4 128 128 -end
	dot -Tpng Hamming_7_2.gv >Hamming_7_2.png



Hamming_graph_7_with_Hamming_code:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define G -graph -Hamming 7 2 \
			-subset "_Hamming_code" "\\_with\\_Hamming\\_code" \
			$(HAMMING_CODE_CODEWORDS) -end \
		-with G -do \
		-graph_theoretic_activity -export_csv -end \
		-with G -do \
		-graph_theoretic_activity -export_graphviz -end \
		-with G -do \
		-graph_theoretic_activity -save -end \
		-with G -do \
		-graph_theoretic_activity -automorphism_group -end
	pdflatex Hamming_7_2_Hamming_code_report.tex
	open Hamming_7_2_Hamming_code_report.pdf

# group of order 2688 = 16 * 168




############################################################################################################
# Section 11.2: Graph Theory: Classification


SECTION_GRAPH_THEORY_CLASSIFICATION:
	

graph_classify_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define GC -graph_classification \
			-n 4  \
			-poset_classification_control \
				-problem_label graphs_v4  -depth 6 -draw_poset \
				-draw_options -radius 250 -embedded -end \
			-end \
		-end \
		-with GC -do \
		-graph_classification_activity \
			-draw_options -embedded -radius 400 \
				-line_width 2 -scale 0.15 -end \
			-draw_graphs_at_level 3 \
		-end \
		-print_symbols
	pdflatex graphs_v4_rep_3_2.tex
	open graphs_v4_rep_3_2.pdf
	#pdflatex graphs_v4_poset_detailed_lvl_6.tex
	#open graphs_v4_poset_detailed_lvl_6.pdf
	#pdflatex graphs_v4_poset_lvl_6.tex
	#open graphs_v4_poset_lvl_6.pdf


tournament_classify_4:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define GC -graph_classification \
			-n 4 -tournament \
			-poset_classification_control \
				-problem_label tournament_4  -depth 6 -draw_poset \
				-draw_options -radius 250 -embedded -end \
			-end \
		-end \
		-with GC -do \
		-graph_classification_activity \
			-draw_options -embedded -radius 400 \
				-line_width 2 -scale 0.15 -end \
			-draw_graphs_at_level 6 \
		-end \
		-print_symbols
	pdflatex tournament_4_rep_6_0.tex
	pdflatex tournament_4_rep_6_1.tex
	pdflatex tournament_4_rep_6_2.tex
	pdflatex tournament_4_rep_6_3.tex
	open tournament_4_rep_6_0.pdf
	open tournament_4_rep_6_1.pdf
	open tournament_4_rep_6_2.pdf
	open tournament_4_rep_6_3.pdf
	


graph_classify_8_r3:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define GC -graph_classification \
			-n 8 -regular 3 \
			-poset_classification_control \
				-problem_label graphs_v8_r3  -depth 12 -draw_poset \
				-draw_options -radius 250 \
					-line_width 0.2 -embedded -end \
			-end \
		-end \
		-with GC -do \
		-graph_classification_activity \
			-draw_options -embedded -radius 400 \
				-line_width 2 -end \
			-draw_graphs_at_level 12 \
		-end \
		-print_symbols
	pdflatex graphs_v8_r3_poset_lvl_12.tex
	open graphs_v8_r3_poset_lvl_12.pdf
	#pdflatex graphs_v8_r3_rep_12_0.tex
	#open graphs_v8_r3_rep_12_0.pdf
	#pdflatex graphs_v8_r3_rep_12_1.tex
	#open graphs_v8_r3_rep_12_1.pdf
	#pdflatex graphs_v8_r3_rep_12_2.tex
	#open graphs_v8_r3_rep_12_2.pdf
	#pdflatex graphs_v8_r3_rep_12_3.tex
	#open graphs_v8_r3_rep_12_3.pdf
	#pdflatex graphs_v8_r3_rep_12_4.tex
	#open graphs_v8_r3_rep_12_4.pdf
	#pdflatex graphs_v8_r3_rep_12_5.tex
	#open graphs_v8_r3_rep_12_5.pdf


#############################################################################################################
# Chapter 12 - Interfaces
#############################################################################################################


#############################################################################################################
# Section 12.1: Graphical Output

SECTION_GRAPHICAL_OUTPUT:
	


PGL_4_2_Wedge_4_0_detached_graphical_output:
	$(ORBITER_PATH)orbiter.out -v 12 \
		-define G -linear_group -PGL 4 2 \
		-wedge_detached -end \
		-with G -do \
		-group_theoretic_activities \
		-report \
		-draw_options -radius 200 -end \
		-end
	pdflatex PGL_4_2_Wedge_4_0_detached_report.tex 
	open PGL_4_2_Wedge_4_0_detached_report.pdf


schreier_tree_graphical_output:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-define G -linear_group -PGL 4 2 -end \
		-with G -do \
		-group_theoretic_activities \
		-orbits_on_polynomials 3 \
		-orbits_on_polynomials_draw_tree 6 \
		-draw_options -yout 500000 -radius 15 -nodes_empty \
			-line_width 0.5 -y_stretch 0.25 -embedded -end \
		-end 
	pdflatex poly_orbits_d3_n3_q2.tex
	open poly_orbits_d3_n3_q2.pdf




#############################################################################################################
# Section 12.2: The Povray Interface


SECTION_POVRAY:


cube:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask cube_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.5 \
		-default_angle 75 \
		-clipping_radius 2.7 \
		-end \
		-scene_objects \
			-obj_file cube_centered.obj \
			-edge "0, 1" \
			-edge "0, 2" \
			-edge "0, 4" \
			-edge "1, 3" \
			-edge "1, 5" \
			-edge "2, 3" \
			-edge "2, 6" \
			-edge "3, 7" \
			-edge "4, 5" \
			-edge "4, 6" \
			-edge "5, 7" \
			-edge "6, 7" \
			-group_of_things_as_interval 0 8 \
			-spheres 0 0.3 "texture{ Polished_Chrome pigment{quick_color White} }" \
			-group_of_things_as_interval 0 6 \
			-prisms 1 0.05 "texture{ pigment{ color Yellow transmit 0.7 } finish {diffuse 0.9 phong 0.6} }" \
			-group_of_things_as_interval 0 12 \
			-cylinders 2 0.15 "texture{ pigment{ color Red  } finish {diffuse 0.9 phong 0.6} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv cube_0_*.pov POV
	mv makefile_animation POV
	
monkey:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask monkey_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.8 \
		-default_angle 75 \
		-clipping_radius 0.8 \
		-camera 0 "0,0,1" "1,1,0.5" "0,0,0"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-cubic_lex "1,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ Gold } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-planes 1 "texture{ pigment{ color Blue transmit 0.5 } finish { diffuse 0.9 phong 0.2}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv monkey_0_*.pov POV
	mv makefile_animation POV





Eckardt:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Eckardt_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.9 \
		-default_angle 75 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-Hilbert_Cohn_Vossen_surface \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things_as_interval 0 6 \
			-group_of_things_as_interval 6 6 \
			-group_of_things_as_interval_with_exceptions 12 15 "14,19,23" \
			-lines 1 0.02 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.02 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.02 "texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}" \
			-label 0 "a1" \
			-label 2 "a2" \
			-label 4 "a3" \
			-label 6 "a4" \
			-label 8 "a5" \
			-label 10 "a6" \
			-label 12 "b1" \
			-label 14 "b2" \
			-label 16 "b3" \
			-label 18 "b4" \
			-label 20 "b5" \
			-label 22 "b6" \
			-label 24 "c12" \
			-label 26 "c13" \
			-label 30 "c15" \
			-label 32 "c16" \
			-label 34 "c23" \
			-label 36 "c24" \
			-label 40 "c26" \
			-label 42 "c34" \
			-label 44 "c35" \
			-label 48 "c45" \
			-label 50 "c46" \
			-label 52 "c56" \
			-group_of_things_as_interval 0 6 \
			-texts 4 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
			-group_of_things_as_interval 6 6 \
			-texts 5 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
			-group_of_things_as_interval 12 12 \
			-texts 6 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Eckardt_0_*.pov POV
	mv makefile_animation POV

#"-3,2.333,4"  * 1.5 = "-4.5,3.5,6"
#M := Matrix([[-4.5, 3.5, 6], [1, 1, 1]])
#NullSpace(M)
#=0.186080731891197,-0.781539073943026,0.595458342051830
#		-rotate_about_custom_axis "0.186080731891197,-0.781539073943026,0.595458342051830" \
#-W 1024 -H 768 
#-W 2560 -H 1920 
#-W 4096 -H 3072 


Eckardt_deform:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 93 \
		-output_mask Eckardt_deform_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.9 \
		-default_angle 75 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-Hilbert_Cohn_Vossen_surface \
			-group_of_things "0" \
			-deformation_of_cubic_lex 93 1.107148718 1.570796327 0 \
				"0, 10, 0, -8, 10, 25, 2, 0, -20, -8, -20, -10, -24, 10, -2, 12, 0, -8, 8, 16" \
				"0, -5, 0, -5, -5, 10, -1, 0, 10, 4, 10, 5, 3, -5, 1, -6, 0, -5, -4, 1" \
			-group_of_things_as_interval 0 93 \
			-group_is_animated 1 \
			-cubics 1 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Eckardt_deform_0_*.pov POV
	mv makefile_animation POV





Eckardt_deform_2:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Eckardt_deform_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.9 \
		-default_angle 75 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-Hilbert_Cohn_Vossen_surface \
			-group_of_things "0" \
			-deformation_of_cubic_lex 93 1.107148718 1.570796327 0 \
				"0, 10, 0, -8, 10, 25, 2, 0, -20, -8, -20, -10, -24, 10, -2, 12, 0, -8, 8, 16" \
				"0, -5, 0, -5, -5, 10, -1, 0, 10, 4, 10, 5, 3, -5, 1, -6, 0, -5, -4, 1" \
			--group_of_things_as_interval 0 93 \
			--group_is_animated 1 \
			-group_of_things "0" \
			-cubics 1 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things "24" \
			-cubics 2 "texture{ pigment{ Red*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things "70" \
			-cubics 3 "texture{ pigment{ Blue*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Eckardt_deform_0_*.pov POV
	mv makefile_animation POV




	


Clebsch:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Clebsch_%d_%03d.pov \
		-video_options -W 1024 -H 768  \
		-global_picture_scale 0.9 \
		-default_angle 80 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1"  "-4.5,3.5,6" "0,0,0"  \
		-end \
		-scene_objects \
			-Clebsch_surface \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things_as_interval 0 6 \
			-group_of_things_as_interval 6 6 \
			-group_of_things_as_interval 12 15  \
			-lines 1 0.02 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.02 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.02 "texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}" \
			-group_of_things_as_interval 0 12 \
			-spheres 4 0.08 "texture{ pigment{Cyan*1.3} finish {ambient 0.4 diffuse 0.6 roughness 0.001 reflection 0 specular .8}  }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Clebsch_0_*.pov POV
	mv makefile_animation POV




endrass8:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask endrass_octic_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.75 \
		-default_angle 75 \
		-clipping_radius 3.7 \
		-no_bottom_plane \
		-camera 0 "1,1,1" "6,6,3" "0,0,0"  \
		-rotate_about_111 \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-group_of_things_as_interval 3 39 \
			-lines 0 0.15 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 1 0.15 "texture{ pigment{ color Green } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.15 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.05 "texture{ pigment{ color Black } finish { diffuse 0.9 phong 1}}" \
			-octic_lex_165 "-93.2548,0,0,0,-309.019,0,0,527.529,0,395.647,0,0,0,0,0,0,0,0,0,0,-687.529,0,0,1582.59,0,1186.94,0,0,0,0,-1055.06,0,-1582.59,0,-593.47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-309.019,0,0,1582.59,0,1186.94,0,0,0,0,-2110.12,0,-3165.17,0,-1186.94,0,0,0,0,0,0,874.039,0,1560.63,0,1677.92,0,343.362,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-93.2548,0,0,527.529,0,395.647,0,0,0,0,-1055.06,0,-1582.59,0,-593.47,0,0,0,0,0,0,874.039,0,1560.63,0,1677.92,0,343.362,0,0,0,0,0,0,0,0,-256,0,-468.077,0,-789.019,0,-525.726,0,0.941125" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-group_of_things "0" \
			-octics 4 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-planes 5 "texture{ pigment{ color Blue transmit 0.5 } finish { diffuse 0.9 phong 1}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv endrass_octic_0_*.pov POV
	mv makefile_animation POV





#############################################################################################################
# Section 12.3: Creating Animations

SECTION_ANIMATIONS:


monkey_video:
	- rm -r FRAMES
	- mkdir FRAMES
	- rm monkey.mp4
	$(ORBITER_PATH)orbiter.out \
		-prepare_frames \
			-i 0 30 monkey_0_%03d.png \
			-output_starts_at 0 \
			-o FRAMES/frame%04d.png \
		-end
	ffmpeg -r 5 -f image2 -i FRAMES/frame%04d.png \
		-f mp4 -q:v 0 -vcodec mpeg4  monkey.mp4

Eckardt_deform_video:
	- rm -r FRAMES
	- mkdir FRAMES
	- rm Eckardt_deform.mp4
	$(ORBITER_PATH)orbiter.out \
		-prepare_frames \
			-i 0 93 Eckardt_deform_0/Eckardt_deform_0_%03d.png \
			-output_starts_at 0 \
			-o FRAMES/frame%04d.png \
		-end
	ffmpeg -r 5 -f image2 -i FRAMES/frame%04d.png \
		-f mp4 -q:v 0 -vcodec mpeg4  Eckardt_deform.mp4


Eckardt_surface:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Eckardt_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.9 \
		-default_angle 75 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-cubic_Goursat "6,3,-15" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Eckardt_0_*.pov POV
	mv makefile_animation POV

Kummer_surface:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Kummer_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.9 \
		-default_angle 75 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-quartic_lex_35 "-2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 2, 0, 2, 0, 0, 0, 0, -2, 0, 2, 0, -2" \
			-group_of_things "0" \
			-quartics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Kummer_0_*.pov POV
	mv makefile_animation POV


# Maple:
#Kummer := expand((x0^2 + x1^2 + x2^2 + x3^2)^2 - 3*(x0^4 + x1^4 + x2^4 + x3^4))


Kummer_video:
	- rm -r FRAMES
	- mkdir FRAMES
	- rm Kummer.mp4
	$(ORBITER_PATH)orbiter.out \
		-prepare_frames \
			-i 0 30 KUMMER/Kummer_0_%03d.png \
			-output_starts_at 0 \
			-o FRAMES/frame%04d.png \
		-end
	ffmpeg -r 5 -f image2 -i FRAMES/frame%04d.png \
		-f mp4 -q:v 0 -vcodec mpeg4  Kummer.mp4


Beauville_surface:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Beauville_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.3 \
		-default_angle 75 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-quintic_lex_56 "-44, 228, 400, 315, -396, -852, -512, -553, -1050, -354, 284, 504, -62, -707, -1390, -1010, 281, -167, -1644, -1024, -72, -196, 192, 373, 322, 78, 150, 966, 1540, 348, -475, -492, 1063, 1550, 390, 0, 96, 3, -337, -426, -66, 425, 673, -156, -216, -223, -60, 1543, 1998, 618, 263, -250, -919, 557, 1800, 741" \
			-group_of_things "0" \
			-quintics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Beauville_0_*.pov POV
	mv makefile_animation POV


#-44, 228, 400, 315, -396, -852, -512, -553, -1050, -354, 284, 504, -62, -707, -1390, -1010, 281, -167, -1644, -1024, -72, -196, 192, 373, 322, 78, 150, 966, 1540, 348, -475, -492, 1063, 1550, 390, 0, 96, 3, -337, -426, -66, 425, 673, -156, -216, -223, -60, 1543, 1998, 618, 263, -250, -919, 557, 1800, 741





# Clebsch map up for surface created using arc lifting
# We take a circle of radius r centered at the origin in the affine real plane 
# and map it up on the surface.
# The Clebsch surface has 
# a = d = 2.618033988 = (3+sqrt(5))/2
# b = c = 1.618033988 = (1+sqrt(5))/2
# 
CLEBSCH_A=2.618033988
CLEBSCH_D=2.618033988
CLEBSCH_B=1.618033988
CLEBSCH_C=1.618033988
TWO_PI=6.283185308


# to go from the arclifting surface to the defining equation:
#Matrix(4, 4, [[-0.44721360215312733, 1.1708204000530853, 1.1708204000530853, -0.4472135957999158], [-1.1708204000530853, 0.4472136021531272, 1.4472136021531272, 0.4472135957999158], [4.2360680044124255, -4.2360680044124255, -4.2360680044124255, 0.], [1.6180340022062127, -2.6180340022062127, -1.6180340022062127, 0.]])
#-0.44721360215312733, 1.1708204000530853, 1.1708204000530853, -0.4472135957999158
#-1.1708204000530853, 0.4472136021531272, 1.4472136021531272, 0.4472135957999158
#4.2360680044124255, -4.2360680044124255, -4.2360680044124255, 0.
#1.6180340022062127, -2.6180340022062127, -1.6180340022062127, 0.


T00=-0.44721360215312733
T01=1.1708204000530853
T02=1.1708204000530853
T03=-0.4472135957999158
T10=-1.1708204000530853
T11=0.4472136021531272
T12=1.4472136021531272
T13=0.4472135957999158
T20=4.2360680044124255
T21=-4.2360680044124255
T22=-4.2360680044124255
T23=0.
T30=1.6180340022062127
T31=-2.6180340022062127
T32=-1.6180340022062127
T33=0.


CLEBSCH_CUBICS=\
				push b push b mult push d push c push m mult add mult \
				push b push c push d push d push m mult mult add mult \
				push a push d push d push m add mult mult add add \
				push a push c push m mult add mult \
				store c001 \
				push b push d mult \
				push b push 1 push m push c mult add mult \
				push d push a push 1 push m mult add mult add \
				push m push a mult add push c add \
				push c push m push a mult add \
				mult mult \
				store c002 \
				push b \
				push d push c push a push m mult add mult \
				push c push a push m push 1 mult add mult add mult \
				push a push d mult push c push 1 push m mult add mult \
				push m mult add \
				push a push c push m mult add mult \
				store c011 \
				push b push b push c mult mult \
				push 1 push d push m mult add mult \
				push a push b mult push c push d push d push m mult mult add mult \
				push m mult add \
				push a push d mult push c push d push m mult add mult add \
				push a push c push m mult add mult \
				store c012 \
				push m \
				push b push d push m mult add push c mult \
				push d push b push 1 push m mult add mult push m mult add push a mult \
				push b push c mult push d push 1 push m mult add mult add mult \
				push b push d push m mult add mult \
				store d001 \
				push m \
				push d push c push m mult add push a push a mult mult \
				push c push c mult push d push m mult add push a mult add \
				push m push b push c mult mult push c push 1 push m mult add mult add mult \
				push b push d push m mult add mult \
				store d011 \
				push m \
				push c push d mult push d push m mult add push a push a mult mult \
				push c push c mult push d push m mult add push a push b push m mult mult mult add \
				push b push d push c push m mult add mult push c push m mult mult add \
				push b push d push m mult add mult mult \
				store d012 \
				push d push 1 push m mult add push a mult push m push b mult push 1 add push c mult add \
				push b add push m push d mult add \
				push a push c mult mult \
				push b push d push m mult add mult \
				store d112 \
				push m \
				push b push d push m mult add push c mult push d push b push 1 push m mult add mult \
				push m mult add push a mult push b push c mult push d push 1 push m mult add mult add \
				push b push d push m mult add mult mult \
				store m002 \
				push m \
				push d push c push m mult add push a push a mult mult \
				push c push c mult push d push m mult add push a mult add \
				push b push c push m mult mult push c push 1 push m mult add mult add \
				push b push d push m mult add mult mult \
				store m012 \
				push m\
				push c push d mult push d push m mult add push a push a mult mult \
				push m push c push c mult push d push m mult add push a push b mult mult mult add \
				push m push b push d push c push m mult add push c mult mult mult add \
				push b push d push m mult add mult mult \
				store m022 \
				push d push 1 push m mult add push a mult \
				push m push b mult push 1 add push c mult add \
				push b add push m push d mult add \
				push a push c mult mult \
				push b push d push m mult add mult \
				store m122 \
				push m push a mult push c add push d mult push c push a push 1 push m mult add mult add \
				push b mult \
				push m push a push d mult mult push c push 1 push m mult add mult add \
				push b push d push m mult add mult \
				store n002 \
				push m \
				push c push d push m mult add push b mult push m push d push c push 1 push m mult add mult mult add \
				push a mult \
				push b push c mult push d push 1 push m mult add mult add mult \
				push a push b push c push m mult push d push m mult add add add mult \
				store n012 \
				push c push d push m mult add push b mult \
				push m push d push c push 1 push m mult add mult mult  add \
				push a mult \
				push b push c mult push d push 1 push m mult add mult add \
				push a push d mult push m push b push c mult mult add mult \
				store n022 \
				push m \
				push c push d push m mult add push b mult \
				push m push d push c push 1 push m mult add mult mult add \
				push a mult \
				push b push c mult push d push 1 push m mult add mult add \
				push m push a mult push c add mult mult \
				store n112 \
				push m \
				push c push d push m mult add push b mult \
				push m push d push c push 1 push m mult add mult mult add \
				push a mult \
				push b push c mult push d push 1 push m mult add mult add \
				push a push d mult push m push b push c mult mult add mult mult \
				store n122 

Clebsch_up_create_points:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-smooth_curve "Clebsch_map_of_circle_to_defininig_eqn_r2" \
			0.07 1000 5 0 $(TWO_PI) \
		-const a $(CLEBSCH_A) b $(CLEBSCH_B) c $(CLEBSCH_C) d $(CLEBSCH_D) \
			t00 $(T00) t01 $(T01) t02 $(T02) t03 $(T03) \
			t10 $(T10) t11 $(T11) t12 $(T12) t13 $(T13) \
			t20 $(T20) t21 $(T21) t22 $(T22) t23 $(T23) \
			t30 $(T30) t31 $(T31) t32 $(T32) t33 $(T33) \
			r 2 one 1 m -1 \
		-const_end \
		-var t \
			c001 c002 c011 c012 \
			d001 d011 d012 d112 \
			m002 m012 m022 m122 \
			n002 n012 n112 n022 n122 \
			y0 y1 y2 \
			y001 y002 y011 y012 y022 y112 y122 \
			x0 x1 x2 x3 \
			-var_end \
		-code \
			push t cos push r mult store y0 \
			push t sin push r mult store y1 \
			push one store y2 \
			push y0 push y0 push y1 mult mult store y001 \
			push y0 push y0 push y2 mult mult store y002 \
			push y0 push y1 push y1 mult mult store y011 \
			push y0 push y1 push y2 mult mult store y012 \
			push y0 push y2 push y2 mult mult store y022 \
			push y1 push y1 push y2 mult mult store y112 \
			push y1 push y2 push y2 mult mult store y122 \
			$(CLEBSCH_CUBICS) \
				push c001 push y001 mult \
				push c002 push y002 mult add \
				push c011 push y011 mult add \
				push c012 push y012 mult add \
				store x0 \
				push d001 push y001 mult \
				push d011 push y011 mult add \
				push d012 push y012 mult add \
				push d112 push y112 mult add \
				store x1 \
				push m002 push y002 mult \
				push m012 push y012 mult add \
				push m022 push y022 mult add \
				push m122 push y122 mult add \
				store x2 \
				push n002 push y002 mult \
				push n012 push y012 mult add \
				push n022 push y022 mult add \
				push n112 push y112 mult add \
				push n122 push y122 mult add \
				store x3 \
				push x0 push t00 mult \
				push x1 push t10 mult add \
				push x2 push t20 mult add \
				push x3 push t30 mult add \
				return \
				push x0 push t01 mult \
				push x1 push t11 mult add \
				push x2 push t21 mult add \
				push x3 push t31 mult add \
				return \
				push x0 push t02 mult \
				push x1 push t12 mult add \
				push x2 push t22 mult add \
				push x3 push t32 mult add \
				return \
				push x0 push t03 mult \
				push x1 push t13 mult add \
				push x2 push t23 mult add \
				push x3 push t33 mult add \
				return \
		-code_end


Clebsch_surface:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Clebsch_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.9 \
		-default_angle 75 \
		-clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-cubic_orbiter "0,0,0,0,0,-4.236067972,0,0,4.236067972,4.236067972,17.94427188,- 17.94427188,0,0,- 9.472135941,0,0,5.236067971,8.472135938,- 27.41640782" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-point_list_from_csv_file function_Clebsch_map_of_circle_N1000_points.csv \
			-group_of_things_as_interval 0 954 \
			-spheres 1 0.07 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Clebsch_0_*.pov POV
	mv makefile_animation POV


Clebsch_surface_defining_equation:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Clebsch_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.6 \
		-default_angle 75 \
		-clipping_radius 1.6 \
		-camera 0 "1,1,1" "-2,0,2" "0,0,0"  \
		-end \
		-scene_objects \
			-cubic_orbiter "0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Clebsch_0_*.pov POV
	mv makefile_animation POV



Clebsch_surface_defining_equation_and_curves:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask Clebsch_2curves_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.6 \
		-default_angle 75 \
		-clipping_radius 1.6 \
		-camera 0 "1,1,1" "-2,0,2" "0,0,0"  \
		-end \
		-scene_objects \
			-cubic_orbiter "0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-point_list_from_csv_file function_Clebsch_map_of_circle_to_defininig_eqn_N1000_points.csv \
			-group_of_things_as_interval 0 656 \
			-spheres 1 0.07 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-point_list_from_csv_file function_Clebsch_map_of_circle_to_defininig_eqn_r2_N1000_points.csv \
			-group_of_things_as_interval 656 1042 \
			-spheres 2 0.07 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Clebsch_2curves_0_*.pov POV
	mv makefile_animation POV

#			-point_list_from_csv_file function_Clebsch_map_of_circle_N1000_points.csv \
#			-group_of_things_as_interval 0 954 \
#			-spheres 1 0.07 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \




F7_povray:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask F7_15_lines_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 1.5 \
		-default_angle 80 \
		-clipping_radius 4.4 \
		-omit_bottom_plane \
		-camera 0 "1,1,1"  "-4.5,3.5,6" "0,0,0"  \
		-end \
		-scene_objects \
			-cubic_lex "0, 0, 6, 0, 0, -13.39014946, -3.341901346, -6.972931640, 5.827182718, 0, 0, 7.390149464, 7.390149464, 6.972931640, -1.512349728, -8.485281372, 0, 0, 0, 0" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-line_through_point_with_direction "0, 0, 0, 1, 0, 0" \
			-line_through_point_with_direction "0, 0, -1, 0, 1, 0" \
			-line_through_point_with_direction "0, 0, 0, 0, 0, -1" \
			-line_through_point_with_direction "1, 0, 0, 1, 1, 1" \
			-line_through_point_with_direction "-1.414213562, 0, 0, 4.146264370, 1.732050808, 1.732050808" \
			-line_through_point_with_direction "0, 1.732050808, -1, 2.414213562, -0.317837246, 2.414213562" \
			-line_through_point_with_direction "-2.133352390, 0, -1, 1.674708020, 1, 0" \
			-line_through_point_with_direction "-2.539058015, 0, 0, 2.211360755, 1, 0" \
			-line_through_point_with_direction "0, 1.148188060, 0, 0, -0.9435440612, 1" \
			-line_through_point_with_direction "-0.9711971171, 0, 0, 1.162155272, 0, 1" \
			-line_through_point_with_direction "2.096037870, 2.096037870, 0, -1.065851905, -1.065851905, 1" \
			-line_through_point_with_direction "3.921555783, 2.921555781, 0, -1.722456585, -1.722456585, 1" \
			-group_of_things_as_interval 0 12 \
			-lines 1 0.04 "texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv F7_15_lines_0_*.pov POV
	mv makefile_animation POV






F7_video:
	- rm -r FRAMES
	- mkdir FRAMES
	- rm fifteen_with_lines.mp4
	$(ORBITER_PATH)orbiter.out \
		-prepare_frames \
			-i 0 30 F7b/F7_15_lines_0_%03d.png \
			-output_starts_at 0 \
			-o FRAMES/frame%04d.png \
		-end
	ffmpeg -r 5 -f image2 -i FRAMES/frame%04d.png \
		-f mp4 -q:v 0 -vcodec mpeg4  fifteen_with_lines.mp4


McKean_povray:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask McKean_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 1.5 \
		-default_angle 80 \
		-clipping_radius 4.4 \
		-omit_bottom_plane \
		-camera 0 "1,1,1"  "-4.5,3.5,6" "0,0,0"  \
		-end \
		-scene_objects \
			-cubic_lex "0, 0, 1, 0, 0, -1, -2, 1, 2, 0, 0, 1, 1,-1, -1, -1, 0, 0, 0, 0" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv McKean_0_*.pov POV
	mv makefile_animation POV

McKean_video:
	- rm -r FRAMES
	- mkdir FRAMES
	- rm McKean.mp4
	$(ORBITER_PATH)orbiter.out \
		-prepare_frames \
			-i 0 30 MCKEAN/McKean_0_%03d.png \
			-output_starts_at 0 \
			-o FRAMES/frame%04d.png \
		-end
	ffmpeg -r 5 -f image2 -i FRAMES/frame%04d.png \
		-f mp4 -q:v 0 -vcodec mpeg4  McKean.mp4




#############################################################################################################
# Section 12.4: Continuous Function Plotter



SECTION_CONTINUOUS_FUNCTION_PLOTTER:


lissajous:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-smooth_curve "lissajous" 0.07 2000 15 0 18.85 \
		-const a 3 b 2 c 1.57 r 7 -const_end \
		-var t -var_end \
		-code \
			push t push a mult push c add sin push r mult return \
			push t push b mult sin push r mult  return \
		-code_end \

#function_lissajous_N2000_points.csv

lissajous_plot:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 1 \
		-output_mask lissajous_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.40 \
		-default_angle 45 \
		-clipping_radius 5 \
		-omit_bottom_plane  \
		-camera 0 "0,-1,0" "0,0,12" "0,0,0"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-lines 0 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 1 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 2 0.09 "texture{ pigment{ color Yellow } }" \
			-group_of_things_as_interval 3 39 \
			-lines 3 0.02 "texture{ pigment{ color Black } }" \
			-point_list_from_csv_file function_lissajous_N2000_points.csv \
			-group_of_things_as_interval 0 6524\
			-spheres 4 0.1 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-planes 5 "texture{ pigment{ color Blue*0.5 transmit 0.5 } }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv lissajous_0_*.pov POV
	mv makefile_animation POV

lissajous_3d:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-smooth_curve "lissajous_3d" 0.07 2000 50 0 18.85 \
		-const a 3 b 2 c 1.57 r 7 -const_end \
		-var t -var_end \
		-code \
			push t push a mult push c add sin push r mult return \
			push t push b mult sin push r mult  return \
			push t return \
		-code_end \

lissajous_3d_plot:
	$(ORBITER_PATH)orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 \
		-output_mask lissajous_3d_%d_%03d.pov \
		-video_options -W 1024 -H 768 \
		-global_picture_scale 0.40 \
		-default_angle 45 \
		-clipping_radius 5 \
		-omit_bottom_plane  \
		-camera 0 "0,0,1" "7,7,5" "0,0,1"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-lines 0 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 1 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 2 0.09 "texture{ pigment{ color Yellow } }" \
			-group_of_things_as_interval 3 39 \
			-lines 3 0.02 "texture{ pigment{ color Black } }" \
			-point_list_from_csv_file function_lissajous_3d_N2000_points.csv \
			-group_of_things_as_interval 0 6538\
			-spheres 4 0.1 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv lissajous_3d_0_*.pov POV
	mv makefile_animation POV

#			-planes 5 "texture{ pigment{ color Blue*0.5 transmit 0.5 } }" \




#############################################################################################################
# Chapter 13 - Miscellaneous
#############################################################################################################


#############################################################################################################
# Section 13.1: Miscellaneous

SECTION_MISCELLANEOUS:




misc_select:
	$(ORBITER_PATH)orbiter.out -v 3 \
		-define F -finite_field -q 7 -end \
		-with F -do -finite_field_activity -cheat_sheet_GF -end
	$(ORBITER_PATH)orbiter.out -v 4 -csv_file_select_rows_and_cols \
		GF_q7_multiplication_table_reordered.csv \
		"0,2,4" "0,2,4" 


misc_join:
	$(ORBITER_PATH)orbiter.out -v 4 \
		-csv_file_join poly_orbits_d3_n3_q2_select_F2.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F4.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F8.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F16.csv Orbit_idx \
		-csv_file_join poly_orbits_d3_n3_q2_select_F32.csv Orbit_idx 




#############################################################################################################
# Section 13.2: Limitations

SECTION_LIMITATIONS:



####


geo_10_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-geometry_builder -V 10 -B 10 -TDO 3 -fuse 1 -fname_GEO 10_3 -test 4,5,6,7,8,9 FFFF \
		-end


# 10 geos
# 8/26/2021: 0 sec on Mac


geo_11_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-geometry_builder -V 11 -B 11 -TDO 3 -fuse 1 -fname_GEO 11_3 -test 4,5,6,7,8,9,10 FFFF \
		-end

# 31 geos
# 8/26/2021: 0 sec on Mac

geo_12_3:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-geometry_builder -V 12 -B 12 -TDO 3 -fuse 1 -fname_GEO 12_3 -test 4,5,6,7,8,9,10,11 FFFF \
		-end

# 229 geos
# 8/26/2021: 41 sec on Mac


geo_12_3_b:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-geometry_builder -V 12 -B 12 -TDO 3 -fuse 1 -fname_GEO 12_3 -test 4,5,6,7,12 FFFF -test2 8,9,10,11 FFFF \
		-end


geo_12_3_c:
	$(ORBITER_PATH)orbiter.out -v 2 \
		-define D -design  -end \
		-with D -do \
		-design_activity \
			-canonical_form \
			-input -file_of_incidence_geometries 12_3.inc 12 12 36 -end \
			-save_ago \
		-end



