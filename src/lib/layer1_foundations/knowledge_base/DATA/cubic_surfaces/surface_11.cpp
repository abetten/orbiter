// file surface_11.cpp
// created by Orbiter
// date Sat Feb 18 02:11:23 +03 2023
static int surface_11_nb_reps = 2;
static int surface_11_size = 20;
// the equations:
static int surface_11_reps[] = {
	0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 7, 0, 2, 0, 0, 2, 7, 7, 6,
	0, 0, 0, 0, 0, 0, 1, 0, 9, 0, 0, 1, 0, 10, 0, 0, 1, 0, 8, 3,
};
// the stabilizer orders:
static const char *surface_11_stab_order[] = {
	"24",
	"120",
};
// the number of Eckardt points:
static int surface_11_nb_E[] = {
	6, 10};
// the lines in the order double six a_i, b_i and 15 more lines c_ij:
static long int surface_11_Lines[] = {
	121, 16104, 255, 659, 15722, 15822, 6528, 624, 299, 1440, 16225, 0, 753, 168, 1181, 7438, 801, 13477, 5785, 16209, 7337, 12720, 10504, 9636, 3585, 3525, 1189,
	121, 16104, 255, 659, 15739, 1985, 9121, 780, 986, 144, 16225, 0, 376, 995, 154, 1592, 1332, 3912, 13322, 16196, 1463, 8220, 1727, 4860, 2119, 12449, 524,
};
static int surface_11_make_element_size = 16;
static int surface_11_stab_gens_fst[] = { 0, 3};
static int surface_11_stab_gens_len[] = { 3, 5};
static int surface_11_stab_gens[] = {
	1,0,4,2,0,1,1,6,0,0,7,3,0,0,6,4,
	1,9,10,2,6,8,5,3,9,6,2,8,5,7,1,4,
	0,0,0,1,0,0,2,0,0,4,0,1,1,0,2,0,
	1,0,0,0,0,1,0,0,2,0,10,0,10,1,0,10,
	1,0,0,0,7,10,0,0,2,0,10,0,6,10,3,1,
	1,0,0,0,8,0,1,0,10,10,0,0,2,2,10,10,
	0,1,0,2,0,2,0,0,9,8,9,9,2,10,0,0,
	1,9,0,1,0,1,0,0,0,5,1,3,0,4,0,10,
};

#if 0
static int surface_11_nb_reps = 2;
static int surface_11_size = 20;
static int surface_11_reps[] = {
	0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 7, 0, 2, 0, 0, 6, 7, 7, 2, 
	0, 0, 0, 0, 0, 0, 1, 0, 9, 0, 0, 1, 0, 10, 0, 0, 3, 8, 0, 1, 
};
static const char *surface_11_stab_order[] = {
	"24",
	"120",
};
static int surface_11_nb_E[] = {
	6, 10};
static long int surface_11_Lines[] = {
	0, 121, 801, 16104, 16225, 255, 9636, 15722, 659, 7337, 1189, 3525, 168, 624, 7438, 299, 1181, 753, 1440, 15822, 12720, 13477, 16209, 10504, 5785, 6528, 3585, 
	0, 121, 1332, 16104, 1463, 16225, 255, 659, 524, 15739, 12449, 4860, 144, 154, 1592, 376, 986, 995, 780, 1727, 13322, 8220, 1985, 16196, 9121, 2119, 3912, 
};
static int surface_11_make_element_size = 16;
static int surface_11_stab_gens_fst[] = { 0, 2};
static int surface_11_stab_gens_len[] = { 2, 4};
static int surface_11_stab_gens[] = {
	 5,  8, 10,  2,  2,  4,  4,  1,  2,  6,  3,  7, 10,  2,  9, 10, 
	 1,  0,  4,  2,  0,  1,  1,  6,  0,  0,  7,  3,  0,  0,  6,  4, 
	 7,  0,  0,  0,  0,  7,  0,  0,  3,  0,  4,  0,  4,  7,  0,  4, 
	 7,  0,  0,  0,  5,  4,  0,  0,  3,  0,  4,  0,  9,  4, 10,  7, 
	 4,  0,  0,  0, 10,  0,  4,  0,  1,  4,  0,  0,  9,  3,  8,  4, 
	 1,  4,  0,  1,  8, 10,  1,  0,  3,  0,  0,  3,  5,  0,  2, 10, 
};
#endif
