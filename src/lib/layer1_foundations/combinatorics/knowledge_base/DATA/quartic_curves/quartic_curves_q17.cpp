// file: quartic_curves_q17.cpp
// created by Orbiter 
// creation date: Sat Jul 19 09:55:54 +03 2025

// 
static int quartic_curves_q17_nb_reps = 7;
static int quartic_curves_q17_size = 15;
// the equations:
static int quartic_curves_q17_reps[] = {
	1, 9, 8, 3, 13, 9, 13, 13, 16, 4, 13, 13, 6, 14, 10, 
	15, 16, 16, 11, 8, 16, 0, 9, 4, 9, 14, 0, 9, 15, 13, 
	15, 16, 6, 11, 14, 16, 0, 0, 0, 9, 1, 3, 3, 6, 10, 
	15, 2, 8, 11, 10, 7, 10, 13, 3, 3, 3, 12, 15, 13, 7, 
	15, 2, 7, 11, 1, 7, 2, 14, 9, 3, 15, 11, 12, 6, 12, 
	8, 9, 13, 7, 7, 6, 8, 1, 14, 5, 13, 5, 1, 15, 7, 
	1, 9, 8, 10, 9, 4, 3, 6, 12, 2, 6, 0, 10, 16, 0, 
};
// the stabilizer orders:
static const char *quartic_curves_q17_stab_order[] = {
	"4",
	"2",
	"96",
	"24",
	"6",
	"8",
	"24",
};
// the 28 bitangents:
static long int quartic_curves_q17_Bitangents[] = { 
	1, 126, 119, 65, 77, 221, 79, 304, 249, 179, 231, 0, 219, 137, 268, 167, 283, 259, 197, 3, 72, 112, 292, 181, 109, 247, 166, 188, 
	4, 54, 272, 129, 246, 89, 188, 243, 83, 55, 206, 0, 75, 112, 282, 105, 60, 182, 154, 179, 270, 9, 127, 155, 114, 209, 87, 148, 
	4, 18, 102, 129, 261, 205, 152, 298, 88, 21, 94, 0, 81, 127, 95, 297, 283, 235, 158, 67, 90, 62, 210, 293, 286, 266, 26, 234, 
	4, 306, 35, 92, 104, 12, 23, 22, 232, 81, 181, 0, 161, 287, 145, 150, 242, 250, 176, 103, 26, 78, 151, 228, 237, 56, 173, 231, 
	4, 18, 204, 154, 219, 90, 110, 202, 156, 188, 20, 0, 227, 31, 200, 268, 242, 240, 93, 153, 262, 7, 184, 228, 243, 30, 121, 259, 
	17, 54, 204, 32, 191, 255, 42, 270, 300, 28, 37, 0, 104, 188, 128, 218, 179, 208, 2, 246, 61, 275, 163, 266, 111, 293, 150, 83, 
	17, 306, 241, 68, 101, 202, 278, 270, 46, 182, 175, 0, 227, 99, 32, 65, 125, 113, 165, 40, 88, 265, 236, 149, 78, 51, 109, 228, 
};
static int quartic_curves_q17_make_element_size = 9;
static int quartic_curves_q17_stab_gens_fst[] = { 
	0, 2, 3, 5, 8, 10, 12};
static int quartic_curves_q17_stab_gens_len[] = { 
	2, 1, 2, 3, 2, 2, 3};
static int quartic_curves_q17_stab_gens[] = {
	0,4,10,16,12,6,2,2,1,
	16,7,1,8,6,3,15,16,14,
	2,3,2,12,11,15,10,6,1,
	13,13,13,13,12,13,3,12,9,
	11,7,0,12,6,0,0,0,1,
	6,0,0,9,11,0,7,2,6,
	13,16,0,13,3,0,6,3,1,
	7,0,6,2,10,5,11,8,7,
	13,0,16,15,14,15,7,0,4,
	8,13,4,3,2,7,2,8,7,
	8,0,0,1,0,1,9,13,0,
	11,8,12,0,7,0,11,5,6,
	2,0,11,13,5,9,5,0,15,
	10,9,7,8,4,8,11,15,3,
	9,1,8,6,15,12,5,14,15,
};
