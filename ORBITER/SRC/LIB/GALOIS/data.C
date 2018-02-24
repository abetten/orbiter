// data.C
//
// Anton Betten
//
// started:  July 9, 2009


// TP data is available for:
// 2^2 = 4
// 3^2 = 9
// 2^4 = 16
// 4^2 = 16
// 5^2 = 25
// 3^3 = 27



#include "galois.h"





#include "../DATA/data_hyperovals.C"

#include "../DATA/surface_4.C"
#include "../DATA/surface_7.C"
#include "../DATA/surface_8.C"
#include "../DATA/surface_9.C"
#include "../DATA/surface_11.C"
#include "../DATA/surface_13.C"
#include "../DATA/surface_16.C"
#include "../DATA/surface_17.C"
#include "../DATA/surface_19.C"
#include "../DATA/surface_23.C"
#include "../DATA/surface_25.C"
#include "../DATA/surface_27.C"
#include "../DATA/surface_29.C"
#include "../DATA/surface_31.C"
#include "../DATA/surface_32.C"
#include "../DATA/surface_37.C"
#include "../DATA/surface_41.C"
#include "../DATA/surface_43.C"
#include "../DATA/surface_47.C"
#include "../DATA/surface_49.C"
#include "../DATA/surface_53.C"
#include "../DATA/surface_59.C"
#include "../DATA/surface_61.C"
#include "../DATA/surface_64.C"
#include "../DATA/surface_67.C"
#include "../DATA/surface_71.C"
#include "../DATA/surface_73.C"
#include "../DATA/surface_79.C"
#include "../DATA/surface_81.C"
#include "../DATA/surface_83.C"
#include "../DATA/surface_89.C"
#include "../DATA/surface_97.C"

#include "../DATA/data_DH.C"
#include "../DATA/data_TP.C"
#include "../DATA/data_BLT.C"

// missing:
//53 done Aug 13, 2017
//59
//61
//67



INT cubic_surface_nb_reps(INT q)
{
	INT nb;

	if (q == 4) {
		nb = surface_4_nb_reps;
		}
	else if (q == 7) {
		nb = surface_7_nb_reps;
		}
	else if (q == 8) {
		nb = surface_8_nb_reps;
		}
	else if (q == 9) {
		nb = surface_9_nb_reps;
		}
	else if (q == 11) {
		nb = surface_11_nb_reps;
		}
	else if (q == 13) {
		nb = surface_13_nb_reps;
		}
	else if (q == 16) {
		nb = surface_16_nb_reps;
		}
	else if (q == 17) {
		nb = surface_17_nb_reps;
		}
	else if (q == 19) {
		nb = surface_19_nb_reps;
		}
	else if (q == 23) {
		nb = surface_23_nb_reps;
		}
	else if (q == 25) {
		nb = surface_25_nb_reps;
		}
	else if (q == 27) {
		nb = surface_27_nb_reps;
		}
	else if (q == 29) {
		nb = surface_29_nb_reps;
		}
	else if (q == 31) {
		nb = surface_31_nb_reps;
		}
	else if (q == 32) {
		nb = surface_32_nb_reps;
		}
	else if (q == 37) {
		nb = surface_37_nb_reps;
		}
	else if (q == 41) {
		nb = surface_41_nb_reps;
		}
	else if (q == 43) {
		nb = surface_43_nb_reps;
		}
	else if (q == 47) {
		nb = surface_47_nb_reps;
		}
	else if (q == 49) {
		nb = surface_49_nb_reps;
		}
	else if (q == 53) {
		nb = surface_53_nb_reps;
		}
	else if (q == 59) {
		nb = surface_59_nb_reps;
		}
	else if (q == 61) {
		nb = surface_61_nb_reps;
		}
	else if (q == 64) {
		nb = surface_64_nb_reps;
		}
	else if (q == 67) {
		nb = surface_67_nb_reps;
		}
	else if (q == 71) {
		nb = surface_71_nb_reps;
		}
	else if (q == 73) {
		nb = surface_73_nb_reps;
		}
	else if (q == 79) {
		nb = surface_79_nb_reps;
		}
	else if (q == 81) {
		nb = surface_81_nb_reps;
		}
	else if (q == 83) {
		nb = surface_83_nb_reps;
		}
	else if (q == 89) {
		nb = surface_89_nb_reps;
		}
	else if (q == 97) {
		nb = surface_97_nb_reps;
		}
	else {
		cout << "cubic_surface_nb_reps q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

INT *cubic_surface_representative(INT q, INT i)
// i starts from 0
{
	INT *p, nb, sz;
	if (q == 4) {
		p = surface_4_reps;
		nb = surface_4_nb_reps;
		sz = surface_4_size;
		}
	else if (q == 7) {
		p = surface_7_reps;
		nb = surface_7_nb_reps;
		sz = surface_7_size;
		}
	else if (q == 8) {
		p = surface_8_reps;
		nb = surface_8_nb_reps;
		sz = surface_8_size;
		}
	else if (q == 9) {
		p = surface_9_reps;
		nb = surface_9_nb_reps;
		sz = surface_9_size;
		}
	else if (q == 11) {
		p = surface_11_reps;
		nb = surface_11_nb_reps;
		sz = surface_11_size;
		}
	else if (q == 13) {
		p = surface_13_reps;
		nb = surface_13_nb_reps;
		sz = surface_13_size;
		}
	else if (q == 16) {
		p = surface_16_reps;
		nb = surface_16_nb_reps;
		sz = surface_16_size;
		}
	else if (q == 17) {
		p = surface_17_reps;
		nb = surface_17_nb_reps;
		sz = surface_17_size;
		}
	else if (q == 19) {
		p = surface_19_reps;
		nb = surface_19_nb_reps;
		sz = surface_19_size;
		}
	else if (q == 23) {
		p = surface_23_reps;
		nb = surface_23_nb_reps;
		sz = surface_23_size;
		}
	else if (q == 25) {
		p = surface_25_reps;
		nb = surface_25_nb_reps;
		sz = surface_25_size;
		}
	else if (q == 27) {
		p = surface_27_reps;
		nb = surface_27_nb_reps;
		sz = surface_27_size;
		}
	else if (q == 29) {
		p = surface_29_reps;
		nb = surface_29_nb_reps;
		sz = surface_29_size;
		}
	else if (q == 31) {
		p = surface_31_reps;
		nb = surface_31_nb_reps;
		sz = surface_31_size;
		}
	else if (q == 32) {
		p = surface_32_reps;
		nb = surface_32_nb_reps;
		sz = surface_32_size;
		}
	else if (q == 37) {
		p = surface_37_reps;
		nb = surface_37_nb_reps;
		sz = surface_37_size;
		}
	else if (q == 41) {
		p = surface_41_reps;
		nb = surface_41_nb_reps;
		sz = surface_41_size;
		}
	else if (q == 43) {
		p = surface_43_reps;
		nb = surface_43_nb_reps;
		sz = surface_43_size;
		}
	else if (q == 47) {
		p = surface_47_reps;
		nb = surface_47_nb_reps;
		sz = surface_47_size;
		}
	else if (q == 49) {
		p = surface_49_reps;
		nb = surface_49_nb_reps;
		sz = surface_49_size;
		}
	else if (q == 53) {
		p = surface_53_reps;
		nb = surface_53_nb_reps;
		sz = surface_53_size;
		}
	else if (q == 59) {
		p = surface_59_reps;
		nb = surface_59_nb_reps;
		sz = surface_59_size;
		}
	else if (q == 61) {
		p = surface_61_reps;
		nb = surface_61_nb_reps;
		sz = surface_61_size;
		}
	else if (q == 64) {
		p = surface_64_reps;
		nb = surface_64_nb_reps;
		sz = surface_64_size;
		}
	else if (q == 67) {
		p = surface_67_reps;
		nb = surface_67_nb_reps;
		sz = surface_67_size;
		}
	else if (q == 71) {
		p = surface_71_reps;
		nb = surface_71_nb_reps;
		sz = surface_71_size;
		}
	else if (q == 73) {
		p = surface_73_reps;
		nb = surface_73_nb_reps;
		sz = surface_73_size;
		}
	else if (q == 79) {
		p = surface_79_reps;
		nb = surface_79_nb_reps;
		sz = surface_79_size;
		}
	else if (q == 81) {
		p = surface_81_reps;
		nb = surface_81_nb_reps;
		sz = surface_81_size;
		}
	else if (q == 83) {
		p = surface_83_reps;
		nb = surface_83_nb_reps;
		sz = surface_83_size;
		}
	else if (q == 89) {
		p = surface_89_reps;
		nb = surface_89_nb_reps;
		sz = surface_89_size;
		}
	else if (q == 97) {
		p = surface_97_reps;
		nb = surface_97_nb_reps;
		sz = surface_97_size;
		}
	else {
		cout << "cubic_surface_representative q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "cubic_surface_representative q=" << q << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "cubic_surface_representative q=" << q << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void cubic_surface_stab_gens(INT q, INT i, INT *&data, INT &nb_gens, INT &data_size, const BYTE *&stab_order)
{
	INT *Reps;
	INT nb, make_element_size;
	INT f, l;
	
	if (q == 4) {
		Reps = surface_4_stab_gens;
		nb = surface_4_nb_reps;
		make_element_size = surface_4_make_element_size;
		f = surface_4_stab_gens_fst[i];
		l = surface_4_stab_gens_len[i];
		stab_order = surface_4_stab_order[i];
		}
	else if (q == 7) {
		Reps = surface_7_stab_gens;
		nb = surface_7_nb_reps;
		make_element_size = surface_7_make_element_size;
		f = surface_7_stab_gens_fst[i];
		l = surface_7_stab_gens_len[i];
		stab_order = surface_7_stab_order[i];
		}
	else if (q == 8) {
		Reps = surface_8_stab_gens;
		nb = surface_8_nb_reps;
		make_element_size = surface_8_make_element_size;
		f = surface_8_stab_gens_fst[i];
		l = surface_8_stab_gens_len[i];
		stab_order = surface_8_stab_order[i];
		}
	else if (q == 9) {
		Reps = surface_9_stab_gens;
		nb = surface_9_nb_reps;
		make_element_size = surface_9_make_element_size;
		f = surface_9_stab_gens_fst[i];
		l = surface_9_stab_gens_len[i];
		stab_order = surface_9_stab_order[i];
		}
	else if (q == 11) {
		Reps = surface_11_stab_gens;
		nb = surface_11_nb_reps;
		make_element_size = surface_11_make_element_size;
		f = surface_11_stab_gens_fst[i];
		l = surface_11_stab_gens_len[i];
		stab_order = surface_11_stab_order[i];
		}
	else if (q == 13) {
		Reps = surface_13_stab_gens;
		nb = surface_13_nb_reps;
		make_element_size = surface_13_make_element_size;
		f = surface_13_stab_gens_fst[i];
		l = surface_13_stab_gens_len[i];
		stab_order = surface_13_stab_order[i];
		}
	else if (q == 16) {
		Reps = surface_16_stab_gens;
		nb = surface_16_nb_reps;
		make_element_size = surface_16_make_element_size;
		f = surface_16_stab_gens_fst[i];
		l = surface_16_stab_gens_len[i];
		stab_order = surface_16_stab_order[i];
		}
	else if (q == 17) {
		Reps = surface_17_stab_gens;
		nb = surface_17_nb_reps;
		make_element_size = surface_17_make_element_size;
		f = surface_17_stab_gens_fst[i];
		l = surface_17_stab_gens_len[i];
		stab_order = surface_17_stab_order[i];
		}
	else if (q == 19) {
		Reps = surface_19_stab_gens;
		nb = surface_19_nb_reps;
		make_element_size = surface_19_make_element_size;
		f = surface_19_stab_gens_fst[i];
		l = surface_19_stab_gens_len[i];
		stab_order = surface_19_stab_order[i];
		}
	else if (q == 23) {
		Reps = surface_23_stab_gens;
		nb = surface_23_nb_reps;
		make_element_size = surface_23_make_element_size;
		f = surface_23_stab_gens_fst[i];
		l = surface_23_stab_gens_len[i];
		stab_order = surface_23_stab_order[i];
		}
	else if (q == 25) {
		Reps = surface_25_stab_gens;
		nb = surface_25_nb_reps;
		make_element_size = surface_25_make_element_size;
		f = surface_25_stab_gens_fst[i];
		l = surface_25_stab_gens_len[i];
		stab_order = surface_25_stab_order[i];
		}
	else if (q == 27) {
		Reps = surface_27_stab_gens;
		nb = surface_27_nb_reps;
		make_element_size = surface_27_make_element_size;
		f = surface_27_stab_gens_fst[i];
		l = surface_27_stab_gens_len[i];
		stab_order = surface_27_stab_order[i];
		}
	else if (q == 29) {
		Reps = surface_29_stab_gens;
		nb = surface_29_nb_reps;
		make_element_size = surface_29_make_element_size;
		f = surface_29_stab_gens_fst[i];
		l = surface_29_stab_gens_len[i];
		stab_order = surface_29_stab_order[i];
		}
	else if (q == 31) {
		Reps = surface_31_stab_gens;
		nb = surface_31_nb_reps;
		make_element_size = surface_31_make_element_size;
		f = surface_31_stab_gens_fst[i];
		l = surface_31_stab_gens_len[i];
		stab_order = surface_31_stab_order[i];
		}
	else if (q == 32) {
		Reps = surface_32_stab_gens;
		nb = surface_32_nb_reps;
		make_element_size = surface_32_make_element_size;
		f = surface_32_stab_gens_fst[i];
		l = surface_32_stab_gens_len[i];
		stab_order = surface_32_stab_order[i];
		}
	else if (q == 37) {
		Reps = surface_37_stab_gens;
		nb = surface_37_nb_reps;
		make_element_size = surface_37_make_element_size;
		f = surface_37_stab_gens_fst[i];
		l = surface_37_stab_gens_len[i];
		stab_order = surface_37_stab_order[i];
		}
	else if (q == 41) {
		Reps = surface_41_stab_gens;
		nb = surface_41_nb_reps;
		make_element_size = surface_41_make_element_size;
		f = surface_41_stab_gens_fst[i];
		l = surface_41_stab_gens_len[i];
		stab_order = surface_41_stab_order[i];
		}
	else if (q == 43) {
		Reps = surface_43_stab_gens;
		nb = surface_43_nb_reps;
		make_element_size = surface_43_make_element_size;
		f = surface_43_stab_gens_fst[i];
		l = surface_43_stab_gens_len[i];
		stab_order = surface_43_stab_order[i];
		}
	else if (q == 47) {
		Reps = surface_47_stab_gens;
		nb = surface_47_nb_reps;
		make_element_size = surface_47_make_element_size;
		f = surface_47_stab_gens_fst[i];
		l = surface_47_stab_gens_len[i];
		stab_order = surface_47_stab_order[i];
		}
	else if (q == 49) {
		Reps = surface_49_stab_gens;
		nb = surface_49_nb_reps;
		make_element_size = surface_49_make_element_size;
		f = surface_49_stab_gens_fst[i];
		l = surface_49_stab_gens_len[i];
		stab_order = surface_49_stab_order[i];
		}
	else if (q == 53) {
		Reps = surface_53_stab_gens;
		nb = surface_53_nb_reps;
		make_element_size = surface_53_make_element_size;
		f = surface_53_stab_gens_fst[i];
		l = surface_53_stab_gens_len[i];
		stab_order = surface_53_stab_order[i];
		}
	else if (q == 59) {
		Reps = surface_59_stab_gens;
		nb = surface_59_nb_reps;
		make_element_size = surface_59_make_element_size;
		f = surface_59_stab_gens_fst[i];
		l = surface_59_stab_gens_len[i];
		stab_order = surface_59_stab_order[i];
		}
	else if (q == 61) {
		Reps = surface_61_stab_gens;
		nb = surface_61_nb_reps;
		make_element_size = surface_61_make_element_size;
		f = surface_61_stab_gens_fst[i];
		l = surface_61_stab_gens_len[i];
		stab_order = surface_61_stab_order[i];
		}
	else if (q == 64) {
		Reps = surface_64_stab_gens;
		nb = surface_64_nb_reps;
		make_element_size = surface_64_make_element_size;
		f = surface_64_stab_gens_fst[i];
		l = surface_64_stab_gens_len[i];
		stab_order = surface_64_stab_order[i];
		}
	else if (q == 67) {
		Reps = surface_67_stab_gens;
		nb = surface_67_nb_reps;
		make_element_size = surface_67_make_element_size;
		f = surface_67_stab_gens_fst[i];
		l = surface_67_stab_gens_len[i];
		stab_order = surface_67_stab_order[i];
		}
	else if (q == 71) {
		Reps = surface_71_stab_gens;
		nb = surface_71_nb_reps;
		make_element_size = surface_71_make_element_size;
		f = surface_71_stab_gens_fst[i];
		l = surface_71_stab_gens_len[i];
		stab_order = surface_71_stab_order[i];
		}
	else if (q == 73) {
		Reps = surface_73_stab_gens;
		nb = surface_73_nb_reps;
		make_element_size = surface_73_make_element_size;
		f = surface_73_stab_gens_fst[i];
		l = surface_73_stab_gens_len[i];
		stab_order = surface_73_stab_order[i];
		}
	else if (q == 79) {
		Reps = surface_79_stab_gens;
		nb = surface_79_nb_reps;
		make_element_size = surface_79_make_element_size;
		f = surface_79_stab_gens_fst[i];
		l = surface_79_stab_gens_len[i];
		stab_order = surface_79_stab_order[i];
		}
	else if (q == 81) {
		Reps = surface_81_stab_gens;
		nb = surface_81_nb_reps;
		make_element_size = surface_81_make_element_size;
		f = surface_81_stab_gens_fst[i];
		l = surface_81_stab_gens_len[i];
		stab_order = surface_81_stab_order[i];
		}
	else if (q == 83) {
		Reps = surface_83_stab_gens;
		nb = surface_83_nb_reps;
		make_element_size = surface_83_make_element_size;
		f = surface_83_stab_gens_fst[i];
		l = surface_83_stab_gens_len[i];
		stab_order = surface_83_stab_order[i];
		}
	else if (q == 89) {
		Reps = surface_89_stab_gens;
		nb = surface_89_nb_reps;
		make_element_size = surface_89_make_element_size;
		f = surface_89_stab_gens_fst[i];
		l = surface_89_stab_gens_len[i];
		stab_order = surface_89_stab_order[i];
		}
	else if (q == 97) {
		Reps = surface_97_stab_gens;
		nb = surface_97_nb_reps;
		make_element_size = surface_97_make_element_size;
		f = surface_97_stab_gens_fst[i];
		l = surface_97_stab_gens_len[i];
		stab_order = surface_97_stab_order[i];
		}
	else {
		cout << "cubic_surface_stab_gens q=" << q << " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "cubic_surface_stab_gens q=" << q << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "cubic_surface_stab_gens q=" << q << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
}

INT cubic_surface_nb_Eckardt_points(INT q, INT i)
// i starts from 0
{
	INT *p, nb; //, sz;
	if (q == 4) {
		p = surface_4_nb_E;
		nb = surface_4_nb_reps;
		//sz = surface_4_size;
		}
	else if (q == 7) {
		p = surface_7_nb_E;
		nb = surface_7_nb_reps;
		//sz = surface_7_size;
		}
	else if (q == 8) {
		p = surface_8_nb_E;
		nb = surface_8_nb_reps;
		//sz = surface_8_size;
		}
	else if (q == 9) {
		p = surface_9_nb_E;
		nb = surface_9_nb_reps;
		//sz = surface_9_size;
		}
	else if (q == 11) {
		p = surface_11_nb_E;
		nb = surface_11_nb_reps;
		//sz = surface_11_size;
		}
	else if (q == 13) {
		p = surface_13_nb_E;
		nb = surface_13_nb_reps;
		//sz = surface_13_size;
		}
	else if (q == 16) {
		p = surface_16_nb_E;
		nb = surface_16_nb_reps;
		//sz = surface_16_size;
		}
	else if (q == 17) {
		p = surface_17_nb_E;
		nb = surface_17_nb_reps;
		//sz = surface_17_size;
		}
	else if (q == 19) {
		p = surface_19_nb_E;
		nb = surface_19_nb_reps;
		//sz = surface_19_size;
		}
	else if (q == 23) {
		p = surface_23_nb_E;
		nb = surface_23_nb_reps;
		//sz = surface_23_size;
		}
	else if (q == 25) {
		p = surface_25_nb_E;
		nb = surface_25_nb_reps;
		//sz = surface_25_size;
		}
	else if (q == 27) {
		p = surface_27_nb_E;
		nb = surface_27_nb_reps;
		//sz = surface_27_size;
		}
	else if (q == 29) {
		p = surface_29_nb_E;
		nb = surface_29_nb_reps;
		//sz = surface_29_size;
		}
	else if (q == 31) {
		p = surface_31_nb_E;
		nb = surface_31_nb_reps;
		//sz = surface_31_size;
		}
	else if (q == 32) {
		p = surface_32_nb_E;
		nb = surface_32_nb_reps;
		//sz = surface_32_size;
		}
	else if (q == 37) {
		p = surface_37_nb_E;
		nb = surface_37_nb_reps;
		//sz = surface_37_size;
		}
	else if (q == 41) {
		p = surface_41_nb_E;
		nb = surface_41_nb_reps;
		//sz = surface_41_size;
		}
	else if (q == 43) {
		p = surface_43_nb_E;
		nb = surface_43_nb_reps;
		//sz = surface_43_size;
		}
	else if (q == 47) {
		p = surface_47_nb_E;
		nb = surface_47_nb_reps;
		//sz = surface_47_size;
		}
	else if (q == 49) {
		p = surface_49_nb_E;
		nb = surface_49_nb_reps;
		//sz = surface_49_size;
		}
	else if (q == 53) {
		p = surface_53_nb_E;
		nb = surface_53_nb_reps;
		//sz = surface_53_size;
		}
	else if (q == 59) {
		p = surface_59_nb_E;
		nb = surface_59_nb_reps;
		//sz = surface_59_size;
		}
	else if (q == 61) {
		p = surface_61_nb_E;
		nb = surface_61_nb_reps;
		//sz = surface_61_size;
		}
	else if (q == 64) {
		p = surface_64_nb_E;
		nb = surface_64_nb_reps;
		//sz = surface_64_size;
		}
	else if (q == 67) {
		p = surface_67_nb_E;
		nb = surface_67_nb_reps;
		//sz = surface_67_size;
		}
	else if (q == 71) {
		p = surface_71_nb_E;
		nb = surface_71_nb_reps;
		//sz = surface_71_size;
		}
	else if (q == 73) {
		p = surface_73_nb_E;
		nb = surface_73_nb_reps;
		//sz = surface_73_size;
		}
	else if (q == 79) {
		p = surface_79_nb_E;
		nb = surface_79_nb_reps;
		//sz = surface_79_size;
		}
	else if (q == 81) {
		p = surface_81_nb_E;
		nb = surface_81_nb_reps;
		//sz = surface_81_size;
		}
	else if (q == 83) {
		p = surface_83_nb_E;
		nb = surface_83_nb_reps;
		//sz = surface_83_size;
		}
	else if (q == 89) {
		p = surface_89_nb_E;
		nb = surface_89_nb_reps;
		//sz = surface_89_size;
		}
	else if (q == 97) {
		p = surface_97_nb_E;
		nb = surface_97_nb_reps;
		//sz = surface_97_size;
		}
	else {
		cout << "cubic_surface_nb_Eckardt_points q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "cubic_surface_nb_Eckardt_points q=" << q << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "cubic_surface_nb_Eckardt_points q=" << q << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	return p[i];
}


INT *cubic_surface_single_six(INT q, INT i)
// i starts from 0
{
	INT *p, nb; //, sz;
	if (q == 4) {
		p = surface_4_single_six;
		nb = surface_4_nb_reps;
		//sz = surface_4_size;
		}
	else if (q == 7) {
		p = surface_7_single_six;
		nb = surface_7_nb_reps;
		//sz = surface_7_size;
		}
	else if (q == 8) {
		p = surface_8_single_six;
		nb = surface_8_nb_reps;
		//sz = surface_8_size;
		}
	else if (q == 9) {
		p = surface_9_single_six;
		nb = surface_9_nb_reps;
		//sz = surface_9_size;
		}
	else if (q == 11) {
		p = surface_11_single_six;
		nb = surface_11_nb_reps;
		//sz = surface_11_size;
		}
	else if (q == 13) {
		p = surface_13_single_six;
		nb = surface_13_nb_reps;
		//sz = surface_13_size;
		}
	else if (q == 16) {
		p = surface_16_single_six;
		nb = surface_16_nb_reps;
		//sz = surface_16_size;
		}
	else if (q == 17) {
		p = surface_17_single_six;
		nb = surface_17_nb_reps;
		//sz = surface_17_size;
		}
	else if (q == 19) {
		p = surface_19_single_six;
		nb = surface_19_nb_reps;
		//sz = surface_19_size;
		}
	else if (q == 23) {
		p = surface_23_single_six;
		nb = surface_23_nb_reps;
		//sz = surface_23_size;
		}
	else if (q == 25) {
		p = surface_25_single_six;
		nb = surface_25_nb_reps;
		//sz = surface_25_size;
		}
	else if (q == 27) {
		p = surface_27_single_six;
		nb = surface_27_nb_reps;
		//sz = surface_27_size;
		}
	else if (q == 29) {
		p = surface_29_single_six;
		nb = surface_29_nb_reps;
		//sz = surface_29_size;
		}
	else if (q == 31) {
		p = surface_31_single_six;
		nb = surface_31_nb_reps;
		//sz = surface_31_size;
		}
	else if (q == 32) {
		p = surface_32_single_six;
		nb = surface_32_nb_reps;
		//sz = surface_32_size;
		}
	else if (q == 37) {
		p = surface_37_single_six;
		nb = surface_37_nb_reps;
		//sz = surface_37_size;
		}
	else if (q == 41) {
		p = surface_41_single_six;
		nb = surface_41_nb_reps;
		//sz = surface_41_size;
		}
	else if (q == 43) {
		p = surface_43_single_six;
		nb = surface_43_nb_reps;
		//sz = surface_43_size;
		}
	else if (q == 47) {
		p = surface_47_single_six;
		nb = surface_47_nb_reps;
		//sz = surface_47_size;
		}
	else if (q == 49) {
		p = surface_49_single_six;
		nb = surface_49_nb_reps;
		//sz = surface_49_size;
		}
	else if (q == 53) {
		p = surface_53_single_six;
		nb = surface_53_nb_reps;
		//sz = surface_53_size;
		}
	else if (q == 59) {
		p = surface_59_single_six;
		nb = surface_59_nb_reps;
		//sz = surface_59_size;
		}
	else if (q == 61) {
		p = surface_61_single_six;
		nb = surface_61_nb_reps;
		//sz = surface_61_size;
		}
	else if (q == 64) {
		p = surface_64_single_six;
		nb = surface_64_nb_reps;
		//sz = surface_64_size;
		}
	else if (q == 67) {
		p = surface_67_single_six;
		nb = surface_67_nb_reps;
		//sz = surface_67_size;
		}
	else if (q == 71) {
		p = surface_71_single_six;
		nb = surface_71_nb_reps;
		//sz = surface_71_size;
		}
	else if (q == 73) {
		p = surface_73_single_six;
		nb = surface_73_nb_reps;
		//sz = surface_73_size;
		}
	else if (q == 79) {
		p = surface_79_single_six;
		nb = surface_79_nb_reps;
		//sz = surface_79_size;
		}
	else if (q == 81) {
		p = surface_81_single_six;
		nb = surface_81_nb_reps;
		//sz = surface_81_size;
		}
	else if (q == 83) {
		p = surface_83_single_six;
		nb = surface_83_nb_reps;
		//sz = surface_83_size;
		}
	else if (q == 89) {
		p = surface_89_single_six;
		nb = surface_89_nb_reps;
		//sz = surface_89_size;
		}
	else if (q == 97) {
		p = surface_97_single_six;
		nb = surface_97_nb_reps;
		//sz = surface_97_size;
		}
	else {
		cout << "cubic_surface_single_six q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "cubic_surface_single_six q=" << q << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "cubic_surface_single_six q=" << q << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	return p + i * 6;
}

INT *cubic_surface_Lines(INT q, INT i)
// i starts from 0
{
	INT *p, nb; //, sz;
	if (q == 4) {
		p = surface_4_Lines;
		nb = surface_4_nb_reps;
		//sz = surface_4_size;
		}
	else if (q == 7) {
		p = surface_7_Lines;
		nb = surface_7_nb_reps;
		//sz = surface_7_size;
		}
	else if (q == 8) {
		p = surface_8_Lines;
		nb = surface_8_nb_reps;
		//sz = surface_8_size;
		}
	else if (q == 9) {
		p = surface_9_Lines;
		nb = surface_9_nb_reps;
		//sz = surface_9_size;
		}
	else if (q == 11) {
		p = surface_11_Lines;
		nb = surface_11_nb_reps;
		//sz = surface_11_size;
		}
	else if (q == 13) {
		p = surface_13_Lines;
		nb = surface_13_nb_reps;
		//sz = surface_13_size;
		}
	else if (q == 16) {
		p = surface_16_Lines;
		nb = surface_16_nb_reps;
		//sz = surface_16_size;
		}
	else if (q == 17) {
		p = surface_17_Lines;
		nb = surface_17_nb_reps;
		//sz = surface_17_size;
		}
	else if (q == 19) {
		p = surface_19_Lines;
		nb = surface_19_nb_reps;
		//sz = surface_19_size;
		}
	else if (q == 23) {
		p = surface_23_Lines;
		nb = surface_23_nb_reps;
		//sz = surface_23_size;
		}
	else if (q == 25) {
		p = surface_25_Lines;
		nb = surface_25_nb_reps;
		//sz = surface_25_size;
		}
	else if (q == 27) {
		p = surface_27_Lines;
		nb = surface_27_nb_reps;
		//sz = surface_27_size;
		}
	else if (q == 29) {
		p = surface_29_Lines;
		nb = surface_29_nb_reps;
		//sz = surface_29_size;
		}
	else if (q == 31) {
		p = surface_31_Lines;
		nb = surface_31_nb_reps;
		//sz = surface_31_size;
		}
	else if (q == 32) {
		p = surface_32_Lines;
		nb = surface_32_nb_reps;
		//sz = surface_32_size;
		}
	else if (q == 37) {
		p = surface_37_Lines;
		nb = surface_37_nb_reps;
		//sz = surface_37_size;
		}
	else if (q == 41) {
		p = surface_41_Lines;
		nb = surface_41_nb_reps;
		//sz = surface_41_size;
		}
	else if (q == 43) {
		p = surface_43_Lines;
		nb = surface_43_nb_reps;
		//sz = surface_43_size;
		}
	else if (q == 47) {
		p = surface_47_Lines;
		nb = surface_47_nb_reps;
		//sz = surface_47_size;
		}
	else if (q == 49) {
		p = surface_49_Lines;
		nb = surface_49_nb_reps;
		//sz = surface_49_size;
		}
	else if (q == 53) {
		p = surface_53_Lines;
		nb = surface_53_nb_reps;
		//sz = surface_53_size;
		}
	else if (q == 59) {
		p = surface_59_Lines;
		nb = surface_59_nb_reps;
		//sz = surface_59_size;
		}
	else if (q == 61) {
		p = surface_61_Lines;
		nb = surface_61_nb_reps;
		//sz = surface_61_size;
		}
	else if (q == 64) {
		p = surface_64_Lines;
		nb = surface_64_nb_reps;
		//sz = surface_64_size;
		}
	else if (q == 67) {
		p = surface_67_Lines;
		nb = surface_67_nb_reps;
		//sz = surface_67_size;
		}
	else if (q == 71) {
		p = surface_71_Lines;
		nb = surface_71_nb_reps;
		//sz = surface_71_size;
		}
	else if (q == 73) {
		p = surface_73_Lines;
		nb = surface_73_nb_reps;
		//sz = surface_73_size;
		}
	else if (q == 79) {
		p = surface_79_Lines;
		nb = surface_79_nb_reps;
		//sz = surface_79_size;
		}
	else if (q == 81) {
		p = surface_81_Lines;
		nb = surface_81_nb_reps;
		//sz = surface_81_size;
		}
	else if (q == 83) {
		p = surface_83_Lines;
		nb = surface_83_nb_reps;
		//sz = surface_83_size;
		}
	else if (q == 89) {
		p = surface_89_Lines;
		nb = surface_89_nb_reps;
		//sz = surface_89_size;
		}
	else if (q == 97) {
		p = surface_97_Lines;
		nb = surface_97_nb_reps;
		//sz = surface_97_size;
		}
	else {
		cout << "cubic_surface_Lines q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "cubic_surface_Lines q=" << q << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "cubic_surface_Lines q=" << q << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	return p + i * 27;
}



INT hyperoval_nb_reps(INT q)
{
	INT nb;

	if (q == 8) {
		nb = arcs_8_10_nb_reps;
		}
	else if (q == 16) {
		nb = arcs_16_18_nb_reps;
		}
	else if (q == 32) {
		nb = arcs_32_34_nb_reps;
		}
	else {
		cout << "hyperoval_nb_reps q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

INT *hyperoval_representative(INT q, INT i)
// i starts from 0
{
	INT *p, nb, sz;
	if (q == 8) {
		p = arcs_8_10_reps;
		nb = arcs_8_10_nb_reps;
		sz = arcs_8_10_size;
		}
	else if (q == 16) {
		p = arcs_16_18_reps;
		nb = arcs_16_18_nb_reps;
		sz = arcs_16_18_size;
		}
	else if (q == 32) {
		p = arcs_32_34_reps;
		nb = arcs_32_34_nb_reps;
		sz = arcs_32_34_size;
		}
	else {
		cout << "hyperovals_representative q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "hyperoval_representative q=" << q << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "hyperoval_representative q=" << q << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void hyperoval_gens(INT q, INT i, INT *&data, INT &nb_gens, INT &data_size, const BYTE *&stab_order)
{
	INT *Reps;
	INT nb, make_element_size;
	INT f, l;
	
	if (q == 8) {
		Reps = arcs_8_10_stab_gens;
		nb = arcs_8_10_nb_reps;
		make_element_size = arcs_8_10_make_element_size;
		f = arcs_8_10_stab_gens_fst[i];
		l = arcs_8_10_stab_gens_len[i];
		stab_order = arcs_8_10_stab_order[i];
		}
	else if (q == 16) {
		Reps = arcs_16_18_stab_gens;
		nb = arcs_16_18_nb_reps;
		make_element_size = arcs_16_18_make_element_size;
		f = arcs_16_18_stab_gens_fst[i];
		l = arcs_16_18_stab_gens_len[i];
		stab_order = arcs_16_18_stab_order[i];
		}
	else if (q == 32) {
		Reps = arcs_32_34_stab_gens;
		nb = arcs_32_34_nb_reps;
		make_element_size = arcs_32_34_make_element_size;
		f = arcs_32_34_stab_gens_fst[i];
		l = arcs_32_34_stab_gens_len[i];
		stab_order = arcs_32_34_stab_order[i];
		}
	else {
		cout << "hyperoval_representative q=" << q << " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "hyperoval_representative q=" << q << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "hyperoval_representative q=" << q << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
}




INT DH_nb_reps(INT k, INT n)
{
	INT nb;

	if (k == 4 && n == 7) {
		nb = DH_4_7_nb_reps;
		}
	else if (k == 4 && n == 8) {
		nb = DH_4_8_nb_reps;
		}
	else {
		cout << "DH_nb_reps k=" << k << " n=" << n << " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

INT *DH_representative(INT k, INT n, INT i)
// i starts from 0
{
	INT *p, nb, sz;
	if (k == 4 && n == 7) {
		p = DH_4_7_reps;
		nb = DH_4_7_nb_reps;
		sz = DH_4_7_size;
		}
	else if (k == 4 && n == 8) {
		p = DH_4_8_reps;
		nb = DH_4_8_nb_reps;
		sz = DH_4_8_size;
		}
	else {
		cout << "DH_representative k=" << k << " n=" << n << " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "DH_representative k=" << k << " n=" << n << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "DH_representative k=" << k << " n=" << n << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void DH_stab_gens(INT k, INT n, INT i, INT *&data, INT &nb_gens, INT &data_size, const BYTE *&stab_order)
{
	INT *Reps;
	INT nb, make_element_size;
	INT f, l;
	
	if (k == 4 && n == 7) {
		Reps = DH_4_7_stab_gens;
		nb = DH_4_7_nb_reps;
		make_element_size = DH_4_7_make_element_size;
		f = DH_4_7_stab_gens_fst[i];
		l = DH_4_7_stab_gens_len[i];
		stab_order = DH_4_7_stab_order[i];
		}
	else if (k == 4 && n == 8) {
		Reps = DH_4_8_stab_gens;
		nb = DH_4_8_nb_reps;
		make_element_size = DH_4_8_make_element_size;
		f = DH_4_8_stab_gens_fst[i];
		l = DH_4_8_stab_gens_len[i];
		stab_order = DH_4_8_stab_order[i];
		}
	else {
		cout << "DH_representative k=" << k << " n=" << n << " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "DH_representative k=" << k << " n=" << n << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "DH_representative k=" << k << " n=" << n << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
}









INT TP_nb_reps(INT q, INT k)
{
	INT nb;

	if (q == 2 && k == 2) {
		nb = TP_2_2_nb_reps;
		}
	else if (q == 3 && k == 2) {
		nb = TP_3_2_nb_reps;
		}
	else if (q == 2 && k == 4) {
		nb = TP_2_4_nb_reps;
		}
	else if (q == 4 && k == 2) {
		nb = TP_4_2_nb_reps;
		}
	else if (q == 5 && k == 2) {
		nb = TP_5_2_nb_reps;
		}
	else if (q == 3 && k == 3) {
		nb = TP_3_3_nb_reps;
		}
	else {
		cout << "TP_nb_reps q=" << q << " k=" << k << " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}


INT *TP_representative(INT q, INT k, INT i)
// i starts from 0
{
	INT *p, nb, sz;

	if (q == 2 && k == 2) {
		p = TP_2_2_reps;
		nb = TP_2_2_nb_reps;
		sz = TP_2_2_size;
		}
	else if (q == 3 && k == 2) {
		p = TP_3_2_reps;
		nb = TP_3_2_nb_reps;
		sz = TP_3_2_size;
		}
	else if (q == 2 && k == 4) {
		p = TP_2_4_reps;
		nb = TP_2_4_nb_reps;
		sz = TP_2_4_size;
		}
	else if (q == 4 && k == 2) {
		p = TP_4_2_reps;
		nb = TP_4_2_nb_reps;
		sz = TP_4_2_size;
		}
	else if (q == 5 && k == 2) {
		p = TP_5_2_reps;
		nb = TP_5_2_nb_reps;
		sz = TP_5_2_size;
		}
	else if (q == 3 && k == 3) {
		p = TP_3_3_reps;
		nb = TP_3_3_nb_reps;
		sz = TP_3_3_size;
		}
	else {
		cout << "TP_representative q=" << q << " k=" << k << " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "TP_representative q=" << q << " k=" << k << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "TP_representative q=" << q << " k=" << k << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void TP_stab_gens(INT q, INT k, INT i, INT *&data, INT &nb_gens, INT &data_size, const BYTE *&stab_order)
{
	INT *Reps;
	INT nb, make_element_size;
	INT f, l;
	
	if (q == 2 && k == 2) {
		Reps = TP_2_2_stab_gens;
		nb = TP_2_2_nb_reps;
		make_element_size = TP_2_2_make_element_size;
		f = TP_2_2_stab_gens_fst[i];
		l = TP_2_2_stab_gens_len[i];
		stab_order = TP_2_2_stab_order[i];
		}
	else if (q == 3 && k == 2) {
		Reps = TP_3_2_stab_gens;
		nb = TP_3_2_nb_reps;
		make_element_size = TP_3_2_make_element_size;
		f = TP_3_2_stab_gens_fst[i];
		l = TP_3_2_stab_gens_len[i];
		stab_order = TP_3_2_stab_order[i];
		}
	else if (q == 2 && k == 4) {
		Reps = TP_2_4_stab_gens;
		nb = TP_2_4_nb_reps;
		make_element_size = TP_2_4_make_element_size;
		f = TP_2_4_stab_gens_fst[i];
		l = TP_2_4_stab_gens_len[i];
		stab_order = TP_2_4_stab_order[i];
		}
	else if (q == 4 && k == 2) {
		Reps = TP_4_2_stab_gens;
		nb = TP_4_2_nb_reps;
		make_element_size = TP_4_2_make_element_size;
		f = TP_4_2_stab_gens_fst[i];
		l = TP_4_2_stab_gens_len[i];
		stab_order = TP_4_2_stab_order[i];
		}
	else if (q == 5 && k == 2) {
		Reps = TP_5_2_stab_gens;
		nb = TP_5_2_nb_reps;
		make_element_size = TP_5_2_make_element_size;
		f = TP_5_2_stab_gens_fst[i];
		l = TP_5_2_stab_gens_len[i];
		stab_order = TP_5_2_stab_order[i];
		}
	else if (q == 3 && k == 3) {
		Reps = TP_3_3_stab_gens;
		nb = TP_3_3_nb_reps;
		make_element_size = TP_3_3_make_element_size;
		f = TP_3_3_stab_gens_fst[i];
		l = TP_3_3_stab_gens_len[i];
		stab_order = TP_3_3_stab_order[i];
		}
	else {
		cout << "TP_representative q=" << q << " k=" << k << " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "TP_representative q=" << q << " k=" << k << " i=" << i << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "TP_representative q=" << q << " k=" << k << " i=" << i << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
}






INT BLT_nb_reps(INT q, INT k)
{
	INT nb;

	if (q == 3) {
		nb = BLT_3_nb_reps;
		}
	else if (q == 5) {
		nb = BLT_5_nb_reps;
		}
	else if (q == 7) {
		nb = BLT_7_nb_reps;
		}
	else if (q == 9) {
		nb = BLT_9_nb_reps;
		}
	else if (q == 11) {
		nb = BLT_11_nb_reps;
		}
	else if (q == 13) {
		nb = BLT_13_nb_reps;
		}
	else if (q == 17) {
		nb = BLT_17_nb_reps;
		}
	else if (q == 19) {
		nb = BLT_19_nb_reps;
		}
	else if (q == 23) {
		nb = BLT_23_nb_reps;
		}
	else if (q == 25) {
		nb = BLT_25_nb_reps;
		}
	else if (q == 27) {
		nb = BLT_27_nb_reps;
		}
	else if (q == 29) {
		nb = BLT_29_nb_reps;
		}
	else if (q == 31) {
		nb = BLT_31_nb_reps;
		}
	else if (q == 37) {
		nb = BLT_37_nb_reps;
		}
	else if (q == 41) {
		nb = BLT_41_nb_reps;
		}
	else if (q == 43) {
		nb = BLT_43_nb_reps;
		}
	else if (q == 47) {
		nb = BLT_47_nb_reps;
		}
	else if (q == 49) {
		nb = BLT_49_nb_reps;
		}
	else if (q == 53) {
		nb = BLT_53_nb_reps;
		}
	else {
		cout << "BLT_nb_reps q=" << q << " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

INT *BLT_representative(INT q, INT no)
// i starts from 0
{
	INT *p, nb, sz;

	if (q == 3) {
		p = BLT_3_reps;
		nb = BLT_3_nb_reps;
		sz = BLT_3_size;
		}
	else if (q == 5) {
		p = BLT_5_reps;
		nb = BLT_5_nb_reps;
		sz = BLT_5_size;
		}
	else if (q == 7) {
		p = BLT_7_reps;
		nb = BLT_7_nb_reps;
		sz = BLT_7_size;
		}
	else if (q == 9) {
		p = BLT_9_reps;
		nb = BLT_9_nb_reps;
		sz = BLT_9_size;
		}
	else if (q == 11) {
		p = BLT_11_reps;
		nb = BLT_11_nb_reps;
		sz = BLT_11_size;
		}
	else if (q == 13) {
		p = BLT_13_reps;
		nb = BLT_13_nb_reps;
		sz = BLT_13_size;
		}
	else if (q == 17) {
		p = BLT_17_reps;
		nb = BLT_17_nb_reps;
		sz = BLT_17_size;
		}
	else if (q == 19) {
		p = BLT_19_reps;
		nb = BLT_19_nb_reps;
		sz = BLT_19_size;
		}
	else if (q == 23) {
		p = BLT_23_reps;
		nb = BLT_23_nb_reps;
		sz = BLT_23_size;
		}
	else if (q == 25) {
		p = BLT_25_reps;
		nb = BLT_25_nb_reps;
		sz = BLT_25_size;
		}
	else if (q == 27) {
		p = BLT_27_reps;
		nb = BLT_27_nb_reps;
		sz = BLT_27_size;
		}
	else if (q == 29) {
		p = BLT_29_reps;
		nb = BLT_29_nb_reps;
		sz = BLT_29_size;
		}
	else if (q == 31) {
		p = BLT_31_reps;
		nb = BLT_31_nb_reps;
		sz = BLT_31_size;
		}
	else if (q == 37) {
		p = BLT_37_reps;
		nb = BLT_37_nb_reps;
		sz = BLT_37_size;
		}
	else if (q == 41) {
		p = BLT_41_reps;
		nb = BLT_41_nb_reps;
		sz = BLT_41_size;
		}
	else if (q == 43) {
		p = BLT_43_reps;
		nb = BLT_43_nb_reps;
		sz = BLT_43_size;
		}
	else if (q == 47) {
		p = BLT_47_reps;
		nb = BLT_47_nb_reps;
		sz = BLT_47_size;
		}
	else if (q == 49) {
		p = BLT_49_reps;
		nb = BLT_49_nb_reps;
		sz = BLT_49_size;
		}
	else if (q == 53) {
		p = BLT_53_reps;
		nb = BLT_53_nb_reps;
		sz = BLT_53_size;
		}
	else {
		cout << "BLT_representative q=" << q << " I don't have information for this field order" << endl;
		exit(1);
		}
	if (no < 0) {
		cout << "BLT_representative q=" << q << " no=" << no << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (no >= nb) {
		cout << "BLT_representative q=" << q << " no=" << no << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += no * sz;
	return p;
}

void BLT_stab_gens(INT q, INT k, INT no, INT *&data, INT &nb_gens, INT &data_size, const BYTE *&stab_order)
{
	INT *Reps;
	INT nb, make_element_size;
	INT f, l;
	
	if (q == 3) {
		Reps = BLT_3_stab_gens;
		nb = BLT_3_nb_reps;
		make_element_size = BLT_3_make_element_size;
		f = BLT_3_stab_gens_fst[no];
		l = BLT_3_stab_gens_len[no];
		stab_order = BLT_3_stab_order[no];
		}
	else if (q == 5) {
		Reps = BLT_5_stab_gens;
		nb = BLT_5_nb_reps;
		make_element_size = BLT_5_make_element_size;
		f = BLT_5_stab_gens_fst[no];
		l = BLT_5_stab_gens_len[no];
		stab_order = BLT_5_stab_order[no];
		}
	else if (q == 7) {
		Reps = BLT_7_stab_gens;
		nb = BLT_7_nb_reps;
		make_element_size = BLT_7_make_element_size;
		f = BLT_7_stab_gens_fst[no];
		l = BLT_7_stab_gens_len[no];
		stab_order = BLT_7_stab_order[no];
		}
	else if (q == 9) {
		Reps = BLT_9_stab_gens;
		nb = BLT_9_nb_reps;
		make_element_size = BLT_9_make_element_size;
		f = BLT_9_stab_gens_fst[no];
		l = BLT_9_stab_gens_len[no];
		stab_order = BLT_9_stab_order[no];
		}
	else if (q == 11) {
		Reps = BLT_11_stab_gens;
		nb = BLT_11_nb_reps;
		make_element_size = BLT_11_make_element_size;
		f = BLT_11_stab_gens_fst[no];
		l = BLT_11_stab_gens_len[no];
		stab_order = BLT_11_stab_order[no];
		}
	else if (q == 13) {
		Reps = BLT_13_stab_gens;
		nb = BLT_13_nb_reps;
		make_element_size = BLT_13_make_element_size;
		f = BLT_13_stab_gens_fst[no];
		l = BLT_13_stab_gens_len[no];
		stab_order = BLT_13_stab_order[no];
		}
	else if (q == 17) {
		Reps = BLT_17_stab_gens;
		nb = BLT_17_nb_reps;
		make_element_size = BLT_17_make_element_size;
		f = BLT_17_stab_gens_fst[no];
		l = BLT_17_stab_gens_len[no];
		stab_order = BLT_17_stab_order[no];
		}
	else if (q == 19) {
		Reps = BLT_19_stab_gens;
		nb = BLT_19_nb_reps;
		make_element_size = BLT_19_make_element_size;
		f = BLT_19_stab_gens_fst[no];
		l = BLT_19_stab_gens_len[no];
		stab_order = BLT_19_stab_order[no];
		}
	else if (q == 23) {
		Reps = BLT_23_stab_gens;
		nb = BLT_23_nb_reps;
		make_element_size = BLT_23_make_element_size;
		f = BLT_23_stab_gens_fst[no];
		l = BLT_23_stab_gens_len[no];
		stab_order = BLT_23_stab_order[no];
		}
	else if (q == 25) {
		Reps = BLT_25_stab_gens;
		nb = BLT_25_nb_reps;
		make_element_size = BLT_25_make_element_size;
		f = BLT_25_stab_gens_fst[no];
		l = BLT_25_stab_gens_len[no];
		stab_order = BLT_25_stab_order[no];
		}
	else if (q == 27) {
		Reps = BLT_27_stab_gens;
		nb = BLT_27_nb_reps;
		make_element_size = BLT_27_make_element_size;
		f = BLT_27_stab_gens_fst[no];
		l = BLT_27_stab_gens_len[no];
		stab_order = BLT_27_stab_order[no];
		}
	else if (q == 29) {
		Reps = BLT_29_stab_gens;
		nb = BLT_29_nb_reps;
		make_element_size = BLT_29_make_element_size;
		f = BLT_29_stab_gens_fst[no];
		l = BLT_29_stab_gens_len[no];
		stab_order = BLT_29_stab_order[no];
		}
	else if (q == 31) {
		Reps = BLT_31_stab_gens;
		nb = BLT_31_nb_reps;
		make_element_size = BLT_31_make_element_size;
		f = BLT_31_stab_gens_fst[no];
		l = BLT_31_stab_gens_len[no];
		stab_order = BLT_31_stab_order[no];
		}
	else if (q == 37) {
		Reps = BLT_37_stab_gens;
		nb = BLT_37_nb_reps;
		make_element_size = BLT_37_make_element_size;
		f = BLT_37_stab_gens_fst[no];
		l = BLT_37_stab_gens_len[no];
		stab_order = BLT_37_stab_order[no];
		}
	else if (q == 41) {
		Reps = BLT_41_stab_gens;
		nb = BLT_41_nb_reps;
		make_element_size = BLT_41_make_element_size;
		f = BLT_41_stab_gens_fst[no];
		l = BLT_41_stab_gens_len[no];
		stab_order = BLT_41_stab_order[no];
		}
	else if (q == 43) {
		Reps = BLT_43_stab_gens;
		nb = BLT_43_nb_reps;
		make_element_size = BLT_43_make_element_size;
		f = BLT_43_stab_gens_fst[no];
		l = BLT_43_stab_gens_len[no];
		stab_order = BLT_43_stab_order[no];
		}
	else if (q == 47) {
		Reps = BLT_47_stab_gens;
		nb = BLT_47_nb_reps;
		make_element_size = BLT_47_make_element_size;
		f = BLT_47_stab_gens_fst[no];
		l = BLT_47_stab_gens_len[no];
		stab_order = BLT_47_stab_order[no];
		}
	else if (q == 49) {
		Reps = BLT_49_stab_gens;
		nb = BLT_49_nb_reps;
		make_element_size = BLT_49_make_element_size;
		f = BLT_49_stab_gens_fst[no];
		l = BLT_49_stab_gens_len[no];
		stab_order = BLT_49_stab_order[no];
		}
	else if (q == 53) {
		Reps = BLT_53_stab_gens;
		nb = BLT_53_nb_reps;
		make_element_size = BLT_53_make_element_size;
		f = BLT_53_stab_gens_fst[no];
		l = BLT_53_stab_gens_len[no];
		stab_order = BLT_53_stab_order[no];
		}
	else {
		cout << "BLT_representative q=" << q << " I don't have information for this field order" << endl;
		exit(1);
		}
	if (no < 0) {
		cout << "BLT_representative q=" << q << " no=" << no << " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (no >= nb) {
		cout << "BLT_representative q=" << q << " no=" << no << " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
}




const BYTE *override_polynomial_subfield(INT q)
{
	const BYTE *override_poly = NULL;
	INT p, h;
	
	if (!is_prime_power(q, p, h)) {
		cout << "override_polynomial_subfield q is not a prime power" << endl;
		exit(1);
		}
	if (h == 1) {
		return NULL;
		}
	if (q == 8) {
		override_poly = "13"; // Warning !!!
		}
	else if (q == 9) {
		override_poly = "17";
		}
	else if (q == 25) {
		override_poly = "47";
		}
	else if (q == 27) {
		override_poly = "34";
		}
	else if (q == 49) {
		override_poly = "94";
		}
	else if (q == 81) {
		override_poly = "89";
		}
	else if (q == 121) {
		override_poly = "200";
		}
	if (override_poly == NULL) {
		cout << "override_polynomial_subfield, do not have a polynomial for q=" << q << endl;
		
		INT verbose_level = 2;
		finite_field f, F;
		INT qq = q * q;
		
		cout << "initializing large field" << endl;
		F.init(qq, verbose_level);
		cout << "initializing small field" << endl;
		f.init(q, verbose_level);
		if (f.e > 1) {
			F.init(qq, 1);
			f.init(q, 3);
			cout << "need to choose the generator polynomial for the field" << endl;
			F.compute_subfields(verbose_level);
			//exit(1);
			}


		return NULL;
		}
	return override_poly;
}

const BYTE *override_polynomial_extension_field(INT q)
{
	const BYTE *override_poly = NULL;
	INT p, h;
	
	if (!is_prime_power(q, p, h)) {
		cout << "override_polynomial_extension_field q is not a prime power" << endl;
		exit(1);
		}
	if (h == 1) {
		return get_primitive_polynomial(q, 2, 0/*verbose_level*/);
		}
#if 0
	if (h == 1) {
		return NULL;
		}
#endif
	if (q == 9) {
		override_poly = "110";
		}
	else if (q == 25) {
		override_poly = "767";
		}
	else if (q == 27) {
		override_poly = "974";
		}
	else if (q == 49) {
		override_poly = "2754";
		}
	else if (q == 81) {
		override_poly = "6590";
		}
	else if (q == 121) {
		override_poly = "15985";
		}
	if (override_poly == NULL) {
		cout << "override_polynomial_extension_field, do not have a polynomial for q=" << q << endl;
		exit(1);
		}
	return override_poly;
}

#if 0
	if (q == 9) {
		BYTE *override_poly_Q = "110"; // X^{4} + X^{3} + 2
		BYTE *override_poly_q = "17"; // X^2 - X - 1 = X^2 +2X + 2 = 2 + 2*3 + 9 = 17
		//finite_field::init_override_polynomial() GF(81) = GF(3^4), polynomial = X^{4} + X^{3} + 2 = 110
		//subfields of F_{81}:
		//subfield 3^2 : subgroup_index = 10
		//0 : 0 : 1 : 1
		//1 : 10 : 46 : X^{3} + 2X^{2} + 1
		//2 : 20 : 47 : X^{3} + 2X^{2} + 2
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 25) {
		BYTE *override_poly_Q = "767"; // X^{4} + X^{3} + 3X + 2
		BYTE *override_poly_q = "47"; // X^2 - X - 3 = X^2 +4X + 2=25+20+2=47
		//subfields of F_{625}:
		//subfield 5^2 : subgroup_index = 26
		//0 : 0 : 1 : 1
		//1 : 26 : 110 : 4X^{2} + 2X
		//2 : 52 : 113 : 4X^{2} + 2X + 3
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 27) {
		BYTE *override_poly_Q = "974"; // X^{6} + X^{5} + 2
		BYTE *override_poly_q = "34"; // X^3 - X + 1 = X^3 +2X + 1 = 27+6+1=34
		//subfields of F_{729}:
		//subfield 3^2 : subgroup_index = 91
		//0 : 0 : 1 : 1
		//1 : 91 : 599 : 2X^{5} + X^{4} + X^{3} + X + 2
		//2 : 182 : 597 : 2X^{5} + X^{4} + X^{3} + X
		//subfield 3^3 : subgroup_index = 28
		//0 : 0 : 1 : 1
		//1 : 28 : 158 : X^{4} + 2X^{3} + 2X^{2} + X + 2
		//2 : 56 : 498 : 2X^{5} + X^{2} + X
		//3 : 84 : 157 : X^{4} + 2X^{3} + 2X^{2} + X + 1
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 49) {
		BYTE *override_poly_Q = "2754"; // X^{4} + X^{3} + X + 3
		BYTE *override_poly_q = "94"; // X^2-X+3 = X^2+6X+3 = 49+6*7+3=94
		//subfields of F_{2401}:
		//subfield 7^2 : subgroup_index = 50
		//0 : 0 : 1 : 1
		//1 : 50 : 552 : X^{3} + 4X^{2} + X + 6
		//2 : 100 : 549 : X^{3} + 4X^{2} + X + 3
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 81) {
		BYTE *override_poly_Q = "6590"; // X^{8} + X^{3} + 2
		BYTE *override_poly_q = "89"; // X^4-X-1=X^4+2X+2=81+2*3+2=89
		//subfields of F_{6561}:
		//subfield 3^4 : subgroup_index = 82
		//0 : 0 : 1 : 1
		//1 : 82 : 5413 : 2X^{7} + X^{6} + X^{5} + 2X^{3} + X^{2} + X + 1
		//2 : 164 : 1027 : X^{6} + X^{5} + 2X^{3} + 1
		//3 : 246 : 3976 : X^{7} + 2X^{6} + X^{5} + X^{4} + 2X + 1
		//4 : 328 : 5414 : 2X^{7} + X^{6} + X^{5} + 2X^{3} + X^{2} + X + 2
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
	else if (q == 121) {
		BYTE *override_poly_Q = "15985"; // X^{4} + X^{3} + X + 2
		BYTE *override_poly_q = "200"; // X^2-4X+2=X^2+7X+2=11^2+7*11+2=200
		//subfields of F_{14641}:
		//subfield 11^2 : subgroup_index = 122
		//0 : 0 : 1 : 1
		//1 : 122 : 4352 : 3X^{3} + 2X^{2} + 10X + 7
		//2 : 244 : 2380 : X^{3} + 8X^{2} + 7X + 4
		F.init_override_polynomial(Q, override_poly_Q, verbose_level - 2);
		cout << "field of order " << Q << " initialized" << endl;
		f.init_override_polynomial(q, override_poly_q, verbose_level - 2);
		}
#endif

void create_Fisher_BLT_set(INT *Fisher_BLT, INT q, const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	unusual_model U;
	
	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Fisher_BLT_set(Fisher_BLT, verbose_level);
	
}

void create_Linear_BLT_set(INT *BLT, INT q, const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	unusual_model U;
	
	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Linear_BLT_set(BLT, verbose_level);
	
}

void create_Mondello_BLT_set(INT *BLT, INT q, const BYTE *poly_q, const BYTE *poly_Q, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	unusual_model U;
	
	U.setup(q, poly_q, poly_Q, verbose_level);
	U.create_Mondello_BLT_set(BLT, verbose_level);
	
}

void print_quadratic_form_list_coded(INT form_nb_terms, 
	INT *form_i, INT *form_j, INT *form_coeff)
{
	INT k;
	
	for (k = 0; k < form_nb_terms; k++) {
		cout << "i=" << form_i[k] << " j=" << form_j[k] << " coeff=" << form_coeff[k] << endl;
		}
}

void make_Gram_matrix_from_list_coded_quadratic_form(INT n, finite_field &F, 
	INT nb_terms, INT *form_i, INT *form_j, INT *form_coeff, INT *Gram)
{
	INT k, i, j, c;
	
	INT_vec_zero(Gram, n * n);
#if 0
	for (i = 0; i < n * n; i++)
		Gram[i] = 0;
#endif
	for (k = 0; k < nb_terms; k++) {
		i = form_i[k];
		j = form_j[k];
		c = form_coeff[k];
		if (c == 0) {
			continue;
			}
		Gram[i * n + j] = F.add(Gram[i * n + j], c);
		Gram[j * n + i] = F.add(Gram[j * n + i], c);
		}
}

void add_term(INT n, finite_field &F, INT &nb_terms, INT *form_i, INT *form_j, INT *form_coeff, INT *Gram, 
	INT i, INT j, INT coeff)
{
	form_i[nb_terms] = i;
	form_j[nb_terms] = j;
	form_coeff[nb_terms] = coeff;
	if (i == j) {
		Gram[i * n + j] = F.mult(2, coeff);
		}
	else {
		Gram[i * n + j] = coeff;
		Gram[j * n + i] = coeff;
		}
	nb_terms++;
}

void create_BLT_point(finite_field *F, INT *v5, INT a, INT b, INT c, INT verbose_level)
// creates the point (-b/2,-c,a,-(b^2/4-ac),1) 
// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
// b^2/4 + (-c)*a + -(b^2/4-ac)
// = b^2/4 -ac -b^2/4 + ac = 0
{
	INT f_v = (verbose_level >= 1);
	INT v0, v1, v2, v3, v4;
	INT half, four, quarter, minus_one;
	
	if (f_v) {
		cout << "create_BLT_point" << endl;
		}
	four = 4 % F->p;
	half = F->inverse(2);
	quarter = F->inverse(four);
	minus_one = F->negate(1);
	if (f_v) {
		cout << "create_BLT_point four=" << four << endl;
		cout << "create_BLT_point half=" << half << endl;
		cout << "create_BLT_point quarter=" << quarter << endl;
		cout << "create_BLT_point minus_one=" << minus_one << endl;
		}

	v0 = F->mult(minus_one, F->mult(b, half));
	v1 = F->mult(minus_one, c);
	v2 = a;
	v3 = F->mult(minus_one, F->add(F->mult(F->mult(b, b), quarter), F->negate(F->mult(a, c))));
	v4 = 1;
	INT_vec_init5(v5, v0, v1, v2, v3, v4);
	if (f_v) {
		cout << "create_BLT_point done" << endl;
		}

}

void create_FTWKB_BLT_set(orthogonal *O, INT *set, INT verbose_level)
// for q congruent 2 mod 3
// a(t)= t, b(t) = 3*t^2, c(t) = 3*t^3, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT v[5];
	INT r, i, a, b, c;
	finite_field *F;
	
	F = O->F;
	INT q = F->q;
	if (q <= 5) {
		cout << "create_FTWKB_BLT_set q <= 5" << endl;
		exit(1);
		}
	r = q % 3;
	if (r != 2) {
		cout << "create_FTWKB_BLT_set q mod 3 must be 2" << endl;
		exit(1);
		}
	for (i = 0; i < q; i++) {
		a = i;
		b = F->mult(3, F->power(i, 2));
		c = F->mult(3, F->power(i, 3));
		if (f_vv) {
			cout << "i=" << i << " a=" << a << " b=" << b << " c=" << c << endl;
			}
		create_BLT_point(F, v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			INT_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = O->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	INT_vec_init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		INT_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = O->rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "the BLT set FTWKB is ";
		INT_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void create_K1_BLT_set(orthogonal *O, INT *set, INT verbose_level)
// for a nonsquare m, and q=p^e
// a(t)= t, b(t) = 0, c(t) = -m*t^p, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT v[5];
	INT i, m, minus_one, exponent, a, b, c;
	finite_field *F;
	INT q;
	
	F = O->F;
	q = F->q;
	m = F->p; // the primitive element is a nonsquare
	exponent = F->p;
	minus_one = F->negate(1);
	if (f_v) {
		cout << "m=" << m << endl;
		cout << "exponent=" << exponent << endl;
		cout << "minus_one=" << minus_one << endl;
		}
	for (i = 0; i < q; i++) {
		a = i;
		b = 0;
		c = F->mult(minus_one, F->mult(m, F->power(i, exponent)));
		if (f_vv) {
			cout << "i=" << i << " a=" << a << " b=" << b << " c=" << c << endl;
			}
		create_BLT_point(F, v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			INT_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = O->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	INT_vec_init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		INT_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = O->rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "the BLT set K1 is ";
		INT_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void create_K2_BLT_set(orthogonal *O, INT *set, INT verbose_level)
// for q congruent 2 or 3 mod 5
// a(t)= t, b(t) = 5*t^3, c(t) = 5*t^5, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT v[5];
	INT five, r, i, a, b, c;
	finite_field *F;
	INT q;
	
	F = O->F;
	q = F->q;
	if (q <= 5) {
		cout << "create_K2_BLT_set q <= 5" << endl;
		return;
		}
	r = q % 5;
	if (r != 2 && r != 3) {
		cout << "create_K2_BLT_set q mod 5 must be 2 or 3" << endl;
		return;
		}
	five = 5 % F->p;
	for (i = 0; i < q; i++) {
		a = i;
		b = F->mult(five, F->power(i, 3));
		c = F->mult(five, F->power(i, 5));
		if (f_vv) {
			cout << "i=" << i << " a=" << a << " b=" << b << " c=" << c << endl;
			}
		create_BLT_point(F, v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			INT_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = O->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	INT_vec_init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		INT_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = O->rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "the BLT set K2 is ";
		INT_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void create_LP_37_72_BLT_set(orthogonal *O, INT *set, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT v[5], v0, v1, v2, v3, v4;
	INT i;
	INT coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,20,1,33,5,
		1,6,23,19,23,
		1,32,11,35,17,
		1,33,12,14,23,
		1,25,8,12,6,
		1,16,6,1,22,
		1,23,8,5,6,
		1,8,6,13,8,
		1,22,19,20,13,
		1,21,23,16,23,
		1,28,6,9,8,
		1,2,26,7,13,
		1,5,9,36,35,
		1,12,23,10,17,
		1,14,16,25,23,
		1,9,8,26,35,
		1,1,11,8,19,
		1,19,12,11,17,
		1,18,27,22,22,
		1,24,36,17,35,
		1,26,27,23,5,
		1,27,25,24,22,
		1,36,21,32,35,
		1,7,16,31,8,
		1,35,5,15,5,
		1,10,36,6,13,
		1,30,4,3,5,
		1,4,3,30,19,
		1,17,13,2,19,
		1,11,28,18,17,
		1,13,16,27,22,
		1,29,12,28,6,
		1,15,10,34,19,
		1,3,30,4,13,
		1,31,9,21,8,
		1,34,9,29,6
		};
	finite_field *F;
	INT q;
	
	F = O->F;
	q = F->q;
	if (q != 37) {
		cout << "create_LP_37_72_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		INT_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			INT_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = O->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "the BLT set LP_37_72 is ";
		INT_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void create_LP_37_4a_BLT_set(orthogonal *O, INT *set, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT v[5], v0, v1, v2, v3, v4;
	INT i;
	INT coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,9,16,8,5,
		1,13,20,26,2,
		1,4,12,14,22,
		1,19,23,5,5,
		1,24,17,19,32,
		1,18,18,10,14,
		1,2,4,36,23,
		1,7,5,24,29,
		1,36,20,22,29,
		1,14,10,13,14,
		1,28,22,7,23,
		1,32,28,20,19,
		1,30,27,23,24,
		1,3,30,28,15,
		1,1,20,31,13,
		1,11,36,33,6,
		1,29,22,30,15,
		1,20,10,4,5,
		1,8,14,32,29,
		1,25,15,9,31,
		1,26,13,18,29,
		1,23,19,6,19,
		1,35,11,15,20,
		1,22,11,25,32,
		1,10,16,2,20,
		1,17,18,27,31,
		1,15,29,16,29,
		1,31,18,1,15,
		1,12,34,35,15,
		1,33,23,17,20,
		1,27,23,21,14,
		1,34,22,3,6,
		1,21,11,11,18,
		1,5,33,12,35,
		1,6,22,34,15,
		1,16,31,29,18
		};
	finite_field *F;
	INT q;
	
	F = O->F;
	q = F->q;
	if (q != 37) {
		cout << "create_LP_37_4a_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		INT_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			INT_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = O->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "the BLT set LP_37_4a is ";
		INT_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void create_LP_37_4b_BLT_set(orthogonal *O, INT *set, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT v[5], v0, v1, v2, v3, v4;
	INT i;
	INT coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,3,7,25,24,
		1,35,30,32,15,
		1,4,10,30,2,
		1,14,8,17,31,
		1,30,18,2,23,
		1,19,0,10,32,
		1,8,18,12,24,
		1,34,2,20,19,
		1,28,34,15,15,
		1,2,21,23,31,
		1,13,29,36,23,
		1,23,13,8,17,
		1,25,12,35,17,
		1,1,14,4,22,
		1,17,2,19,6,
		1,12,17,1,32,
		1,27,23,3,19,
		1,20,2,21,20,
		1,33,30,22,2,
		1,11,16,31,32,
		1,29,6,13,31,
		1,16,17,7,6,
		1,6,25,14,31,
		1,32,27,29,8,
		1,15,8,9,23,
		1,5,17,24,35,
		1,18,13,33,14,
		1,7,36,26,2,
		1,21,34,28,32,
		1,10,22,16,22,
		1,26,34,27,29,
		1,31,13,34,35,
		1,9,13,18,2,
		1,22,28,5,31,
		1,24,3,11,23,
		1,36,27,6,17
		};
	finite_field *F;
	INT q;
	
	F = O->F;
	q = F->q;
	if (q != 37) {
		cout << "create_LP_37_4b_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		INT_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			INT_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = O->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "the BLT set LP_37_4b is ";
		INT_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void Segre_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q = F->q;
	INT e = F->e;
	INT N = q + 2;
	INT i, t, a, t6;
	INT *Mtx;

	if (f_v) {
		cout << "Segre_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "Segre_hyperoval needs e odd" << endl;
		exit(1);
		}

	nb_pts = N;

	Pts = NEW_INT(N);
	Mtx = NEW_INT(N * 3);
	INT_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t6 = F->power(t, 6);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = t6;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(*F, Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_INT(Mtx);
	if (f_v) {
		cout << "Segre_hyperoval q=" << q << " done" << endl;
		}
}


void GlynnI_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q = F->q;
	INT e = F->e;
	INT N = q + 2;
	INT i, t, te, a;
	INT sigma, gamma = 0, Sigma, /*Gamma,*/ exponent;
	INT *Mtx;

	if (f_v) {
		cout << "GlynnI_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "GlynnI_hyperoval needs e odd" << endl;
		exit(1);
		}

	sigma = e - 1;
	for (i = 0; i < e; i++) {
		if (((i * i) % e) == sigma) {
			gamma = i;
			break;
			}
		}
	if (i == e) {
		cout << "GlynnI_hyperoval did not find gamma" << endl;
		exit(1);
		}

	cout << "GlynnI_hyperoval sigma = " << sigma << " gamma = " << gamma << endl;
	//Gamma = i_power_j(2, gamma);
	Sigma = i_power_j(2, sigma);

	exponent = 3 * Sigma + 4;

	nb_pts = N;

	Pts = NEW_INT(N);
	Mtx = NEW_INT(N * 3);
	INT_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		te = F->power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(*F, Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_INT(Mtx);
	if (f_v) {
		cout << "GlynnI_hyperoval q=" << q << " done" << endl;
		}
}

void GlynnII_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q = F->q;
	INT e = F->e;
	INT N = q + 2;
	INT i, t, te, a;
	INT sigma, gamma = 0, Sigma, Gamma, exponent;
	INT *Mtx;

	if (f_v) {
		cout << "GlynnII_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "GlynnII_hyperoval needs e odd" << endl;
		exit(1);
		}

	sigma = e - 1;
	for (i = 0; i < e; i++) {
		if (((i * i) % e) == sigma) {
			gamma = i;
			break;
			}
		}
	if (i == e) {
		cout << "GlynnII_hyperoval did not find gamma" << endl;
		exit(1);
		}

	cout << "GlynnI_hyperoval sigma = " << sigma << " gamma = " << i << endl;
	Gamma = i_power_j(2, gamma);
	Sigma = i_power_j(2, sigma);

	exponent = Sigma + Gamma;

	nb_pts = N;

	Pts = NEW_INT(N);
	Mtx = NEW_INT(N * 3);
	INT_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		te = F->power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(*F, Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_INT(Mtx);
	if (f_v) {
		cout << "GlynnII_hyperoval q=" << q << " done" << endl;
		}
}



//Date: Tue, 30 Dec 2014 21:08:19 -0700
//From: Tim Penttila 

//To: "betten@math.colostate.edu" <betten@math.colostate.edu>
//Subject: RE: Oops
//Parts/Attachments:
//   1   OK    ~3 KB     Text
//   2 Shown   ~4 KB     Text
//----------------------------------------
//
//Hi Anton,
//
//Friday is predicted to be 42 Celsius, here in Adelaide. So you are
//right! (And I do like that!)
//
//Let b be an element of GF(q^2) of relative norm 1 over GF(q),i.e, b is
//different from 1 but b^{q+1} = 1 . Consider the polynomial
//
//f(t) = (tr(b))^{−1}tr(b^{(q-1)/3})(t + 1) + (tr(b))^{−1}tr((bt +
//b^q)^{(q-1)/3})(t + tr(b)t^{1/2}+ 1)^{1-(q-1)/3} + t^{1/2},
//where tr(x) =x + x^q is the relative trace. When q = 2^h, with h even,
//f(t) is an o-polynomial for the Adelaide hyperoval.
//
//Best,Tim


void Adelaide_hyperoval(subfield_structure *S, INT *&Pts, INT &nb_pts, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	finite_field *Fq = S->Fq;
	finite_field *FQ = S->FQ;
	INT q = Fq->q;
	INT e = Fq->e;
	INT N = q + 2;
	
	INT i, t, b, bq, bk, tr_b, tr_bk, tr_b_down, tr_bk_down, tr_b_down_inv;
	INT a, tr_a, tr_a_down, t_lift, alpha, k;
	INT sqrt_t, c, cv, d, f;
	INT top1, top2, u, v, w, r;
	INT *Mtx;

	if (f_v) {
		cout << "Adelaide_hyperoval q=" << q << endl;
		}

	if (ODD(e)) {
		cout << "Adelaide_hyperoval need e even" << endl;
		exit(1);
		}
	nb_pts = N;

	k = (q - 1) / 3;
	if (k * 3 != q - 1) {
		cout << "Adelaide_hyperoval k * 3 != q - 1" << endl;
		exit(1);
		}

	alpha = FQ->alpha;
	b = FQ->power(alpha, q - 1);
	if (FQ->power(b, q + 1) != 1) {
		cout << "Adelaide_hyperoval FQ->power(b, q + 1) != 1" << endl;
		exit(1);
		}
	bk = FQ->power(b, k);
	bq = FQ->frobenius_power(b, e);
	tr_b = FQ->add(b, bq);
	tr_bk = FQ->add(bk, FQ->frobenius_power(bk, e));
	tr_b_down = S->Fq_element[tr_b];
	if (tr_b_down == -1) {
		cout << "Adelaide_hyperoval tr_b_down == -1" << endl;
		exit(1);
		}
	tr_bk_down = S->Fq_element[tr_bk];
	if (tr_bk_down == -1) {
		cout << "Adelaide_hyperoval tr_bk_down == -1" << endl;
		exit(1);
		}

	tr_b_down_inv = Fq->inverse(tr_b_down);


	Pts = NEW_INT(N);
	Mtx = NEW_INT(N * 3);
	INT_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {

		sqrt_t = Fq->frobenius_power(t, e - 1);
		if (Fq->mult(sqrt_t, sqrt_t) != t) {
			cout << "Adelaide_hyperoval Fq->mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
			}


		t_lift = S->FQ_embedding[t];
		a = FQ->power(FQ->add(FQ->mult(b, t_lift), bq), k);
		tr_a = FQ->add(a, FQ->frobenius_power(a, e));
		tr_a_down = S->Fq_element[tr_a];
		if (tr_a_down == -1) {
			cout << "Adelaide_hyperoval tr_a_down == -1" << endl;
			exit(1);
			}
		
		c = Fq->add3(t, Fq->mult(tr_b_down, sqrt_t), 1);
		cv = Fq->inverse(c);
		d = Fq->power(cv, k);
		f = Fq->mult(c, d);

		top1 = Fq->mult(tr_bk_down, Fq->add(t, 1));
		u = Fq->mult(top1, tr_b_down_inv);

		top2 = Fq->mult(tr_a_down, f);
		v = Fq->mult(top2, tr_b_down_inv);


		w = Fq->add3(u, v, sqrt_t);


		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = w;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(*Fq, Mtx + i * 3, 1, 3, r);
		Pts[i] = r;
		}

	FREE_INT(Mtx);

	if (f_v) {
		cout << "Adelaide_hyperoval q=" << q << " done" << endl;
		}



}


// following Payne, Penttila, Pinneri: Isomorphisms Between Subiaco q-Clan Geometries, 
// Bull. Belg. Math. Soc. 2 (1995) 197-222.
// formula (53)

void Subiaco_oval(finite_field *F, INT *&Pts, INT &nb_pts, INT f_short, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q = F->q;
	INT e = F->e;
	INT N = q + 1;
	INT i, t, a, b, h, alpha, k, top, bottom;
	INT omega, omega2;
	INT t2, t3, t4, sqrt_t;
	INT *Mtx;

	if (f_v) {
		cout << "Subiaco_oval q=" << q << " f_short=" << f_short << endl;
		}

	nb_pts = N;
	k = (q - 1) / 3;
	if (k * 3 != q - 1) {
		cout << "Subiaco_oval k * 3 != q - 1" << endl;
		exit(1);
		}
	alpha = F->alpha;
	omega = F->power(alpha, k);
	omega2 = F->mult(omega, omega);
	if (F->add3(omega2, omega, 1) != 0) {
		cout << "Subiaco_oval F->add3(omega2, omega, 1) != 0" << endl;
		exit(1);
		}
	Pts = NEW_INT(N);
	Mtx = NEW_INT(N * 3);
	INT_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t2 = F->mult(t, t);
		t3 = F->mult(t2, t);
		t4 = F->mult(t2, t2);
		sqrt_t = F->frobenius_power(t, e - 1);
		if (F->mult(sqrt_t, sqrt_t) != t) {
			cout << "Subiaco_oval F->mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
			}
		bottom = F->add3(t4, F->mult(omega2, t2), 1);
		if (f_short) {
			top = F->mult(omega2, F->add(t4, t));
			}
		else {
			top = F->add3(t3, t2, F->mult(omega2, t));
			}
		if (FALSE) {
			cout << "t=" << t << " top=" << top << " bottom=" << bottom << endl;
			}
		a = F->mult(top, F->inverse(bottom));
		if (f_short) {
			b = sqrt_t;
			}
		else {
			b = F->mult(omega, sqrt_t);
			}
		h = F->add(a, b);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(*F, Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_INT(Mtx);
	if (f_v) {
		cout << "Subiaco_oval q=" << q << " done" << endl;
		}
}



// email 12/27/2014
//The o-polynomial of the Subiaco hyperoval is

//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3 + d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

//where d has absolute trace 1.

//Best,
//Tim

//absolute trace of 1/d is 1 not d...


void Subiaco_hyperoval(finite_field *F, INT *&Pts, INT &nb_pts, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q = F->q;
	INT e = F->e;
	INT N = q + 2;
	INT i, t, d, dv, d2, one_d_d2, a, h;
	INT t2, t3, t4, sqrt_t;
	INT top1, top2, top3, top4, top, bottom;
	INT *Mtx;

	if (f_v) {
		cout << "Subiaco_hyperoval q=" << q << endl;
		}

	nb_pts = N;
	for (d = 1; d < q; d++) {
		dv = F->inverse(d);
		if (F->absolute_trace(dv) == 1) {
			break;
			}
		}
	if (d == q) {
		cout << "Subiaco_hyperoval cannot find element d" << endl;
		exit(1);
		}
	d2 = F->mult(d, d);
	one_d_d2 = F->add3(1, d, d2);

	Pts = NEW_INT(N);
	Mtx = NEW_INT(N * 3);
	INT_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t2 = F->mult(t, t);
		t3 = F->mult(t2, t);
		t4 = F->mult(t2, t2);
		sqrt_t = F->frobenius_power(t, e - 1);
		if (F->mult(sqrt_t, sqrt_t) != t) {
			cout << "Subiaco_hyperoval F->mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
			}


		bottom = F->add3(t4, F->mult(d2, t2), 1);

		//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3 + d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

		top1 = F->mult(d2,t4);
		top2 = F->mult3(d2, one_d_d2, t3);
		top3 = F->mult3(d2, one_d_d2, t2);
		top4 = F->mult(d2, t);
		top = F->add4(top1, top2, top3, top4);

		if (f_v) {
			cout << "t=" << t << " top=" << top << " bottom=" << bottom << endl;
			}
		a = F->mult(top, F->inverse(bottom));
		h = F->add(a, sqrt_t);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(*F, Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_INT(Mtx);
	if (f_v) {
		cout << "Subiaco_hyperoval q=" << q << " done" << endl;
		}
}




// From Bill Cherowitzo's web page:
// In 1991, O'Keefe and Penttila [OKPe92] by means of a detailed investigation 
// of the divisibility properties of the orders of automorphism groups 
// of hypothetical hyperovals in this plane, discovered a new hyperoval. 
// Its o-polynomial is given by:

//f(x) = x4 + x16 + x28 + ß11(x6 + x10 + x14 + x18 + x22 + x26) 
// + ß20(x8 + x20) + ß6(x12 + x24),
//where ß is a primitive root of GF(32) satisfying ß5 = ß2 + 1. 
//The full automorphism group of this hyperoval has order 3.

INT OKeefe_Penttila_32(finite_field *F, INT t)
// needs the field generated by beta with beta^5 = beta^2+1
// From Bill Cherowitzo's hyperoval page
{
	INT *t_powers;
	INT a, b, c, d, e, beta6, beta11, beta20;

	t_powers = NEW_INT(31);
	
	F->power_table(t, t_powers, 31);
	a = F->add3(t_powers[4], t_powers[16], t_powers[28]);
	b = F->add6(t_powers[6], t_powers[10], t_powers[14], t_powers[18], t_powers[22], t_powers[26]);
	c = F->add(t_powers[8], t_powers[20]);
	d = F->add(t_powers[12], t_powers[24]);

	beta6 = F->power(2, 6);
	beta11 = F->power(2, 11);
	beta20 = F->power(2, 20);

	b = F->mult(b, beta11);
	c = F->mult(c, beta20);
	d = F->mult(d, beta6);

	e = F->add4(a, b, c, d);

	FREE_INT(t_powers);
	return e;
}



INT Subiaco64_1(finite_field *F, INT t)
// needs the field generated by beta with beta^6 = beta+1
// The first one from Bill Cherowitzo's hyperoval page
{
	INT *t_powers;
	INT a, b, c, d, beta21, beta42;

	t_powers = NEW_INT(65);
	
	F->power_table(t, t_powers, 65);
	a = F->add6(t_powers[8], t_powers[12], t_powers[20], t_powers[22], t_powers[42], t_powers[52]);
	b = F->add6(t_powers[4], t_powers[10], t_powers[14], t_powers[16], t_powers[30], t_powers[38]);
	c = F->add6(t_powers[44], t_powers[48], t_powers[54], t_powers[56], t_powers[58], t_powers[60]);
	b = F->add3(b, c, t_powers[62]);
	c = F->add7(t_powers[2], t_powers[6], t_powers[26], t_powers[28], t_powers[32], t_powers[36], t_powers[40]);
	beta21 = F->power(2, 21);
	beta42 = F->mult(beta21, beta21);
	d = F->add3(a, F->mult(beta21, b), F->mult(beta42, c));
	FREE_INT(t_powers);
	return d;
}

INT Subiaco64_2(finite_field *F, INT t)
// needs the field generated by beta with beta^6 = beta+1
// The second one from Bill Cherowitzo's hyperoval page
{
	INT *t_powers;
	INT a, b, c, d, beta21, beta42;

	t_powers = NEW_INT(65);
	
	F->power_table(t, t_powers, 65);
	a = F->add3(t_powers[24], t_powers[30], t_powers[62]);
	b = F->add6(t_powers[4], t_powers[8], t_powers[10], t_powers[14], t_powers[16], t_powers[34]);
	c = F->add6(t_powers[38], t_powers[40], t_powers[44], t_powers[46], t_powers[52], t_powers[54]);
	b = F->add4(b, c, t_powers[58], t_powers[60]);
	c = F->add5(t_powers[6], t_powers[12], t_powers[18], t_powers[20], t_powers[26]);
	d = F->add5(t_powers[32], t_powers[36], t_powers[42], t_powers[48], t_powers[50]);
	c = F->add(c, d);
	beta21 = F->power(2, 21);
	beta42 = F->mult(beta21, beta21);
	d = F->add3(a, F->mult(beta21, b), F->mult(beta42, c));
	FREE_INT(t_powers);
	return d;
}

INT Adelaide64(finite_field *F, INT t)
// needs the field generated by beta with beta^6 = beta+1
{
	INT *t_powers;
	INT a, b, c, d, beta21, beta42;

	t_powers = NEW_INT(65);
	
	F->power_table(t, t_powers, 65);
	a = F->add7(t_powers[4], t_powers[8], t_powers[14], t_powers[34], t_powers[42], t_powers[48], t_powers[62]);
	b = F->add8(t_powers[6], t_powers[16], t_powers[26], t_powers[28], t_powers[30], t_powers[32], t_powers[40], t_powers[58]);
	c = F->add8(t_powers[10], t_powers[18], t_powers[24], t_powers[36], t_powers[44], t_powers[50], t_powers[52], t_powers[60]);
	beta21 = F->power(2, 21);
	beta42 = F->mult(beta21, beta21);
	d = F->add3(a, F->mult(beta21, b), F->mult(beta42, c));
	FREE_INT(t_powers);
	return d;
}



void LunelliSce(finite_field *Fq, INT *pts18, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//const BYTE *override_poly = "19";
	//finite_field F;
	//INT n = 3;
	//INT q = 16;
	INT v[3];
	//INT w[3];

	if (f_v) {
		cout << "LunelliSce" << endl;
		}
	//F.init(q), verbose_level - 2);
	//F.init_override_polynomial(q, override_poly, verbose_level);

#if 0
	INT cubic1[100];
	INT cubic1_size = 0;
	INT cubic2[100];
	INT cubic2_size = 0;
	INT hoval[100];
	INT hoval_size = 0;
#endif
	
	INT a, b, i, sz, N;

	if (Fq->q != 16) {
		cout << "LunelliSce field order must be 16" << endl;
		exit(1);
		}
	N = nb_PG_elements(2, 16);
	sz = 0;
	for (i = 0; i < N; i++) {
		PG_element_unrank_modified(*Fq, v, 1, 3, i);
		//cout << "i=" << i << " v=";
		//INT_vec_print(cout, v, 3);
		//cout << endl;
		
		a = LunelliSce_evaluate_cubic1(Fq, v);
		b = LunelliSce_evaluate_cubic2(Fq, v);

		// form the symmetric difference of the two cubics:
		if ((a == 0 && b) || (b == 0 && a)) {
			pts18[sz++] = i;
			}
		}
	if (sz != 18) {
		cout << "sz != 18" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the size of the LinelliSce hyperoval is " << sz << endl;
		cout << "the LinelliSce hyperoval is:" << endl;
		INT_vec_print(cout, pts18, sz);
		cout << endl;
		}

#if 0
	cout << "the size of cubic1 is " << cubic1_size << endl;
	cout << "the cubic1 is:" << endl;
	INT_vec_print(cout, cubic1, cubic1_size);
	cout << endl;
	cout << "the size of cubic2 is " << cubic2_size << endl;
	cout << "the cubic2 is:" << endl;
	INT_vec_print(cout, cubic2, cubic2_size);
	cout << endl;
#endif

}

INT LunelliSce_evaluate_cubic1(finite_field *F, INT *v)
// computes X^3 + Y^3 + Z^3 + \eta^3 XYZ
{
	INT a, b, c, d, e, eta3;

	eta3 = F->power(2, 3);
	//eta12 = F->power(2, 12);
	a = F->power(v[0], 3);
	b = F->power(v[1], 3);
	c = F->power(v[2], 3);
	d = F->product4(eta3, v[0], v[1], v[2]);
	e = F->add4(a, b, c, d);
	return e;
}

INT LunelliSce_evaluate_cubic2(finite_field *F, INT *v)
// computes X^3 + Y^3 + Z^3 + \eta^{12} XYZ
{
	INT a, b, c, d, e, eta12;

	//eta3 = F->power(2, 3);
	eta12 = F->power(2, 12);
	a = F->power(v[0], 3);
	b = F->power(v[1], 3);
	c = F->power(v[2], 3);
	d = F->product4(eta12, v[0], v[1], v[2]);
	e = F->add4(a, b, c, d);
	return e;
}


// formerly DISCRETA/extras.C
//
// Anton Betten
// Sept 17, 2010

// plane_invariant started 2/23/09


void plane_invariant(INT q, orthogonal *O, unusual_model *U, 
	INT size, INT *set, 
	INT &nb_planes, INT *&intersection_matrix, 
	INT &Block_size, INT *&Blocks, 
	INT verbose_level)
// using hash values
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT *Mtx;
	INT *Hash;
	INT rk, H, log2_of_q, n_choose_k;
	INT f_special = FALSE;
	INT f_complete = TRUE;
	INT base_col[1000];
	INT subset[1000];
	INT level = 3;
	INT n = 5;
	INT cnt;
	INT i;
	
	
	n_choose_k = INT_n_choose_k(size, level);
	log2_of_q = INT_log2(q);
	
	Mtx = NEW_INT(level * n);
	Hash = NEW_INT(n_choose_k);
	
	first_k_subset(subset, size, level);
	cnt = -1;
	
	if (f_v) {
		cout << "computing planes spanned by 3-subsets" << endl;
		cout << "n_choose_k=" << n_choose_k << endl;
		cout << "log2_of_q=" << log2_of_q << endl;
		}
	while (TRUE) {
		cnt++;
		
		for (i = 0; i < level; i++) {
			Q_unrank(*O->F, Mtx + i * n, 1, n - 1, set[subset[i]]);
			}
		if (f_vvv) {
			cout << "subset " << setw(5) << cnt << " : ";
			INT_vec_print(cout, subset, level);
			cout << " : "; // << endl;
			}
		//print_integer_matrix_width(cout, Mtx, level, n, n, 3);
		rk = O->F->Gauss_INT(Mtx, f_special, f_complete, base_col, FALSE, NULL, level, n, n, 0);
		if (f_vvv) {
			cout << "after Gauss, rank = " << rk << endl;
			print_integer_matrix_width(cout, Mtx, level, n, n, 3);
			}
		H = 0;
		for (i = 0; i < level * n; i++) {
			H = hashing_fixed_width(H, Mtx[i], log2_of_q);
			}
		if (f_vvv) {
			cout << "hash =" << setw(10) << H << endl;
			}
		Hash[cnt] = H;
		if (!next_k_subset(subset, size, level)) {
			break;
			}
		}
	INT *Hash_sorted, *sorting_perm, *sorting_perm_inv, nb_types, *type_first, *type_len;
	
	INT_vec_classify(n_choose_k, Hash, Hash_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
	
	
	if (f_v) {
		cout << nb_types << " types of planes" << endl;
		}
	if (f_vvv) {
		for (i = 0; i < nb_types; i++) {
			cout << setw(3) << i << " : " 
				<< setw(4) << type_first[i] << " : " 
				<< setw(4) << type_len[i] << " : " 
				<< setw(10) << Hash_sorted[type_first[i]] << endl;
			}
		}
	INT *type_len_sorted, *sorting_perm2, *sorting_perm_inv2, 
		nb_types2, *type_first2, *type_len2;
	
	INT_vec_classify(nb_types, type_len, type_len_sorted, 
		sorting_perm2, sorting_perm_inv2, 
		nb_types2, type_first2, type_len2);

	if (f_v) {
		cout << "multiplicities:" << endl;
		for (i = 0; i < nb_types2; i++) {
			//cout << setw(3) << i << " : " 
			//<< setw(4) << type_first2[i] << " : " 
			cout << setw(4) << type_len2[i] << " x " 
				<< setw(10) << type_len_sorted[type_first2[i]] << endl;
			}
		}
	INT f, ff, ll, j, u, ii, jj, idx;
	
	f = type_first2[nb_types2 - 1];
	nb_planes = type_len2[nb_types2 - 1];
	if (f_v) {
		if (nb_planes == 1) {
			cout << "there is a unique plane that appears " << type_len_sorted[f] << " times among the 3-sets of points" << endl;
			}
		else {
			cout << "there are " << nb_planes << " planes that each appear " << type_len_sorted[f] << " times among the 3-sets of points" << endl;
			for (i = 0; i < nb_planes; i++) {
				j = sorting_perm_inv2[f + i];
				cout << "The " << i << "-th plane, which is " << j << ", appears " << type_len_sorted[f + i] << " times" << endl;
				}
			}
		}
	if (f_vvv) {
		cout << "these planes are:" << endl;
		for (i = 0; i < nb_planes; i++) {
			cout << "plane " << i << endl;
			j = sorting_perm_inv2[f + i];
			ff = type_first[j];
			ll = type_len[j];
			for (u = 0; u < ll; u++) {
				cnt = sorting_perm_inv[ff + u];
				unrank_k_subset(cnt, subset, size, level);
				cout << "subset " << setw(5) << cnt << " : ";
				INT_vec_print(cout, subset, level);
				cout << " : " << endl;
				}
			}
		}	
	
	//return;
	
	//INT *Blocks;
	INT *Block;
	//INT Block_size;
	
	
	Block = NEW_INT(size);
	Blocks = NEW_INT(nb_planes * size);
	
	for (i = 0; i < nb_planes; i++) {
		j = sorting_perm_inv2[f + i];
		ff = type_first[j];
		ll = type_len[j];
		if (f_vv) {
			cout << setw(3) << i << " : " << setw(3) << " : " 
				<< setw(4) << ff << " : " 
				<< setw(4) << ll << " : " 
				<< setw(10) << Hash_sorted[type_first[j]] << endl;
			}
		Block_size = 0;
		for (u = 0; u < ll; u++) {
			cnt = sorting_perm_inv[ff + u];
			unrank_k_subset(cnt, subset, size, level);
			if (f_vvv) {
				cout << "subset " << setw(5) << cnt << " : ";
				INT_vec_print(cout, subset, level);
				cout << " : " << endl;
				}
			for (ii = 0; ii < level; ii++) {
				Q_unrank(*O->F, Mtx + ii * n, 1, n - 1, set[subset[ii]]);
				}
			for (ii = 0; ii < level; ii++) {
				if (!INT_vec_search(Block, Block_size, subset[ii], idx)) {
					for (jj = Block_size; jj > idx; jj--) {
						Block[jj] = Block[jj - 1];
						}
					Block[idx] = subset[ii];
					Block_size++;
					}
				}
			rk = O->F->Gauss_INT(Mtx, f_special, f_complete, base_col, FALSE, NULL, level, n, n, 0);
			if (f_vvv)  {
				cout << "after Gauss, rank = " << rk << endl;
				print_integer_matrix_width(cout, Mtx, level, n, n, 3);
				}
			
			H = 0;
			for (ii = 0; ii < level * n; ii++) {
				H = hashing_fixed_width(H, Mtx[ii], log2_of_q);
				}
			if (f_vvv) {
				cout << "hash =" << setw(10) << H << endl;
				}
			}
		if (f_vv) {
			cout << "found Block ";
			INT_vec_print(cout, Block, Block_size);
			cout << endl;
			}
		for (u = 0; u < Block_size; u++) {
			Blocks[i * Block_size + u] = Block[u];
			}
		}
	if (f_vv) {
		cout << "Incidence structure between points and high frequency planes:" << endl;
		if (nb_planes < 30) {
			print_integer_matrix_width(cout, Blocks, nb_planes, Block_size, Block_size, 3);
			}
		}
	
	INT *Incma, *Incma_t, *IIt, *ItI;
	INT a;
	
	Incma = NEW_INT(size * nb_planes);
	Incma_t = NEW_INT(nb_planes * size);
	IIt = NEW_INT(size * size);
	ItI = NEW_INT(nb_planes * nb_planes);


	for (i = 0; i < size * nb_planes; i++) {
		Incma[i] = 0;
		}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < Block_size; j++) {
			a = Blocks[i * Block_size + j];
			Incma[a * nb_planes + i] = 1;
			}
		}
	if (f_vv) {
		cout << "Incidence matrix:" << endl;
		print_integer_matrix_width(cout, Incma, size, nb_planes, nb_planes, 1);
		}
	for (i = 0; i < size; i++) {
		for (j = 0; j < nb_planes; j++) {
			Incma_t[j * size + i] = Incma[i * nb_planes + j];
			}
		}
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] * Incma_t[u * size + j];
				}
			IIt[i * size + j] = a;
			}
		}
	if (f_vv) {
		cout << "I * I^\\top = " << endl;
		print_integer_matrix_width(cout, IIt, size, size, size, 2);
		}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < size; u++) {
				a += Incma[u * nb_planes + i] * Incma[u * nb_planes + j];
				}
			ItI[i * nb_planes + j] = a;
			}
		}
	if (f_v) {
		cout << "I^\\top * I = " << endl;
		print_integer_matrix_width(cout, ItI, nb_planes, nb_planes, nb_planes, 3);
		}
	
	intersection_matrix = NEW_INT(nb_planes * nb_planes);
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			intersection_matrix[i * nb_planes + j] = ItI[i * nb_planes + j];
			}
		}
	
#if 0
	{
		BYTE fname[1000];
		
		sprintf(fname, "plane_invariant_%ld_%ld.txt", q, k);
		
		ofstream fp(fname);
		fp << nb_planes << endl;
		for (i = 0; i < nb_planes; i++) {
			for (j = 0; j < nb_planes; j++) {
				fp << ItI[i * nb_planes + j] << " ";
				}
			fp << endl;
			}
		fp << -1 << endl;
		fp << "# Incidence structure between points and high frequency planes:" << endl;
		fp << l << " " << Block_size << endl;
		print_integer_matrix_width(fp, Blocks, nb_planes, Block_size, Block_size, 3);
		fp << -1 << endl;
		
	}
#endif

	FREE_INT(Mtx);
	FREE_INT(Hash);
	FREE_INT(Block);
	//FREE_INT(Blocks);
	FREE_INT(Incma);
	FREE_INT(Incma_t);
	FREE_INT(IIt);
	FREE_INT(ItI);


	FREE_INT(Hash_sorted);
	FREE_INT(sorting_perm);
	FREE_INT(sorting_perm_inv);
	FREE_INT(type_first);
	FREE_INT(type_len);
	


	FREE_INT(type_len_sorted);
	FREE_INT(sorting_perm2);
	FREE_INT(sorting_perm_inv2);
	FREE_INT(type_first2);
	FREE_INT(type_len2);



}



void create_Law_71_BLT_set(orthogonal *O, INT *set, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT v[5], v0, v1, v2, v3, v4;
	INT i;
	INT coordinates[] = {
#if 1
		0,0,0,0,1,
		1,0,0,0,0,
		1,20,1,33,5,
		1,6,23,19,23,
		1,32,11,35,17,
		1,33,12,14,23,
		1,25,8,12,6,
		1,16,6,1,22,
		1,23,8,5,6,
		1,8,6,13,8,
		1,22,19,20,13,
		1,21,23,16,23,
		1,28,6,9,8,
		1,2,26,7,13,
		1,5,9,36,35,
		1,12,23,10,17,
		1,14,16,25,23,
		1,9,8,26,35,
		1,1,11,8,19,
		1,19,12,11,17,
		1,18,27,22,22,
		1,24,36,17,35,
		1,26,27,23,5,
		1,27,25,24,22,
		1,36,21,32,35,
		1,7,16,31,8,
		1,35,5,15,5,
		1,10,36,6,13,
		1,30,4,3,5,
		1,4,3,30,19,
		1,17,13,2,19,
		1,11,28,18,17,
		1,13,16,27,22,
		1,29,12,28,6,
		1,15,10,34,19,
		1,3,30,4,13,
		1,31,9,21,8,
		1,34,9,29,6
#endif
		};
	finite_field *F;
	INT q;
	
	F = O->F;
	q = F->q;
	if (q != 71) {
		cout << "create_LP_71_BLT_set q = 71" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		INT_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			INT_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = O->rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "the BLT set LP_71 is ";
		INT_vec_print(cout, set, q + 1);
		cout << endl;
		}
}




