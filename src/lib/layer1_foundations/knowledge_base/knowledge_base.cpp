// knowledge_base.cpp
//
// Anton Betten
//
// started:  July 9, 2009


// Spread data is available for:
// 2^2 = 4
// 3^2 = 9
// 2^4 = 16
// 4^2 = 16
// 5^2 = 25
// 3^3 = 27



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {



#include "DATA/quartic_curves/quartic_curves_q9.cpp"
#include "DATA/quartic_curves/quartic_curves_q13.cpp"
#include "DATA/quartic_curves/quartic_curves_q17.cpp"
#include "DATA/quartic_curves/quartic_curves_q19.cpp"
#include "DATA/quartic_curves/quartic_curves_q23.cpp"
#include "DATA/quartic_curves/quartic_curves_q25.cpp"
#include "DATA/quartic_curves/quartic_curves_q27.cpp"
#include "DATA/quartic_curves/quartic_curves_q29.cpp"
#include "DATA/quartic_curves/quartic_curves_q31.cpp"

#include "DATA/data_hyperovals.cpp"

#include "DATA/cubic_surfaces/surface_4.cpp"
#include "DATA/cubic_surfaces/surface_7.cpp"
#include "DATA/cubic_surfaces/surface_8.cpp"
#include "DATA/cubic_surfaces/surface_9.cpp"
#include "DATA/cubic_surfaces/surface_11.cpp"
#include "DATA/cubic_surfaces/surface_13.cpp"
#include "DATA/cubic_surfaces/surface_16.cpp"
#include "DATA/cubic_surfaces/surface_17.cpp"
#include "DATA/cubic_surfaces/surface_19.cpp"
#include "DATA/cubic_surfaces/surface_23.cpp"
#include "DATA/cubic_surfaces/surface_25.cpp"
#include "DATA/cubic_surfaces/surface_27.cpp"
#include "DATA/cubic_surfaces/surface_29.cpp"
#include "DATA/cubic_surfaces/surface_31.cpp"
#include "DATA/cubic_surfaces/surface_32.cpp"
#include "DATA/cubic_surfaces/surface_37.cpp"
#include "DATA/cubic_surfaces/surface_41.cpp"
#include "DATA/cubic_surfaces/surface_43.cpp"
#include "DATA/cubic_surfaces/surface_47.cpp"
#include "DATA/cubic_surfaces/surface_49.cpp"
#include "DATA/cubic_surfaces/surface_53.cpp"
#include "DATA/cubic_surfaces/surface_59.cpp"
#include "DATA/cubic_surfaces/surface_61.cpp"
#include "DATA/cubic_surfaces/surface_64.cpp"
#include "DATA/cubic_surfaces/surface_67.cpp"
#include "DATA/cubic_surfaces/surface_71.cpp"
#include "DATA/cubic_surfaces/surface_73.cpp"
#include "DATA/cubic_surfaces/surface_79.cpp"
#include "DATA/cubic_surfaces/surface_81.cpp"
#include "DATA/cubic_surfaces/surface_83.cpp"
#include "DATA/cubic_surfaces/surface_89.cpp"
#include "DATA/cubic_surfaces/surface_97.cpp"
#include "DATA/cubic_surfaces/surface_101.cpp"
#include "DATA/cubic_surfaces/surface_103.cpp"
#include "DATA/cubic_surfaces/surface_107.cpp"
#include "DATA/cubic_surfaces/surface_109.cpp"
#include "DATA/cubic_surfaces/surface_113.cpp"
#include "DATA/cubic_surfaces/surface_121.cpp"
#include "DATA/cubic_surfaces/surface_127.cpp"
#include "DATA/cubic_surfaces/surface_128.cpp"

#include "DATA/data_DH.cpp"

#include "DATA/data_spreads.cpp"

#include "DATA/data_tensor.cpp"

#include "DATA/data_BLT.cpp"

#include "DATA/planes_16.cpp"


knowledge_base::knowledge_base()
{

}

knowledge_base::~knowledge_base()
{

}


// #############################################################################
// quartic curves:
// #############################################################################



int knowledge_base::quartic_curves_nb_reps(int q)
{
	int nb;

	if (q == 9) {
		nb = quartic_curves_q9_nb_reps;
	}
	else if (q == 13) {
		nb = quartic_curves_q13_nb_reps;
	}
	else if (q == 17) {
		nb = quartic_curves_q17_nb_reps;
	}
	else if (q == 19) {
		nb = quartic_curves_q19_nb_reps;
	}
	else if (q == 23) {
		nb = quartic_curves_q23_nb_reps;
	}
	else if (q == 25) {
		nb = quartic_curves_q25_nb_reps;
	}
	else if (q == 27) {
		nb = quartic_curves_q27_nb_reps;
	}
	else if (q == 29) {
		nb = quartic_curves_q29_nb_reps;
	}
	else if (q == 31) {
		nb = quartic_curves_q31_nb_reps;
	}
	else {
		cout << "knowledge_base::quartic_curves_nb_reps q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

int *knowledge_base::quartic_curves_representative(int q, int i)
// i starts from 0
{
	int *p, nb, sz;

	if (q == 9) {
		p = quartic_curves_q9_reps;
		nb = quartic_curves_q9_nb_reps;
		sz = quartic_curves_q9_size;
	}
	else if (q == 13) {
		p = quartic_curves_q13_reps;
		nb = quartic_curves_q13_nb_reps;
		sz = quartic_curves_q13_size;
	}
	else if (q == 17) {
		p = quartic_curves_q17_reps;
		nb = quartic_curves_q17_nb_reps;
		sz = quartic_curves_q17_size;
	}
	else if (q == 19) {
		p = quartic_curves_q19_reps;
		nb = quartic_curves_q19_nb_reps;
		sz = quartic_curves_q19_size;
	}
	else if (q == 23) {
		p = quartic_curves_q23_reps;
		nb = quartic_curves_q23_nb_reps;
		sz = quartic_curves_q23_size;
	}
	else if (q == 25) {
		p = quartic_curves_q25_reps;
		nb = quartic_curves_q25_nb_reps;
		sz = quartic_curves_q25_size;
	}
	else if (q == 27) {
		p = quartic_curves_q27_reps;
		nb = quartic_curves_q27_nb_reps;
		sz = quartic_curves_q27_size;
	}
	else if (q == 29) {
		p = quartic_curves_q29_reps;
		nb = quartic_curves_q29_nb_reps;
		sz = quartic_curves_q29_size;
	}
	else if (q == 31) {
		p = quartic_curves_q31_reps;
		nb = quartic_curves_q31_nb_reps;
		sz = quartic_curves_q31_size;
	}
	else {
		cout << "knowledge_base::quartic_curves_representative q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
	}
	if (i < 0) {
		cout << "knowledge_base::quartic_curves_representative q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
	}
	if (i >= nb) {
		cout << "knowledge_base::quartic_curves_representative q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
	}
	p += i * sz;
	return p;
}

long int *knowledge_base::quartic_curves_bitangents(int q, int i)
// i starts from 0
{
	long int *p;
	int nb;

	if (q == 9) {
		p = quartic_curves_q9_Bitangents;
		nb = quartic_curves_q9_nb_reps;
	}
	else if (q == 13) {
		p = quartic_curves_q13_Bitangents;
		nb = quartic_curves_q13_nb_reps;
	}
	else if (q == 17) {
		p = quartic_curves_q17_Bitangents;
		nb = quartic_curves_q17_nb_reps;
	}
	else if (q == 19) {
		p = quartic_curves_q19_Bitangents;
		nb = quartic_curves_q19_nb_reps;
	}
	else if (q == 23) {
		p = quartic_curves_q23_Bitangents;
		nb = quartic_curves_q23_nb_reps;
	}
	else if (q == 25) {
		p = quartic_curves_q25_Bitangents;
		nb = quartic_curves_q25_nb_reps;
	}
	else if (q == 27) {
		p = quartic_curves_q27_Bitangents;
		nb = quartic_curves_q27_nb_reps;
	}
	else if (q == 29) {
		p = quartic_curves_q29_Bitangents;
		nb = quartic_curves_q29_nb_reps;
	}
	else if (q == 31) {
		p = quartic_curves_q31_Bitangents;
		nb = quartic_curves_q31_nb_reps;
	}
	else {
		cout << "knowledge_base::quartic_curves_bitangents q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
	}
	if (i < 0) {
		cout << "knowledge_base::quartic_curves_bitangents q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
	}
	if (i >= nb) {
		cout << "knowledge_base::quartic_curves_bitangents q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
	}
	p += i * 28;
	return p;
}

void knowledge_base::quartic_curves_stab_gens(int q, int i,
		int *&data, int &nb_gens, int &data_size, std::string &stab_order_str)
{
	int *Reps;
	int nb, make_element_size;
	int f, l;
	const char *stab_order;

	if (q == 9) {
		Reps = quartic_curves_q9_stab_gens;
		nb = quartic_curves_q9_nb_reps;
		make_element_size = quartic_curves_q9_make_element_size;
		f = quartic_curves_q9_stab_gens_fst[i];
		l = quartic_curves_q9_stab_gens_len[i];
		stab_order = quartic_curves_q9_stab_order[i];
	}
	else if (q == 13) {
		Reps = quartic_curves_q13_stab_gens;
		nb = quartic_curves_q13_nb_reps;
		make_element_size = quartic_curves_q13_make_element_size;
		f = quartic_curves_q13_stab_gens_fst[i];
		l = quartic_curves_q13_stab_gens_len[i];
		stab_order = quartic_curves_q13_stab_order[i];
	}
	else if (q == 17) {
		Reps = quartic_curves_q17_stab_gens;
		nb = quartic_curves_q17_nb_reps;
		make_element_size = quartic_curves_q17_make_element_size;
		f = quartic_curves_q17_stab_gens_fst[i];
		l = quartic_curves_q17_stab_gens_len[i];
		stab_order = quartic_curves_q17_stab_order[i];
	}
	else if (q == 19) {
		Reps = quartic_curves_q19_stab_gens;
		nb = quartic_curves_q19_nb_reps;
		make_element_size = quartic_curves_q19_make_element_size;
		f = quartic_curves_q19_stab_gens_fst[i];
		l = quartic_curves_q19_stab_gens_len[i];
		stab_order = quartic_curves_q19_stab_order[i];
	}
	else if (q == 23) {
		Reps = quartic_curves_q23_stab_gens;
		nb = quartic_curves_q23_nb_reps;
		make_element_size = quartic_curves_q23_make_element_size;
		f = quartic_curves_q23_stab_gens_fst[i];
		l = quartic_curves_q23_stab_gens_len[i];
		stab_order = quartic_curves_q23_stab_order[i];
	}
	else if (q == 25) {
		Reps = quartic_curves_q25_stab_gens;
		nb = quartic_curves_q25_nb_reps;
		make_element_size = quartic_curves_q25_make_element_size;
		f = quartic_curves_q25_stab_gens_fst[i];
		l = quartic_curves_q25_stab_gens_len[i];
		stab_order = quartic_curves_q25_stab_order[i];
	}
	else if (q == 27) {
		Reps = quartic_curves_q27_stab_gens;
		nb = quartic_curves_q27_nb_reps;
		make_element_size = quartic_curves_q27_make_element_size;
		f = quartic_curves_q27_stab_gens_fst[i];
		l = quartic_curves_q27_stab_gens_len[i];
		stab_order = quartic_curves_q27_stab_order[i];
	}
	else if (q == 29) {
		Reps = quartic_curves_q29_stab_gens;
		nb = quartic_curves_q29_nb_reps;
		make_element_size = quartic_curves_q29_make_element_size;
		f = quartic_curves_q29_stab_gens_fst[i];
		l = quartic_curves_q29_stab_gens_len[i];
		stab_order = quartic_curves_q29_stab_order[i];
	}
	else if (q == 31) {
		Reps = quartic_curves_q31_stab_gens;
		nb = quartic_curves_q31_nb_reps;
		make_element_size = quartic_curves_q31_make_element_size;
		f = quartic_curves_q31_stab_gens_fst[i];
		l = quartic_curves_q31_stab_gens_len[i];
		stab_order = quartic_curves_q31_stab_order[i];
	}
	else {
		cout << "knowledge_base::quartic_curves_stab_gens q=" << q
				<< " I don't have information for this field order" << endl;
		exit(1);
	}
	if (i < 0) {
		cout << "knowledge_base::quartic_curves_stab_gens q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
	}
	if (i >= nb) {
		cout << "knowledge_base::quartic_curves_stab_gens q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
	}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
	stab_order_str.assign(stab_order);
}



// #############################################################################
// Cubic surfaces:
// #############################################################################



int knowledge_base::cubic_surface_nb_reps(int q)
{
	int nb;

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
	else if (q == 101) {
		nb = surface_101_nb_reps;
		}
	else if (q == 103) {
		nb = surface_103_nb_reps;
		}
	else if (q == 107) {
		nb = surface_107_nb_reps;
		}
	else if (q == 109) {
		nb = surface_109_nb_reps;
		}
	else if (q == 113) {
		nb = surface_113_nb_reps;
		}
	else if (q == 121) {
		nb = surface_121_nb_reps;
		}
	else if (q == 128) {
		nb = surface_128_nb_reps;
		}
	else {
		cout << "knowledge_base::cubic_surface_nb_reps q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

int *knowledge_base::cubic_surface_representative(int q, int i)
// i starts from 0
{
	int *p, nb, sz;
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
	else if (q == 101) {
		p = surface_101_reps;
		nb = surface_101_nb_reps;
		sz = surface_101_size;
		}
	else if (q == 103) {
		p = surface_103_reps;
		nb = surface_103_nb_reps;
		sz = surface_103_size;
		}
	else if (q == 107) {
		p = surface_107_reps;
		nb = surface_107_nb_reps;
		sz = surface_107_size;
		}
	else if (q == 109) {
		p = surface_109_reps;
		nb = surface_109_nb_reps;
		sz = surface_109_size;
		}
	else if (q == 113) {
		p = surface_113_reps;
		nb = surface_113_nb_reps;
		sz = surface_113_size;
		}
	else if (q == 121) {
		p = surface_121_reps;
		nb = surface_121_nb_reps;
		sz = surface_121_size;
		}
	else if (q == 127) {
		p = surface_127_reps;
		nb = surface_127_nb_reps;
		sz = surface_127_size;
		}
	else if (q == 128) {
		p = surface_128_reps;
		nb = surface_128_nb_reps;
		sz = surface_128_size;
		}
	else {
		cout << "knowledge_base::cubic_surface_representative q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::cubic_surface_representative q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::cubic_surface_representative q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void knowledge_base::cubic_surface_stab_gens(int q, int i,
		int *&data, int &nb_gens, int &data_size, std::string &stab_order_str)
{
	int *Reps;
	int nb, make_element_size;
	int f, l;
	const char *stab_order;
	
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
	else if (q == 101) {
		Reps = surface_101_stab_gens;
		nb = surface_101_nb_reps;
		make_element_size = surface_101_make_element_size;
		f = surface_101_stab_gens_fst[i];
		l = surface_101_stab_gens_len[i];
		stab_order = surface_101_stab_order[i];
		}
	else if (q == 103) {
		Reps = surface_103_stab_gens;
		nb = surface_103_nb_reps;
		make_element_size = surface_103_make_element_size;
		f = surface_103_stab_gens_fst[i];
		l = surface_103_stab_gens_len[i];
		stab_order = surface_103_stab_order[i];
		}
	else if (q == 107) {
		Reps = surface_107_stab_gens;
		nb = surface_107_nb_reps;
		make_element_size = surface_107_make_element_size;
		f = surface_107_stab_gens_fst[i];
		l = surface_107_stab_gens_len[i];
		stab_order = surface_107_stab_order[i];
		}
	else if (q == 109) {
		Reps = surface_109_stab_gens;
		nb = surface_109_nb_reps;
		make_element_size = surface_109_make_element_size;
		f = surface_109_stab_gens_fst[i];
		l = surface_109_stab_gens_len[i];
		stab_order = surface_109_stab_order[i];
		}
	else if (q == 113) {
		Reps = surface_113_stab_gens;
		nb = surface_113_nb_reps;
		make_element_size = surface_113_make_element_size;
		f = surface_113_stab_gens_fst[i];
		l = surface_113_stab_gens_len[i];
		stab_order = surface_113_stab_order[i];
		}
	else if (q == 121) {
		Reps = surface_121_stab_gens;
		nb = surface_121_nb_reps;
		make_element_size = surface_121_make_element_size;
		f = surface_121_stab_gens_fst[i];
		l = surface_121_stab_gens_len[i];
		stab_order = surface_121_stab_order[i];
		}
	else if (q == 127) {
		Reps = surface_127_stab_gens;
		nb = surface_127_nb_reps;
		make_element_size = surface_127_make_element_size;
		f = surface_127_stab_gens_fst[i];
		l = surface_127_stab_gens_len[i];
		stab_order = surface_127_stab_order[i];
		}
	else if (q == 128) {
		Reps = surface_128_stab_gens;
		nb = surface_128_nb_reps;
		make_element_size = surface_128_make_element_size;
		f = surface_128_stab_gens_fst[i];
		l = surface_128_stab_gens_len[i];
		stab_order = surface_128_stab_order[i];
		}
	else {
		cout << "knowledge_base::cubic_surface_stab_gens q=" << q
				<< " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::cubic_surface_stab_gens q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::cubic_surface_stab_gens q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
	stab_order_str.assign(stab_order);
}

int knowledge_base::cubic_surface_nb_Eckardt_points(int q, int i)
// i starts from 0
{
	int *p, nb; //, sz;
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
	else if (q == 101) {
		p = surface_101_nb_E;
		nb = surface_101_nb_reps;
		//sz = surface_101_size;
		}
	else if (q == 103) {
		p = surface_103_nb_E;
		nb = surface_103_nb_reps;
		//sz = surface_103_size;
		}
	else if (q == 107) {
		p = surface_107_nb_E;
		nb = surface_107_nb_reps;
		//sz = surface_107_size;
		}
	else if (q == 109) {
		p = surface_109_nb_E;
		nb = surface_109_nb_reps;
		//sz = surface_109_size;
		}
	else if (q == 113) {
		p = surface_113_nb_E;
		nb = surface_113_nb_reps;
		//sz = surface_113_size;
		}
	else if (q == 121) {
		p = surface_121_nb_E;
		nb = surface_121_nb_reps;
		//sz = surface_121_size;
		}
	else if (q == 127) {
		p = surface_127_nb_E;
		nb = surface_127_nb_reps;
		//sz = surface_127_size;
		}
	else if (q == 128) {
		p = surface_128_nb_E;
		nb = surface_128_nb_reps;
		//sz = surface_128_size;
		}
	else {
		cout << "knowledge_base::cubic_surface_nb_Eckardt_points q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::cubic_surface_nb_Eckardt_points q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::cubic_surface_nb_Eckardt_points q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	return p[i];
}


long int *knowledge_base::cubic_surface_Lines(int q, int i)
// i starts from 0
{
	long int *p, nb;
	if (q == 4) {
		p = surface_4_Lines;
		nb = surface_4_nb_reps;
		}
	else if (q == 7) {
		p = surface_7_Lines;
		nb = surface_7_nb_reps;
		}
	else if (q == 8) {
		p = surface_8_Lines;
		nb = surface_8_nb_reps;
		}
	else if (q == 9) {
		p = surface_9_Lines;
		nb = surface_9_nb_reps;
		}
	else if (q == 11) {
		p = surface_11_Lines;
		nb = surface_11_nb_reps;
		}
	else if (q == 13) {
		p = surface_13_Lines;
		nb = surface_13_nb_reps;
		}
	else if (q == 16) {
		p = surface_16_Lines;
		nb = surface_16_nb_reps;
		}
	else if (q == 17) {
		p = surface_17_Lines;
		nb = surface_17_nb_reps;
		}
	else if (q == 19) {
		p = surface_19_Lines;
		nb = surface_19_nb_reps;
		}
	else if (q == 23) {
		p = surface_23_Lines;
		nb = surface_23_nb_reps;
		}
	else if (q == 25) {
		p = surface_25_Lines;
		nb = surface_25_nb_reps;
		}
	else if (q == 27) {
		p = surface_27_Lines;
		nb = surface_27_nb_reps;
		}
	else if (q == 29) {
		p = surface_29_Lines;
		nb = surface_29_nb_reps;
		}
	else if (q == 31) {
		p = surface_31_Lines;
		nb = surface_31_nb_reps;
		}
	else if (q == 32) {
		p = surface_32_Lines;
		nb = surface_32_nb_reps;
		}
	else if (q == 37) {
		p = surface_37_Lines;
		nb = surface_37_nb_reps;
		}
	else if (q == 41) {
		p = surface_41_Lines;
		nb = surface_41_nb_reps;
		}
	else if (q == 43) {
		p = surface_43_Lines;
		nb = surface_43_nb_reps;
		}
	else if (q == 47) {
		p = surface_47_Lines;
		nb = surface_47_nb_reps;
		}
	else if (q == 49) {
		p = surface_49_Lines;
		nb = surface_49_nb_reps;
		}
	else if (q == 53) {
		p = surface_53_Lines;
		nb = surface_53_nb_reps;
		}
	else if (q == 59) {
		p = surface_59_Lines;
		nb = surface_59_nb_reps;
		}
	else if (q == 61) {
		p = surface_61_Lines;
		nb = surface_61_nb_reps;
		}
	else if (q == 64) {
		p = surface_64_Lines;
		nb = surface_64_nb_reps;
		}
	else if (q == 67) {
		p = surface_67_Lines;
		nb = surface_67_nb_reps;
		}
	else if (q == 71) {
		p = surface_71_Lines;
		nb = surface_71_nb_reps;
		}
	else if (q == 73) {
		p = surface_73_Lines;
		nb = surface_73_nb_reps;
		}
	else if (q == 79) {
		p = surface_79_Lines;
		nb = surface_79_nb_reps;
		}
	else if (q == 81) {
		p = surface_81_Lines;
		nb = surface_81_nb_reps;
		}
	else if (q == 83) {
		p = surface_83_Lines;
		nb = surface_83_nb_reps;
		}
	else if (q == 89) {
		p = surface_89_Lines;
		nb = surface_89_nb_reps;
		}
	else if (q == 97) {
		p = surface_97_Lines;
		nb = surface_97_nb_reps;
		}
	else if (q == 101) {
		p = surface_101_Lines;
		nb = surface_101_nb_reps;
		}
	else if (q == 103) {
		p = surface_103_Lines;
		nb = surface_103_nb_reps;
		}
	else if (q == 107) {
		p = surface_107_Lines;
		nb = surface_107_nb_reps;
		}
	else if (q == 109) {
		p = surface_109_Lines;
		nb = surface_109_nb_reps;
		}
	else if (q == 113) {
		p = surface_113_Lines;
		nb = surface_113_nb_reps;
		}
	else if (q == 121) {
		p = surface_121_Lines;
		nb = surface_121_nb_reps;
		}
	else if (q == 127) {
		p = surface_127_Lines;
		nb = surface_127_nb_reps;
		}
	else if (q == 128) {
		p = surface_128_Lines;
		nb = surface_128_nb_reps;
		}
	else {
		cout << "knowledge_base::cubic_surface_Lines q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::cubic_surface_Lines q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::cubic_surface_Lines q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	return p + i * 27;
}



// #############################################################################
// Hyperovals:
// #############################################################################


int knowledge_base::hyperoval_nb_reps(int q)
{
	int nb;

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
		cout << "knowledge_base::hyperoval_nb_reps q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

int *knowledge_base::hyperoval_representative(int q, int i)
// i starts from 0
{
	int *p, nb, sz;
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
		cout << "knowledge_base::hyperovals_representative q=" << q
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::hyperoval_representative q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::hyperoval_representative q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void knowledge_base::hyperoval_gens(int q, int i,
		int *&data, int &nb_gens, int &data_size, std::string &stab_order_str)
{
	int *Reps;
	int nb, make_element_size;
	int f, l;
	const char *stab_order;
	
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
		cout << "knowledge_base::hyperoval_representative q=" << q
				<< " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::hyperoval_representative q=" << q << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::hyperoval_representative q=" << q << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
	stab_order_str.assign(stab_order);
}




// #############################################################################
// Dual hyperovals:
// #############################################################################




int knowledge_base::DH_nb_reps(int k, int n)
{
	int nb;

	if (k == 4 && n == 7) {
		nb = DH_4_7_nb_reps;
		}
	else if (k == 4 && n == 8) {
		nb = DH_4_8_nb_reps;
		}
	else {
		cout << "knowledge_base::DH_nb_reps k=" << k << " n=" << n
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}

long int *knowledge_base::DH_representative(int k, int n, int i)
// i starts from 0
{
	long int *p, nb, sz;
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
		cout << "knowledge_base::DH_representative k=" << k << " n=" << n
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::DH_representative k=" << k << " n=" << n << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::DH_representative k=" << k << " n=" << n << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void knowledge_base::DH_stab_gens(int k, int n, int i,
		int *&data, int &nb_gens, int &data_size, std::string &stab_order_str)
{
	int *Reps;
	int nb, make_element_size;
	int f, l;
	const char *stab_order;
	
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
		cout << "knowledge_base::DH_representative k=" << k << " n=" << n
				<< " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::DH_representative k=" << k << " n=" << n << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::DH_representative k=" << k << " n=" << n << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
	stab_order_str.assign(stab_order);
}





// #############################################################################
// Spreads:
// #############################################################################




int knowledge_base::Spread_nb_reps(int q, int k)
{
	int nb;

	if (q == 2 && k == 2) {
		nb = Spreads_2_2_nb_reps;
		}
	else if (q == 3 && k == 2) {
		nb = Spreads_3_2_nb_reps;
		}
	else if (q == 2 && k == 4) {
		nb = Spreads_2_4_nb_reps;
		}
	else if (q == 4 && k == 2) {
		nb = Spreads_4_2_nb_reps;
		}
	else if (q == 5 && k == 2) {
		nb = Spreads_5_2_nb_reps;
		}
	else if (q == 3 && k == 3) {
		nb = Spreads_3_3_nb_reps;
		}
	else {
		cout << "knowledge_base::Spread_nb_reps q=" << q << " k=" << k
				<< " I don't have information for this case" << endl;
		exit(1);
		}
	return nb;
}


long int *knowledge_base::Spread_representative(int q, int k, int i, int &sz)
// i starts from 0
{
	long int *p, nb;

	if (q == 2 && k == 2) {
		p = Spreads_2_2_reps;
		nb = Spreads_2_2_nb_reps;
		sz = Spreads_2_2_size;
		}
	else if (q == 3 && k == 2) {
		p = Spreads_3_2_reps;
		nb = Spreads_3_2_nb_reps;
		sz = Spreads_3_2_size;
		}
	else if (q == 2 && k == 4) {
		p = Spreads_2_4_reps;
		nb = Spreads_2_4_nb_reps;
		sz = Spreads_2_4_size;
		}
	else if (q == 4 && k == 2) {
		p = Spreads_4_2_reps;
		nb = Spreads_4_2_nb_reps;
		sz = Spreads_4_2_size;
		}
	else if (q == 5 && k == 2) {
		p = Spreads_5_2_reps;
		nb = Spreads_5_2_nb_reps;
		sz = Spreads_5_2_size;
		}
	else if (q == 3 && k == 3) {
		p = Spreads_3_3_reps;
		nb = Spreads_3_3_nb_reps;
		sz = Spreads_3_3_size;
		}
	else {
		cout << "knowledge_base::Spread_representative q=" << q << " k=" << k
				<< " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::Spread_representative q=" << q << " k=" << k << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::Spread_representative q=" << q << " k=" << k << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += i * sz;
	return p;
}

void knowledge_base::Spread_stab_gens(int q, int k, int i,
		int *&data, int &nb_gens, int &data_size, std::string &stab_order_str)
{
	int *Reps;
	int nb, make_element_size;
	int f, l;
	const char *stab_order;
	
	if (q == 2 && k == 2) {
		Reps = Spreads_2_2_stab_gens;
		nb = Spreads_2_2_nb_reps;
		make_element_size = Spreads_2_2_make_element_size;
		f = Spreads_2_2_stab_gens_fst[i];
		l = Spreads_2_2_stab_gens_len[i];
		stab_order = Spreads_2_2_stab_order[i];
		}
	else if (q == 3 && k == 2) {
		Reps = Spreads_3_2_stab_gens;
		nb = Spreads_3_2_nb_reps;
		make_element_size = Spreads_3_2_make_element_size;
		f = Spreads_3_2_stab_gens_fst[i];
		l = Spreads_3_2_stab_gens_len[i];
		stab_order = Spreads_3_2_stab_order[i];
		}
	else if (q == 2 && k == 4) {
		Reps = Spreads_2_4_stab_gens;
		nb = Spreads_2_4_nb_reps;
		make_element_size = Spreads_2_4_make_element_size;
		f = Spreads_2_4_stab_gens_fst[i];
		l = Spreads_2_4_stab_gens_len[i];
		stab_order = Spreads_2_4_stab_order[i];
		}
	else if (q == 4 && k == 2) {
		Reps = Spreads_4_2_stab_gens;
		nb = Spreads_4_2_nb_reps;
		make_element_size = Spreads_4_2_make_element_size;
		f = Spreads_4_2_stab_gens_fst[i];
		l = Spreads_4_2_stab_gens_len[i];
		stab_order = Spreads_4_2_stab_order[i];
		}
	else if (q == 5 && k == 2) {
		Reps = Spreads_5_2_stab_gens;
		nb = Spreads_5_2_nb_reps;
		make_element_size = Spreads_5_2_make_element_size;
		f = Spreads_5_2_stab_gens_fst[i];
		l = Spreads_5_2_stab_gens_len[i];
		stab_order = Spreads_5_2_stab_order[i];
		}
	else if (q == 3 && k == 3) {
		Reps = Spreads_3_3_stab_gens;
		nb = Spreads_3_3_nb_reps;
		make_element_size = Spreads_3_3_make_element_size;
		f = Spreads_3_3_stab_gens_fst[i];
		l = Spreads_3_3_stab_gens_len[i];
		stab_order = Spreads_3_3_stab_order[i];
		}
	else {
		cout << "knowledge_base::Spread_representative q=" << q << " k=" << k
				<< " I don't have information for this field order" << endl;
		exit(1);
		}
	if (i < 0) {
		cout << "knowledge_base::Spread_representative q=" << q << " k=" << k << " i=" << i
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (i >= nb) {
		cout << "knowledge_base::Spread_representative q=" << q << " k=" << k << " i=" << i
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
	stab_order_str.assign(stab_order);
}


// #############################################################################
// BLT sets:
// #############################################################################




int knowledge_base::BLT_nb_reps(int q)
{
	int nb;

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
	else if (q == 59) {
		nb = BLT_59_nb_reps;
		}
	else if (q == 61) {
		nb = BLT_61_nb_reps;
		}
	else if (q == 67) {
		nb = BLT_67_nb_reps;
		}
	else if (q == 71) {
		nb = BLT_71_nb_reps;
		}
	else if (q == 73) {
		nb = BLT_73_nb_reps;
		}
	else {
		cout << "knowledge_base::BLT_nb_reps q=" << q
				<< " I don't have information for this order" << endl;
		exit(1);
		}
	return nb;
}

long int *knowledge_base::BLT_representative(int q, int no)
// i starts from 0
{
	long int *p;
	int nb, sz;

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
	else if (q == 59) {
		p = BLT_59_reps;
		nb = BLT_59_nb_reps;
		sz = BLT_59_size;
		}
	else if (q == 61) {
		p = BLT_61_reps;
		nb = BLT_61_nb_reps;
		sz = BLT_61_size;
		}
	else if (q == 67) {
		p = BLT_67_reps;
		nb = BLT_67_nb_reps;
		sz = BLT_67_size;
		}
	else if (q == 71) {
		p = BLT_71_reps;
		nb = BLT_71_nb_reps;
		sz = BLT_71_size;
		}
	else if (q == 73) {
		p = BLT_73_reps;
		nb = BLT_73_nb_reps;
		sz = BLT_73_size;
		}
	else {
		cout << "knowledge_base::BLT_representative q=" << q
				<< " I don't have information for this field order" << endl;
		exit(1);
		}
	if (no < 0) {
		cout << "knowledge_base::BLT_representative q=" << q << " no=" << no
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (no >= nb) {
		cout << "knowledge_base::BLT_representative q=" << q << " no=" << no
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	p += no * sz;
	return p;
}

void knowledge_base::BLT_stab_gens(int q, int no,
		int *&data, int &nb_gens, int &data_size, std::string &stab_order_str)
{
	int *Reps;
	int nb, make_element_size;
	int f, l;
	const char *stab_order;
	
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
	else if (q == 59) {
		Reps = BLT_59_stab_gens;
		nb = BLT_59_nb_reps;
		make_element_size = BLT_59_make_element_size;
		f = BLT_59_stab_gens_fst[no];
		l = BLT_59_stab_gens_len[no];
		stab_order = BLT_59_stab_order[no];
		}
	else if (q == 61) {
		Reps = BLT_61_stab_gens;
		nb = BLT_61_nb_reps;
		make_element_size = BLT_61_make_element_size;
		f = BLT_61_stab_gens_fst[no];
		l = BLT_61_stab_gens_len[no];
		stab_order = BLT_61_stab_order[no];
		}
	else if (q == 67) {
		Reps = BLT_67_stab_gens;
		nb = BLT_67_nb_reps;
		make_element_size = BLT_67_make_element_size;
		f = BLT_67_stab_gens_fst[no];
		l = BLT_67_stab_gens_len[no];
		stab_order = BLT_67_stab_order[no];
		}
	else if (q == 71) {
		Reps = BLT_71_stab_gens;
		nb = BLT_71_nb_reps;
		make_element_size = BLT_71_make_element_size;
		f = BLT_71_stab_gens_fst[no];
		l = BLT_71_stab_gens_len[no];
		stab_order = BLT_71_stab_order[no];
		}
	else if (q == 73) {
		Reps = BLT_73_stab_gens;
		nb = BLT_73_nb_reps;
		make_element_size = BLT_73_make_element_size;
		f = BLT_73_stab_gens_fst[no];
		l = BLT_73_stab_gens_len[no];
		stab_order = BLT_73_stab_order[no];
		}
	else {
		cout << "knowledge_base::BLT_representative q=" << q << " I don't have "
				"information for this field order" << endl;
		exit(1);
		}
	if (no < 0) {
		cout << "knowledge_base::BLT_representative q=" << q << " no=" << no
				<< " but i must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
		}
	if (no >= nb) {
		cout << "knowledge_base::BLT_representative q=" << q << " no=" << no
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
		}
	nb_gens = l;
	data_size = make_element_size;
	data = Reps + f * make_element_size;
	stab_order_str.assign(stab_order);
}




void knowledge_base::override_polynomial_subfield(std::string &poly, int q)
{
	const char *override_poly = NULL;
	int p, h;
	number_theory::number_theory_domain NT;
	
	if (!NT.is_prime_power(q, p, h)) {
		cout << "knowledge_base::override_polynomial_subfield "
				"q is not a prime power" << endl;
		exit(1);
		}
	if (h == 1) {
		poly.assign("");
		return;
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
		cout << "knowledge_base::override_polynomial_subfield, "
				"do not have a polynomial for q=" << q << endl;
		exit(1);

#if 0
		int verbose_level = 2;
		finite_field f, F;
		int qq = q * q;
		
		cout << "knowledge_base::override_polynomial_subfield initializing large field" << endl;
		F.finite_field_init(qq, FALSE /* f_without_tables */, verbose_level);
		cout << "knowledge_base::override_polynomial_subfield initializing small field" << endl;
		f.finite_field_init(q, FALSE /* f_without_tables */, verbose_level);
		if (f.e > 1) {
			F.finite_field_init(qq, FALSE /* f_without_tables */, 1);
			f.finite_field_init(q, FALSE /* f_without_tables */, 3);
			cout << "knowledge_base::override_polynomial_subfield need to choose the generator "
					"polynomial for the field" << endl;
			F.compute_subfields(verbose_level);
			//exit(1);
			}
#endif
		//return NULL;
		}
	poly.assign(override_poly);
}

void knowledge_base::override_polynomial_extension_field(std::string &poly, int q)
{
	const char *override_poly = NULL;
	int p, h;
	number_theory::number_theory_domain NT;
	
	if (!NT.is_prime_power(q, p, h)) {
		cout << "knowledge_base::override_polynomial_extension_field "
				"q is not a prime power" << endl;
		exit(1);
		}
	if (h == 1) {
		get_primitive_polynomial(poly, q, 2, 0/*verbose_level*/); // ToDo
		return;
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
		cout << "knowledge_base::override_polynomial_extension_field, "
				"do not have a polynomial for q=" << q << endl;
		exit(1);
	}
	poly.assign(override_poly);
}

#if 0
	if (q == 9) {
		char *override_poly_Q = "110"; // X^{4} + X^{3} + 2
		char *override_poly_q = "17"; // X^2 - X - 1 = X^2 +2X + 2 = 2 + 2*3 + 9 = 17
		//finite_field::init_override_polynomial
		// GF(81) = GF(3^4), polynomial = X^{4} + X^{3} + 2 = 110
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
		char *override_poly_Q = "767"; // X^{4} + X^{3} + 3X + 2
		char *override_poly_q = "47"; // X^2 - X - 3 = X^2 +4X + 2=25+20+2=47
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
		char *override_poly_Q = "974"; // X^{6} + X^{5} + 2
		char *override_poly_q = "34"; // X^3 - X + 1 = X^3 +2X + 1 = 27+6+1=34
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
		char *override_poly_Q = "2754"; // X^{4} + X^{3} + X + 3
		char *override_poly_q = "94"; // X^2-X+3 = X^2+6X+3 = 49+6*7+3=94
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
		char *override_poly_Q = "6590"; // X^{8} + X^{3} + 2
		char *override_poly_q = "89"; // X^4-X-1=X^4+2X+2=81+2*3+2=89
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
		char *override_poly_Q = "15985"; // X^{4} + X^{3} + X + 2
		char *override_poly_q = "200"; // X^2-4X+2=X^2+7X+2=11^2+7*11+2=200
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




// #############################################################################
// projective planes
// #############################################################################

void knowledge_base::get_projective_plane_list_of_lines(
		int *&list_of_lines,
		int &order, int &nb_lines, int &line_size,
		const char *label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "knowledge_base::get_projective_plane_list_of_lines" << endl;
	}
	if (strcmp(label, "Mathon_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_mathon[i] - 1;
		}
	}
	else if (strcmp(label, "Semi4_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_semi4[i] - 1;
		}
	}
	else if (strcmp(label, "Semi2_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_semi2[i] - 1;
		}
	}
	else if (strcmp(label, "PG_2_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_PG_2_16[i] - 1;
		}
	}
	else if (strcmp(label, "Lmrh_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_Lmrh[i] - 1;
		}
	}
	else if (strcmp(label, "Jowk_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_jowk[i] - 1;
		}
	}
	else if (strcmp(label, "Jo_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_johnson[i] - 1;
		}
	}
	else if (strcmp(label, "Hall_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_Hall[i] - 1;
		}
	}
	else if (strcmp(label, "Dsfp_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_dsfp[i] - 1;
		}
	}
	else if (strcmp(label, "Demp_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_demp[i] - 1;
		}
	}
	else if (strcmp(label, "Bbh1_16") == 0) {
		order = 16;
		nb_lines = order * order + order + 1;
		line_size = order + 1;
		list_of_lines = NEW_int(nb_lines * line_size);
		for (i = 0; i < nb_lines * line_size; i++) {
			list_of_lines[i] = plane_bbh1[i] - 1;
		}
	}
	else {
		cout << "knowledge_base::get_projective_plane_list_of_lines unrecognized type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "knowledge_base::get_projective_plane_list_of_lines done" << endl;
	}
}


// #############################################################################
// tensor orbits:
// #############################################################################


int knowledge_base::tensor_orbits_nb_reps(int n)
{
	int nb;

	if (n == 4) {
		nb = data_tensor_nb_w4_reps;
	}
	else if (n == 5) {
		nb = data_tensor_nb_w5_reps;
	}
	else {
		cout << "knowledge_base::tensor_orbits_nb_reps n=" << n
				<< " I don't have information for this case" << endl;
		exit(1);
	}
	return nb;
}

long int *knowledge_base::tensor_orbits_rep(int n, int idx)
// idx starts from 0
{
	long int *p;
	int nb, sz = 3;

	if (n == 4) {
		p = data_tensor_w4_reps;
		nb = data_tensor_nb_w4_reps;
	}
	else if (n == 5) {
		p = data_tensor_w5_reps;
		nb = data_tensor_nb_w5_reps;
	}
	else {
		cout << "knowledge_base::tensor_orbits_rep n=" << n
				<< " I don't have information for this case" << endl;
		exit(1);
	}
	if (idx < 0) {
		cout << "knowledge_base::tensor_orbits_rep n=" << n << " idx=" << idx
				<< " but idx must be at least 0 (numbering starts at 0)" << endl;
		exit(1);
	}
	if (idx >= nb) {
		cout << "knowledge_base::tensor_orbits_rep n=" << n << " idx=" << idx
				<< " but I have only " << nb << " representatives" << endl;
		exit(1);
	}
	p += idx * sz;
	return p;
}

void knowledge_base::retrieve_BLT_set_from_database_embedded(
		orthogonal_geometry::quadratic_form *Quadratic_form,
		int BLT_k,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::retrieve_BLT_set_from_database_embedded" << endl;
	}

	retrieve_BLT_set_from_database(
			Quadratic_form,
			TRUE /* f_embedded */,
			BLT_k,
			label_txt,
			label_tex,
			nb_pts, Pts,
			verbose_level);

	if (f_v) {
		cout << "finite_field::retrieve_BLT_set_from_database_embedded done" << endl;
	}
}

void knowledge_base::retrieve_BLT_set_from_database(
		orthogonal_geometry::quadratic_form *Quadratic_form,
		int f_embedded,
		int BLT_k,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::retrieve_BLT_set_from_database" << endl;
	}
	int i;
	long int j;
	//int epsilon = 0;
	//int n = 4;
	int d = 5;
	long int *BLT;
	int *v;
	//knowledge_base K;

	nb_pts = Quadratic_form->F->q + 1;

	BLT = BLT_representative(Quadratic_form->F->q, BLT_k);

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (f_v) {
		cout << "i : orthogonal rank : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		//Quadratic_form->Orthogonal_indexing->Q_epsilon_unrank(v, 1,
		//		epsilon, n,
		//		Quadratic_form->form_c1,
		//		Quadratic_form->form_c2,
		//		Quadratic_form->form_c3,
		//		BLT[i], 0 /* verbose_level */);
		Quadratic_form->unrank_point(v, BLT[i], 0 /* verbose_level */);
		if (f_embedded) {
			Quadratic_form->F->PG_element_rank_modified_lint(v, 1, d, j);
		}
		else {
			j = BLT[i];
		}
		// recreate v:
		//Quadratic_form->Orthogonal_indexing->Q_epsilon_unrank(v, 1,
		//		epsilon, n,
		//		Quadratic_form->form_c1,
		//		Quadratic_form->form_c2,
		//		Quadratic_form->form_c3,
		//		BLT[i], 0 /* verbose_level */);
		Quadratic_form->unrank_point(v, BLT[i], 0 /* verbose_level */);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : " << setw(4) << BLT[i] << " : ";
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
		}
	}

#if 0
	cout << "list of points:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];

	if (f_embedded) {
		snprintf(str, sizeof(str), "%d_%d_embedded", Quadratic_form->F->q, BLT_k);
		label_txt.assign("BLT_");
		label_txt.append(str);

		snprintf(str, sizeof(str), "%d\\_%d\\_embedded", Quadratic_form->F->q, BLT_k);
		label_tex.assign("BLT\\_");
		label_tex.append(str);
	}
	else {
		snprintf(str, sizeof(str), "%d_%d", Quadratic_form->F->q, BLT_k);
		label_txt.assign("BLT_");
		label_txt.append(str);

		snprintf(str, sizeof(str), "%d\\_%d", Quadratic_form->F->q, BLT_k);
		label_tex.assign("BLT\\_");
		label_tex.append(str);
	}
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_int(v);
	//FREE_int(L);
	//delete F;
	if (f_v) {
		cout << "finite_field::retrieve_BLT_set_from_database done" << endl;
	}
}



}
}


