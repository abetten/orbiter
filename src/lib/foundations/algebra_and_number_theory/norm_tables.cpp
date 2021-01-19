// norm_tables.cpp
// 
// Anton Betten
// 11/28/2008
//
//
// 
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


norm_tables::norm_tables()
{
	norm_table = NULL;
	norm_table_sorted = NULL;
	sorting_perm = NULL;
	sorting_perm_inv = NULL;
	nb_types = 0;
	type_first = NULL;
	type_len = NULL;
	the_type = NULL;
}

norm_tables::~norm_tables()
{
	if (norm_table) {
		FREE_int(norm_table);
		norm_table = NULL;
		}
	if (norm_table_sorted) {
		FREE_int(norm_table_sorted);
		norm_table_sorted = NULL;
		}
	if (sorting_perm) {
		FREE_int(sorting_perm);
		sorting_perm = NULL;
		}
	if (sorting_perm_inv) {
		FREE_int(sorting_perm_inv);
		sorting_perm_inv = NULL;
		}
	if (type_first) {
		FREE_int(type_first);
		type_first = NULL;
		}
	if (type_len) {
		FREE_int(type_len);
		type_len = NULL;
		}
	if (the_type) {
		FREE_int(the_type);
		the_type = NULL;
		}
}

void norm_tables::init(unusual_model &U, int verbose_level)
{
	int qq = U.FQ->q;
	int i, f, l, j, a, b, c, jj;
	sorting Sorting;
	
	norm_table = NEW_int(qq);
	for (i = 1; i < qq; i++) {
		norm_table[i - 1] = U.N2(i);
		}
	
	Sorting.int_vec_classify(qq - 1, norm_table, norm_table_sorted,
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
	
	//cout << "nb_types=" << NT.nb_types << endl;
	the_type = NEW_int(nb_types);
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		//cout << "type " << i << " f=" << f << " l=" << l << endl;
		for (j = 0; j < l; j++) {
			jj = f + j;
			a = sorting_perm_inv[jj];
			b = a + 1;
			c = U.N2(b);
			//cout << "j=" << j << " a=" << a << " b=" << b
			// << " N2(b)=" << c << endl;
			if (j == 0) {
				the_type[i] = c;
				}
			}
		}

}

int norm_tables::choose_an_element_of_given_norm(
		int norm, int verbose_level)
{
	int idx, f, gamma;
	sorting Sorting;
	
	Sorting.int_vec_search(the_type, nb_types, norm, idx);
	f = type_first[idx];
	//l = type_len[idx];
	gamma = sorting_perm_inv[f + 0] + 1;
	return gamma;
}

}
}

