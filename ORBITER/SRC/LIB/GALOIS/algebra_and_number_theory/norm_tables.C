// norm_tables.C
// 
// Anton Betten
// 11/28/2008
//
//
// 
//
//

#include "galois.h"

norm_tables::norm_tables()
{
	norm_table = NULL;
	norm_table_sorted = NULL;
	sorting_perm = NULL;
	sorting_perm_inv = NULL;
	type_first = NULL;
	type_len = NULL;
	the_type = NULL;
}

norm_tables::~norm_tables()
{
	if (norm_table) {
		delete [] norm_table;
		norm_table = NULL;
		}
	if (norm_table_sorted) {
		delete [] norm_table_sorted;
		norm_table_sorted = NULL;
		}
	if (sorting_perm) {
		delete [] sorting_perm;
		sorting_perm = NULL;
		}
	if (sorting_perm_inv) {
		delete [] sorting_perm_inv;
		sorting_perm_inv = NULL;
		}
	if (type_first) {
		delete [] type_first;
		type_first = NULL;
		}
	if (type_len) {
		delete [] type_len;
		type_len = NULL;
		}
	if (the_type) {
		delete [] the_type;
		the_type = NULL;
		}
}

void norm_tables::init(unusual_model &U, INT verbose_level)
{
	INT qq = U.F.q;
	INT i, f, l, j, a, b, c, jj;
	
	norm_table = new INT[qq];
	for (i = 1; i < qq; i++) {
		norm_table[i - 1] = U.N2(i);
		}
	
	INT_vec_classify(qq - 1, norm_table, norm_table_sorted, 
		sorting_perm, sorting_perm_inv, 
		nb_types, type_first, type_len);
	
	//cout << "nb_types=" << NT.nb_types << endl;
	the_type = new INT[nb_types];
	for (i = 0; i < nb_types; i++) {
		f = type_first[i];
		l = type_len[i];
		//cout << "type " << i << " f=" << f << " l=" << l << endl;
		for (j = 0; j < l; j++) {
			jj = f + j;
			a = sorting_perm_inv[jj];
			b = a + 1;
			c = U.N2(b);
			//cout << "j=" << j << " a=" << a << " b=" << b << " N2(b)=" << c << endl;
			if (j == 0) {
				the_type[i] = c;
				}
			}
		}

}

INT norm_tables::choose_an_element_of_given_norm(INT norm, INT verbose_level)
{
	INT idx, f, gamma;
	
	INT_vec_search(the_type, nb_types, norm, idx);
	f = type_first[idx];
	//l = type_len[idx];
	gamma = sorting_perm_inv[f + 0] + 1;
	return gamma;
}


