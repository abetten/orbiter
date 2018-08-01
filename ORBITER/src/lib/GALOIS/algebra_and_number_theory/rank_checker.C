// rank_checker.C
//
// Anton Betten
//
// moved here from projective:  May 10, 2009




#include "galois.h"

INT rank_checker::cntr_new = 0;
INT rank_checker::cntr_objects = 0;
INT rank_checker::f_debug_memory = FALSE;

void *rank_checker::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "rank_checker::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *rank_checker::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(rank_checker);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "rank_checker::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void rank_checker::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "rank_checker::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return free(ptr);
}

void rank_checker::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(rank_checker);
	if (f_debug_memory) {
		cout << "rank_checker::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return free(ptr);
}


rank_checker::rank_checker()
{
	M1 = NULL;
	M2 = NULL;
	base_cols = NULL;
	set = NULL;
}

rank_checker::~rank_checker()
{
	//cout << "in ~rank_checker()" << endl;
	if (M1)
		FREE_INT(M1);
	if (M2)
		FREE_INT(M2);
	if (base_cols)
		FREE_INT(base_cols);
	if (set)
		FREE_INT(set);
	//cout << "~rank_checker() finished" << endl;
}

void rank_checker::init(finite_field *GFq, INT m, INT n, INT d)
{
	rank_checker::GFq = GFq;
	rank_checker::m = m;
	rank_checker::n = n;
	rank_checker::d = d;
	M1 = NEW_INT(m * n);
	M2 = NEW_INT(m * n);
	base_cols = NEW_INT(n);
	set = NEW_INT(n);
}

INT rank_checker::check_rank(INT len, INT *S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, aj, rk, f_OK = TRUE;
	
	if (f_v) {
		cout << "rank_checker::check_rank: checking the set ";
		print_set(cout, len, S);
		cout << endl;
		}
	// M1 will be used as a m x len matrix
	for (j = 0; j < len; j++) {
		PG_element_unrank_modified(*GFq, M1 + j, len /* stride */, m /* len */, S[j]);
		}
	if (f_vv) {
		cout << "\n";
		//print_integer_matrix(cout, gen.S, 1, len);
		print_integer_matrix(cout, M1, m, len);
		}
	if (len <= 1)
		return TRUE;
	if (d <= 1)
		return TRUE;
	INT d1 = MINIMUM(d - 2, len  - 1);
	if (f_vv) {
		cout << "d1=" << d1 << endl;
		}


	// M2 will be used as a m x (d1 + 1) matrix	
	
	first_k_subset(set, len - 1, d1);
	while (TRUE) {
	
		// get the subset of columns:
		if (f_vv) {
			cout << "subset: ";
			print_set(cout, d1, set);
			cout << endl;
			}
		
		for (j = 0; j < d1; j++) {
			aj = set[j];
			for (i = 0; i < m; i++) {
				M2[i * (d1 + 1) + j] = M1[i * len + aj];
				}
			}
		for (i = 0; i < m; i++) {
			M2[i * (d1 + 1) + d1] = M1[i * len + len - 1];
			}
		if (FALSE) {
			print_integer_matrix(cout, M2, m, d1 + 1);
			}
		
		rk = GFq->Gauss_INT(M2, FALSE /* f_special */, FALSE /* f_complete */, base_cols, 
			FALSE /* f_P */, NULL, m /* m */, d1 + 1 /* n */, 0 /* Pn */, 
			0 /* verbose_level */);
		if (rk <= d1) {
			f_OK = FALSE;
			if (f_v) {
				cout << "not OK; subset: ";
				print_set(cout, d1, set);
				cout << " leads to a rk " << rk << " submatrix" << endl;
				}
			break;
			}
		if (!next_k_subset(set, len - 1, d1))
			break;
		}
	if (!f_OK)
		return FALSE;
	return TRUE;
}

INT rank_checker::check_rank_matrix_input(INT len, INT *S, INT dim_S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, aj, rk, f_OK = TRUE;
	
	// S is a m x len matrix
	if (len <= 1)
		return TRUE;
	if (d <= 1)
		return TRUE;
	INT d1 = MINIMUM(d - 2, len  - 1);
	if (f_vv) {
		cout << "d1=" << d1 << endl;
		}


	// M2 will be used as a m x (d1 + 1) matrix	
	
	first_k_subset(set, len - 1, d1);
	while (TRUE) {
	
		// get the subset of columns:
		if (f_vv) {
			cout << "subset: ";
			print_set(cout, d1, set);
			cout << endl;
			}
		
		for (j = 0; j < d1; j++) {
			aj = set[j];
			for (i = 0; i < m; i++) {
				M2[i * (d1 + 1) + j] = S[i * dim_S + aj];
				}
			}
		for (i = 0; i < m; i++) {
			M2[i * (d1 + 1) + d1] = S[i * dim_S + len - 1];
			}
		
		rk = GFq->Gauss_INT(M2, FALSE /* f_special */, FALSE /* f_complete */, base_cols, 
			FALSE /* f_P */, NULL, m /* m */, d1 + 1 /* n */, 0 /* Pn */, 
			0 /* verbose_level */);
		if (rk <= d1) {
			f_OK = FALSE;
			if (f_v) {
				cout << "not OK; subset: ";
				print_set(cout, d1, set);
				cout << " leads to a rk " << rk << " submatrix, but we want rank " << d1 + 1 << endl;
				}
			break;
			}
		if (!next_k_subset(set, len - 1, d1))
			break;
		}
	if (!f_OK)
		return FALSE;
	return TRUE;
}

INT rank_checker::check_rank_last_two_are_fixed(INT len, INT *S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, aj, rk, f_OK = TRUE;
	
	if (f_v) {
		cout << "rank_checker::check_rank_last_two_are_fixed: checking the set ";
		print_set(cout, len, S);
		cout << endl;
		}
	// M1 will be used as a m x len matrix
	for (j = 0; j < len; j++) {
		PG_element_unrank_modified(*GFq, M1 + j, len /* stride */, m /* len */, S[j]);
		}
	if (f_vv) {
		cout << "\n";
		//print_integer_matrix(cout, gen.S, 1, len);
		print_integer_matrix(cout, M1, m, len);
		}
	if (len <= 1)
		return TRUE;
	if (d <= 2)
		return TRUE;
	INT d1 = MINIMUM(d - 3, len  - 2);
	if (f_vv) {
		cout << "d1=" << d1 << endl;
		}


	// M2 will be used as a m x (d1 + 2) matrix	
	
	first_k_subset(set, len - 2, d1);
	while (TRUE) {
	
		// get the subset of columns:
		if (f_vv) {
			cout << "subset: ";
			print_set(cout, d1, set);
			cout << endl;
			}
		
		for (j = 0; j < d1; j++) {
			aj = set[j];
			for (i = 0; i < m; i++) {
				M2[i * (d1 + 2) + j] = M1[i * len + aj];
				}
			}
		for (i = 0; i < m; i++) {
			M2[i * (d1 + 2) + d1] = M1[i * len + len - 2];
			M2[i * (d1 + 2) + d1 + 1] = M1[i * len + len - 1];
			}
		if (FALSE) {
			print_integer_matrix(cout, M2, m, d1 + 2);
			}
		
		rk = GFq->Gauss_INT(M2, FALSE /* f_special */, FALSE /* f_complete */, base_cols, 
			FALSE /* f_P */, NULL, m /* m */, d1 + 2 /* n */, 0 /* Pn */, 
			0 /* verbose_level */);
		if (rk <= d1 + 1) {
			f_OK = FALSE;
			if (f_v) {
				cout << "not OK; subset: ";
				print_set(cout, d1, set);
				cout << " leads to a rk " << rk << " submatrix" << endl;
				}
			break;
			}
		if (!next_k_subset(set, len - 2, d1))
			break;
		}
	if (!f_OK)
		return FALSE;
	if (f_v)
		cout << "is OK" << endl;
	return TRUE;
}

INT rank_checker::compute_rank_row_vectors(INT len, INT *S, INT f_projective, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT j, rk;
	
	if (f_vv) {
		cout << "rank_checker::compute_rank_row_vectors set ";
		print_set(cout, len, S);
		cout << endl;
		}
	// M1 will be used as a len x n matrix
	for (j = 0; j < len; j++) {
		if (f_projective) {
			PG_element_unrank_modified(*GFq, M1 + j * n, 1 /* stride */, n /* len */, S[j]);
			}
		else {
			AG_element_unrank(GFq->q, M1 + j * n, 1, n, S[j]);
			}
		}
	if (f_v) {
		cout << "\n";
		//print_integer_matrix(cout, gen.S, 1, len);
		print_integer_matrix(cout, M1, len, n);
		}

		
	rk = GFq->Gauss_INT(M1, FALSE /* f_special */, FALSE /* f_complete */, base_cols, 
		FALSE /* f_P */, NULL, len /* m */, n /* n */, 0 /* Pn */, 
		0 /* verbose_level */);

	return rk;
}

