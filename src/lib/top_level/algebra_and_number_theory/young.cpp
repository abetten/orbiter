// young.cpp
// 
// Anton Betten
//
// started: March 16, 2015
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

young::young()
{
	null();
}

young::~young()
{
	freeself();
}

void young::null()
{
	A = NULL;
	Elt = NULL;
	v = NULL;
	Aconj = NULL;
	ABC = NULL;
	Sch = NULL;
	SG = NULL;
	class_size = NULL;
	class_rep = NULL;
	D = NULL;

	row_parts = NULL;
	col_parts = NULL;
	Tableau = NULL;

	Row_partition = NULL;
	Col_partition = NULL;

	gens1 = NULL;
	gens2 = NULL;
	S1 = NULL;
	S2 = NULL;
}

void young::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "young::freeself" << endl;
		}

	if (f_v) {
		cout << "young::freeself before delete Sch" << endl;
		}
	if (Sch) {
		delete Sch;
		}
	if (f_v) {
		cout << "young::freeself before delete SG" << endl;
		}
	if (SG) {
		delete SG;
		}
	if (class_size) {
		FREE_int(class_size);
		}
	if (class_rep) {
		FREE_int(class_rep);
		}
	if (f_v) {
		cout << "young::freeself before delete D" << endl;
		}
	if (D) {
		delete D;
		}
	if (col_parts) {
		FREE_int(col_parts);
		}
	if (Tableau) {
		FREE_int(Tableau);
		}
	if (Row_partition) {
		delete Row_partition;
		}
	if (Col_partition) {
		delete Col_partition;
		}
	if (f_v) {
		cout << "young::freeself before delete gens1" << endl;
		}
	if (gens1) {
		delete gens1;
		}
	if (gens2) {
		delete gens2;
		}
	if (f_v) {
		cout << "young::freeself before delete S1" << endl;
		}
	if (S1) {
		delete S1;
		}
	if (S2) {
		delete S2;
		}


	if (f_v) {
		cout << "young::freeself before delete A" << endl;
		}
	if (A) {
		delete A;
		}
	if (Elt) {
		FREE_int(Elt);
		}
	if (v) {
		FREE_int(v);
		}

	if (f_v) {
		cout << "young::freeself before delete Aconj" << endl;
		}
	if (Aconj) {
		delete Aconj;
		}
#if 0
	if (f_v) {
		cout << "young::freeself before delete ABC" << endl;
		}
	if (ABC) {
		delete ABC;
		}
#endif
	if (f_v) {
		cout << "young::freeself done" << endl;
		}
}

void young::init(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "young::init" << endl;
		}
	young::n = n;
	A = NEW_OBJECT(action);
	A->init_symmetric_group(n, verbose_level);
	A->group_order(go);

	goi = go.as_int();

	if (f_v) {
		cout << "Created group Sym(" << n << ") of size " << goi << endl;
		}

	Elt = NEW_int(A->elt_size_in_int);

	v = NEW_int(n);

	S = A->Sims;

	if (f_vv) {
		cout << "Listing all elements in the group "
				"Sym(" << n << "):" << endl;
		for (i = 0; i < goi; i++) {
			S->element_unrank_lint(i, Elt);
			cout << "element " << i << " is ";
			A->element_print_quick(Elt, cout);
			cout << endl;
			}
		}


	if (f_v) {
		cout << "computing conjugacy classes:" << endl;
		}

	S->compute_conjugacy_classes(Aconj, ABC, Sch,
		SG, nb_classes, class_size, class_rep, 
		verbose_level);

	if (f_v) {
		cout << "computing conjugacy classes done" << endl;
		}



	D = NEW_OBJECT(a_domain);
	D->init_integer_fractions(verbose_level);

	if (f_v) {
		cout << "young::init done" << endl;
		}
}

void young::create_module(int *h_alpha, 
	int *&Base, int *&base_cols, int &rk, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "young::create_module" << endl;
		}

	int sz;


	goi = S->group_order_lint();
	sz = group_ring_element_size(A, S);
	

	int *M1;

	M1 = NEW_int(goi * sz);
	Base = NEW_int(goi * sz * D->size_of_instance_in_int);
	for (j = 0; j < sz; j++) {
		M1[0 * sz + j] = h_alpha[j];
		}

	int *elt4, *elt5;

	group_ring_element_create(A, S, elt4);
	group_ring_element_create(A, S, elt5);


	for (i = 1; i < goi; i++) {
		group_ring_element_zero(A, S, elt4);
		elt4[i] = 1;
		group_ring_element_mult(A, S, h_alpha, elt4, elt5);
		for (j = 0; j < sz; j++) {
			M1[i * sz + j] = elt5[j];
			}
		}


	if (FALSE) {
		cout << "M1=" << endl;
		int_matrix_print(M1, goi, sz);
		}

	for (i = 0; i < goi * sz; i++) {
		D->make_integer(D->offset(Base, i), M1[i], 0 /* verbose_level*/);
		}

	if (f_v) {
		cout << "A basis is:" << endl;
		D->print_matrix(Base, goi, sz);
		}

	int f_special = FALSE;
	int f_complete = TRUE;
	int f_P = FALSE;
	
	base_cols = NEW_int(sz);

	if (f_v) {
		cout << "Calling Gauss_echelon_form:" << endl;
		}

	rk = D->Gauss_echelon_form(Base, f_special, f_complete, base_cols, 
		f_P, NULL, goi, sz, goi, 0 /*verbose_level*/);

	if (f_v) {
		cout << "rk=" << rk << endl;
		cout << "Basis=" << endl;
		D->print_matrix(Base, rk, sz);
		}

	if (f_v) {
		cout << "base_cols=" << endl;
		int_vec_print(cout, base_cols, rk);
		cout << endl;
		}

	
	FREE_int(M1);
	FREE_int(elt4);
	FREE_int(elt5);
	
	if (f_v) {
		cout << "young::create_module done" << endl;
		}
}

void young::create_representations(int *Base,
		int *base_cols, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v3 = (verbose_level >= 3);
	int i, j, h, sz;

	if (f_v) {
		cout << "young::create_representations" << endl;
		}
	int ii, idx, c;

	int *Mtx;
	int *M4;
	int *elt3, *elt4, *elt5;

	sz = group_ring_element_size(A, S);

	Mtx = NEW_int(rk * rk * D->size_of_instance_in_int);
	M4 = NEW_int((rk + 1) * sz * D->size_of_instance_in_int);

	group_ring_element_create(A, S, elt3);
	group_ring_element_create(A, S, elt4);
	group_ring_element_create(A, S, elt5);

	for (c = 0; c < nb_classes; c++) {


		h = class_rep[c];

	//for (c = -1, h = 0; h < sz; h++) {


		for (i = 0; i < rk; i++) {
			for (j = 0; j < sz; j++) {
				elt3[j] = D->as_int(D->offset(Base, i * sz + j), 0);
				}
			group_ring_element_zero(A, S, elt4);
			elt4[h] = 1;
			group_ring_element_mult(A, S, elt3, elt4, elt5);

			for (ii = 0; ii < rk * sz; ii++) {
				D->copy(D->offset(Base, ii), D->offset(M4, ii), 0);
				}
			for (j = 0; j < sz; j++) {
				D->make_integer(D->offset(M4, rk * sz + j),
						elt5[j], 0 /* verbose_level*/);
				}
			
			
			if (f_v3) {
				cout << "The extended matrix is:" << endl;
				D->print_matrix(M4, rk + 1, sz);
				}
			

			for (ii = 0; ii < rk; ii++) {
				idx = base_cols[ii];
				D->copy(D->offset(M4, rk * sz + idx),
						D->offset(Mtx, i * rk + ii), 0);
				}
			
			for (ii = 0; ii < rk; ii++) {
				idx = base_cols[ii];
				D->Gauss_step(D->offset(M4, ii * sz),
						D->offset(M4, rk * sz), sz, idx,
						0 /* verbose_level */);
				if (f_v3) {
					cout << "Row " << i << " basis vector " << ii
							<< " has been subtracted" << endl;
					D->print_matrix(M4, rk + 1, sz);
					}

				}
			if (!D->is_zero_vector(D->offset(M4, rk * sz), sz,
					0 /* verbose_level */)) {
				cout << "The vector does not lie in the span of the "
						"basis, something is wrong" << endl;
				exit(1);
				}

			if (f_v3) {
				cout << "Row " << i << " has been computed" << endl;
				}
			
			} // next i


		cout << "Conjugacy class " << c << " is represented "
				"by group element " << h;
		S->element_unrank_lint(h, Elt);
		cout << " which is ";
		A->element_print(Elt, cout);
		cout << " which is represented by the matrix:" << endl;
		D->print_matrix(Mtx, rk, rk);
		cout << endl;

			
		} // next c
	

	FREE_int(elt3);
	FREE_int(elt4);
	FREE_int(elt5);
	FREE_int(Mtx);

	if (f_v) {
		cout << "young::create_representations done" << endl;
		}
}

void young::create_representation(int *Base,
		int *Base_inv, int rk, int group_elt, int *Mtx,
		int verbose_level)
// Mtx[rk * rk * D->size_of_instance_in_int]
{
	int f_v = (verbose_level >= 1);
	int f_v3 = (verbose_level >= 3);
	int i, j, sz;

	if (f_v) {
		cout << "young::create_representation" << endl;
		}
	int ii;

	int *M4;
	int *elt3, *elt4, *elt5;

	sz = group_ring_element_size(A, S);

	M4 = NEW_int((rk + 1) * sz * D->size_of_instance_in_int);

	group_ring_element_create(A, S, elt3);
	group_ring_element_create(A, S, elt4);
	group_ring_element_create(A, S, elt5);


	for (i = 0; i < rk; i++) {
		for (j = 0; j < sz; j++) {
			elt3[j] = D->as_int(D->offset(Base, i * sz + j), 0);
			}
		group_ring_element_zero(A, S, elt4);
		elt4[group_elt] = 1;
		group_ring_element_mult(A, S, elt3, elt4, elt5);

		for (ii = 0; ii < rk * sz; ii++) {
			D->copy(D->offset(Base, ii), D->offset(M4, ii), 0);
			}
		for (j = 0; j < sz; j++) {
			D->make_integer(D->offset(M4, rk * sz + j), elt5[j],
					0 /* verbose_level*/);
			}
		
		
		if (f_v3) {
			cout << "The extended matrix is:" << endl;
			D->print_matrix(M4, rk + 1, sz);
			}
		
		D->mult_matrix(D->offset(M4, rk * sz), Base_inv,
				D->offset(Mtx, i * rk), 1, rk, rk, 0);

#if 0
		for (ii = 0; ii < rk; ii++) {
			idx = base_cols[ii];
			D->copy(D->offset(M4, rk * sz + idx),
					D->offset(Mtx, i * rk + ii), 0);
			D->Gauss_step(D->offset(M4, ii * sz),
					D->offset(M4, rk * sz), sz, idx,
					0 /* verbose_level */);
			if (f_v3) {
				cout << "Row " << i << " basis vector " << ii
						<< " has been subtracted" << endl;
				D->print_matrix(M4, rk + 1, sz);
				}

			}
		if (!D->is_zero_vector(D->offset(M4, rk * sz), sz,
				0 /* verbose_level */)) {
			cout << "The vector does not lie in the span of the "
					"basis, something is wrong" << endl;
			exit(1);
			}
#endif

		if (f_v3) {
			cout << "Row " << i << " has been computed" << endl;
			}
		
		} // next i

	if (f_v) {
		cout << "The element " << group_elt << " is represented "
				"by the matrix:" << endl;
		D->print_matrix(Mtx, rk, rk);
		cout << endl;
		}

	

	FREE_int(elt3);
	FREE_int(elt4);
	FREE_int(elt5);

	if (f_v) {
		cout << "young::create_representation done" << endl;
		}
}

void young::young_symmetrizer(int *row_parts, int nb_row_parts, 
	int *tableau, 
	int *elt1, int *elt2, int *elt3, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, a, b, h;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "young::young_symmetrizer" << endl;
		}

	young::row_parts = row_parts;
	l1 = nb_row_parts;
	l2 = row_parts[0];
	Tableau = NEW_int(l1 * l2);
	col_parts = NEW_int(l2);

	a = row_parts[l1 - 1];
	for (j = 0; j < a; j++) {
		col_parts[j] = l1;
		}
	for (i = l1 - 2; i >= 0; i--) {
		b = row_parts[i];
		//cout << "i=" << i << " a=" << a << " b=" << b << endl;
		if (b == a) {
			continue;
			}
		for ( ; j < b; j++) {
			col_parts[j] = i + 1;
			}
		a = b;
		}

	if (f_v) {
		cout << "row_part: ";
		int_vec_print(cout, row_parts, l1);
		cout << endl;
		cout << "col_part: ";
		int_vec_print(cout, col_parts, l2);
		cout << endl;
		}

	h = 0;
	for (i = 0; i < l1; i++) {
		a = row_parts[i];
		for (j = 0; j < a; j++) {
			Tableau[i * l2 + j] = tableau[h];
			h++;
			}
		}
	
	if (f_v) {
		cout << "We are using the following tableau:" << endl;
		Combi.print_tableau(Tableau, l1, l2, row_parts, col_parts);
		}


	Row_partition = NEW_OBJECT(set_of_sets);
	Col_partition = NEW_OBJECT(set_of_sets);

	Row_partition->init_basic(n, l1, row_parts, 0 /* verbose_level*/);
	for (i = 0; i < l1; i++) {
		a = row_parts[i];
		for (j = 0; j < a; j++) {
			b = Tableau[i * l2 + j];
			Row_partition->Sets[i][j] = b;
			}
		cout << endl;
		}
	Col_partition->init_basic(n, l2, col_parts, 0 /* verbose_level*/);
	for (i = 0; i < l2; i++) {
		a = col_parts[i];
		for (j = 0; j < a; j++) {
			b = Tableau[j * l2 + i];
			Col_partition->Sets[i][j] = b;
			}
		cout << endl;
		}

	if (f_v) {
		cout << "Row partition:" << endl;
		Row_partition->print();
		cout << "Col partition:" << endl;
		Col_partition->print();
		}


	int go1, go2;
		
	compute_generators(go1, go2, verbose_level);


	S1 = A->create_sims_from_generators_with_target_group_order_int(
		gens1, go1, 0 /* verbose_level */);
	if (f_v) {
		cout << "Row stabilizer created" << endl;
		}


	if (f_vv) {
		for (i = 0; i < go1; i++) {
			S1->element_unrank_lint(i, Elt);
			cout << "element " << i << " is ";
			A->element_print_quick(Elt, cout);
			cout << endl;
			}
		}



	S2 = A->create_sims_from_generators_with_target_group_order_int(
		gens2, go2, 0 /* verbose_level */);
	if (f_v) {
		cout << "Column stabilizer created" << endl;
		}

	if (f_v) {
		for (i = 0; i < go2; i++) {
			S2->element_unrank_lint(i, Elt);
			cout << "element " << i << " is ";
			A->element_print_quick(Elt, cout);
			cout << endl;
			}
		}


	int_vec_zero(elt1, goi);
	int_vec_zero(elt2, goi);


	// Unrank the element in the row-stabilizer subgroup,
	// then rank the element in the symmetric group.
	// Put the coefficient as one.
	for (i = 0; i < go1; i++) {
		S1->element_unrank_lint(i, Elt);
		j = S->element_rank_lint(Elt);
		elt1[j] += 1;
		}

	int s;
	
	// Unrank the element in the column-stabilizer subgroup,
	// then rank the element in the symmetric group.
	// Put the coefficient as the sign of the permutation.
	for (i = 0; i < go2; i++) {
		S2->element_unrank_lint(i, Elt);
		j = S->element_rank_lint(Elt);

		s = Combi.perm_signum(Elt, n);
		
		elt2[j] += s;
		}


	if (f_v) {
		cout << "elt1=" << endl;
		group_ring_element_print(A, S, elt1);
		cout << endl;

		cout << "elt2=" << endl;
		group_ring_element_print(A, S, elt2);
		cout << endl;
		}

	// multiply the two group ring elements:

	group_ring_element_mult(A, S, elt1, elt2, elt3);

	if (f_v) {
		cout << "elt3=" << endl;
		group_ring_element_print(A, S, elt3);
		cout << endl;
		}



	if (f_v) {
		cout << "young::young_symmetrizer done" << endl;
		}
}

void young::compute_generators(int &go1, int &go2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, h;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "young::compute_generators" << endl;
		}

	int nb_gens1, nb_gens2;

	nb_gens1 = 0;	
	for (i = 0; i < l1; i++) {
		a = row_parts[i];
		if (a > 1) {
			nb_gens1 += a - 1;
			}
		}
	if (f_v) {
		cout << "nb_gens1 = " << nb_gens1 << endl;
		}

	nb_gens2 = 0;	
	for (i = 0; i < l2; i++) {
		a = col_parts[i];
		if (a > 1) {
			nb_gens2 += a - 1;
			}
		}
	if (f_v) {
		cout << "nb_gens2 = " << nb_gens2 << endl;
		}

	gens1 = NEW_OBJECT(vector_ge);
	gens2 = NEW_OBJECT(vector_ge);

	int u, s, t;

	gens1->init(A, verbose_level - 2);
	gens1->allocate(nb_gens1, verbose_level - 2);
	h = 0;
	go1 = 1;
	for (i = 0; i < l1; i++) {
		a = row_parts[i];
		if (a > 1) {
			go1 *= Combi.int_factorial(a);
			for (j = 1; j < a; j++, h++) {
				for (u = 0; u < n; u++) {
					v[u] = u;
					}
				s = Row_partition->Sets[i][0];
				t = Row_partition->Sets[i][j];
				v[s] = t;
				v[t] = s;
				A->make_element(Elt, v, 0 /* verbose_level */);
				A->element_move(Elt, gens1->ith(h), 0);
				}
			}
		}
	if (f_v) {
		cout << "go1=" << go1 << endl;
		cout << "Generators for row stabilizer:" << endl;
		gens1->print(cout);
		}

	gens2->init(A, verbose_level - 2);
	gens2->allocate(nb_gens2, verbose_level - 2);
	h = 0;
	go2 = 1;
	for (i = 0; i < l2; i++) {
		a = col_parts[i];
		if (a > 1) {
			go2 *= Combi.int_factorial(a);
			for (j = 1; j < a; j++, h++) {
				for (u = 0; u < n; u++) {
					v[u] = u;
					}
				s = Col_partition->Sets[i][0];
				t = Col_partition->Sets[i][j];
				v[s] = t;
				v[t] = s;
				A->make_element(Elt, v, 0 /* verbose_level */);
				A->element_move(Elt, gens2->ith(h), 0);
				}
			}
		}
	if (f_v) {
		cout << "go2=" << go2 << endl;
		cout << "Generators for col stabilizer:" << endl;
		gens2->print(cout);
		}
	
	if (f_v) {
		cout << "young::compute_generators done" << endl;
		}

}

void young::Maschke(int *Rep, 
	int dim_of_module, int dim_of_submodule, 
	int *&Mu, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n, k, r;
	int i, j, h, hv;
	int sz;
	int *A, *Av;
	int *g, *gv, *Tau, *Theta, *TauTheta;

	if (f_v) {
		cout << "young::Maschke" << endl;
		}

	n = dim_of_module;
	k = dim_of_submodule;
	r = n - k;
	
	sz = dim_of_module * dim_of_module * D->size_of_instance_in_int;
	if (f_v) {
		cout << "young::Maschke n=" << n << endl;
		cout << "young::Maschke k=" << k << endl;
		cout << "young::Maschke r=" << r << endl;
		}


	if (f_v) {
		cout << "young::Maschke checking if submodule "
				"is invariant" << endl;
		}
	for (h = 0; h < goi; h++) {
		A = Rep + h * sz;
		for (i = 0; i < dim_of_submodule; i++) {
			for (j = dim_of_submodule; j < dim_of_module; j++) {
				if (!D->is_zero(D->offset(A, i * dim_of_module + j), 0)) {
					cout << "The submodule is not invariant "
							"under the action" << endl;
					exit(1);
					}
				}
			}
		}
	if (f_v) {
		cout << "young::Maschke submodule is invariant, OK" << endl;
		}

	g = NEW_int(D->size_of_instance_in_int);
	gv = NEW_int(D->size_of_instance_in_int);
	Tau = NEW_int(r * r * D->size_of_instance_in_int);
	Theta = NEW_int(r * k * D->size_of_instance_in_int);
	TauTheta = NEW_int(r * k * D->size_of_instance_in_int);
	Mu = NEW_int(r * k * D->size_of_instance_in_int);

	D->make_integer(g, goi, 0);
	D->inverse(g, gv, 0);
	D->negate(gv, 0);

	if (f_v) {
		cout << "-1/g=";
		D->print(gv);
		cout << endl;
		}

	D->make_zero_vector(Mu, r * k, 0);

	if (f_vv) {
		cout << "Mu (beginning) = " << endl;
		D->print_matrix(Mu, r, k);
		cout << endl;
		}


	for (h = 0; h < goi; h++) {

		hv = S->invert_by_rank(h, 0 /* verbose_level */);

		if (f_v) {
			cout << "h=" << h << " / " << goi << " hv=" << hv << endl;
			}
		A = Rep + h * sz;
		Av = Rep + hv * sz;

		// get Tau(hv):
		for (i = 0; i < r; i++) {
			for (j = 0; j < r; j++) {
				D->copy(D->offset(Av, (k + i) * n + k + j),
						D->offset(Tau, i * r + j), 0);
				}
			}

		if (f_vv) {
			cout << "Tau(hv) = " << endl;
			D->print_matrix(Tau, r, r);
			cout << endl;
			}

		// get Theta(h):
		for (i = 0; i < r; i++) {
			for (j = 0; j < k; j++) {
				D->copy(D->offset(A, (k + i) * n + j),
						D->offset(Theta, i * k + j), 0);
				}
			}
		if (f_vv) {
			cout << "Theta(h) = " << endl;
			D->print_matrix(Theta, r, k);
			cout << endl;
			}

		// multiply Tau and Theta:

		D->mult_matrix(Tau, Theta, TauTheta, r, r, k,
				0 /* verbose_level */);

		if (f_vv) {
			cout << "TauTheta = " << endl;
			D->print_matrix(TauTheta, r, k);
			cout << endl;
			}
		
		// add to Mu:
		
		D->add_apply_matrix(Mu, TauTheta, r, k,
				0 /* verbose_level */);

		if (f_vv) {
			cout << "Mu (partial sum) = " << endl;
			D->print_matrix(Mu, r, k);
			cout << endl;
			}

		}

	D->matrix_mult_apply_scalar(Mu, gv, r, k,
			0 /* verbose_level */);

	if (f_v) {
		cout << "Mu = " << endl;
		D->print_matrix(Mu, r, k);
		cout << endl;
		}


	FREE_int(g);
	FREE_int(gv);
	FREE_int(Tau);
	FREE_int(Theta);
	FREE_int(TauTheta);
	//FREE_int(Mu);
	

	if (f_v) {
		cout << "young::Maschke done" << endl;
		}
}

long int young::group_ring_element_size(action *A, sims *S)
{
	long int goi;

	goi = S->group_order_lint();
	return goi;
}

void young::group_ring_element_create(action *A, sims *S, int *&elt)
{
	long int goi;

	goi = S->group_order_lint();
	elt = NEW_int(goi);
	group_ring_element_zero(A, S, elt);
}

void young::group_ring_element_free(action *A, sims *S, int *elt)
{
	FREE_int(elt);
}

void young::group_ring_element_print(action *A, sims *S, int *elt)
{
	long int goi;

	goi = S->group_order_lint();
	int_vec_print(cout, elt, goi);
}

void young::group_ring_element_copy(action *A, sims *S,
		int *elt_from, int *elt_to)
{
	long int goi;

	goi = S->group_order_lint();
	int_vec_copy(elt_from, elt_to, goi);
}

void young::group_ring_element_zero(action *A, sims *S, int *elt)
{
	long int goi;

	goi = S->group_order_lint();
	int_vec_zero(elt, goi);
}

void young::group_ring_element_mult(action *A,
		sims *S, int *elt1, int *elt2, int *elt3)
{
	long int goi;
	int i, j, k;
	int a, b, c;

	goi = S->group_order_lint();
	int_vec_zero(elt3, goi);
	for (i = 0; i < goi; i++) {
		a = elt1[i];
		for (j = 0; j < goi; j++) {
			b = elt2[j];
			c = a * b;
			k = S->mult_by_rank(i, j, 0 /* verbose_level */);
			elt3[k] += c;
			}
		}
}




}}


