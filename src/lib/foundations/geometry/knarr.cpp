// knarr.cpp
// 
// Anton Betten
//
// started:    March 4, 2011
// moved here: March 30, 2011
// 
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {


	// The Knarr construction of a GQ(q^2,q) from a BLT set of lines in W(3,q):
	//
	// let P = (1,0,0,0,0,0) in W(5,q)
	//
	// Let B be a BLT-set of lines in W(3,q), 
	// lifted into P^\perp in W(5,q)
	//
	// type i) points:
	// the q^5 points in W(5,q) \setminus P^\perp
	// type ii) points: 
	// lines in the BLT-planes, not containing the point P
	// there are (q+1)*q^2 of them (q^2 for each BLT-plane)
	// type iii) points:
	// The unique point P=(1,0,0,0,0,0)

	// For a total of q^5 + q^3 + q^2 + 1 = (q^2 + 1)(q^3 + 1) points
	
	// type a) lines:
	// t.i. planes \pi, not containing P, 
	// with \pi \cap P^\perp a line of a BLT-plane (such that the line avoids P),
	// i.e. a point of type ii).
	// There are (q+1)*q^3 such planes
	// 
	// type b) lines:
	// the q+1 elements of the BLT set, 
	// lifted to become t.i. planes contaning P in W(5,q)

	// For a total of 
	// q^4 + q^3 + q + 1 = (q + 1)*(q^3 + 1) lines

	// This is the required number for a GQ(q^2,q).
	// Recall that a GQ(s,t) has 
	// (s+1)(st+1) points and 
	// (t+1)(st+1) lines.

knarr::knarr()
{
	q = 0;
	BLT_no = 0;
	W = NULL;
	P5 = NULL;
	G63 = NULL;
	F = NULL;

	BLT = NULL;
	BLT_line_idx = NULL;
	Basis = NULL;
	Basis2 = NULL;
	subspace_basis = NULL;
	Basis_Pperp = NULL;

	six_choose_three_q = NULL;
	six_choose_three_q_int = 0;

	f_show = FALSE;
	dim_intersection = 0;

	Basis_intersection = NULL;
	type_i_points = NULL;
	type_ii_points = NULL;
	type_iii_points = NULL;
	type_a_lines = NULL;
	type_b_lines = NULL;
	type_a_line_BLT_idx = NULL;
	q2 = 0;
	q5 = 0;
	//int v5[5];
	//int v6[6];

}




knarr::~knarr()
{
	freeself();
}

void knarr::freeself()
{
	if (BLT_line_idx) {
		FREE_int(BLT_line_idx);
		}
	if (Basis) {
		FREE_int(Basis);
		}
	if (Basis2) {
		FREE_int(Basis2);
		}
	if (subspace_basis) {
		FREE_int(subspace_basis);
		}
	if (Basis_Pperp) {
		FREE_int(Basis_Pperp);
		}
	if (six_choose_three_q) {
		FREE_OBJECT(six_choose_three_q);
	}
	if (Basis_intersection) {
		FREE_int(Basis_intersection);
		}

	//cout << "freeing things 1" << endl;
	
	if (type_i_points) {
		FREE_OBJECT(type_i_points);
		FREE_OBJECT(type_ii_points);
		FREE_OBJECT(type_iii_points);
		FREE_OBJECT(type_a_lines);
		FREE_OBJECT(type_b_lines);
		}
	if (type_a_line_BLT_idx) {
		FREE_int(type_a_line_BLT_idx);
		}

	//cout << "freeing things 2" << endl;
	
	if (W) {
		FREE_OBJECT(W);
		FREE_OBJECT(P5);
		FREE_OBJECT(G63);
		}
}

void knarr::init(field_theory::finite_field *F, int BLT_no, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;
	knowledge_base K;
	combinatorics::combinatorics_domain C;
	
	knarr::F = F;
	knarr::q = F->q;
	knarr::BLT_no = BLT_no;


	if (f_v) {
		cout << "knarr::init q=" << q << " BLT_no=" << BLT_no << endl;
#if 0
		if (f_poly) {
			cout << "poly = " << poly << endl;
			}
#endif
		}

	W = NEW_OBJECT(W3q);
	P5 = NEW_OBJECT(projective_space);

	W->init(F, verbose_level - 1);
	P5->init(5, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	if (f_v) {
		cout << "P5->nb_points=" << P5->N_points << endl;
		cout << "P5->nb_lines=" << P5->N_lines << endl;
		}

	G63 = NEW_OBJECT(grassmann);
	G63->init(6, 3, F, verbose_level - 2);


	six_choose_three_q = NEW_OBJECT(ring_theory::longinteger_object);


	C.q_binomial(*six_choose_three_q, 6, 3, q, 0);
	six_choose_three_q_int = six_choose_three_q->as_int();
	if (f_v) {
		cout << "Number of planes in P5 = " << six_choose_three_q_int << endl;
		}

	BLT = K.BLT_representative(q, BLT_no);
	BLT_line_idx = NEW_int(q + 1);
	Basis = NEW_int(4 * 6);
	Basis2 = NEW_int(4 * 6);
	subspace_basis = NEW_int(3 * 6);
	Basis_Pperp = NEW_int(5 * 6);
	Basis_intersection = NEW_int(6 * 6);
	
	for (i = 0; i < q + 1; i++) {
		BLT_line_idx[i] = W->Line_idx[BLT[i]];
		}


	if (f_vv) {
		cout << "the BLT-set of lines is:" << endl;
		for (i = 0; i < q + 1; i++) {
			a = BLT_line_idx[i];
			cout << setw(4) << i << " : " << setw(4) << a << " : "
					<< setw(5) << W->Lines[a] << ":" << endl;
			W->P3->unrank_line(Basis, W->Lines[a]);
			Orbiter->Int_vec->print_integer_matrix_width(cout, Basis, 2, 4, 4, F->log10_of_q);
			cout << endl;
			}
		}

}

void knarr::points_and_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);

	int i, j, a, c, jj, h, hh, rk, u;
	number_theory::number_theory_domain NT;
	geometry_global Gg;


	if (f_v) {
		cout << "knarr::points_and_lines" << endl;
		}
	
	type_i_points = NEW_OBJECT(data_structures::fancy_set);
	type_ii_points = NEW_OBJECT(data_structures::fancy_set);
	type_iii_points = NEW_OBJECT(data_structures::fancy_set);
	type_a_lines = NEW_OBJECT(data_structures::fancy_set);
	type_b_lines = NEW_OBJECT(data_structures::fancy_set);



	// figure out type i) points:

	q2 = q * q;
	q5 = NT.i_power_j(q, 5);
	type_i_points->init(P5->N_points, 0);
	for (i = 0; i < q5; i++) {
		Gg.AG_element_unrank(q, v5, 1, 5, i);
		v6[0] = v5[0];
		v6[1] = 1;
		v6[2] = v5[1];
		v6[3] = v5[2];
		v6[4] = v5[3];
		v6[5] = v5[4];
		j = P5->rank_point(v6);
		type_i_points->add_element(j);
		}

	if (f_vv) {
		cout << "We found " << type_i_points->k
				<< " type i points" << endl;
		}
	if (f_vvv) {
		cout << "The " << type_i_points->k
				<< " type i points are:" << endl;
		type_i_points->println();
		}
	
	type_ii_points->init(P5->N_lines, 0);

	type_a_lines->init(six_choose_three_q_int, 0);
	type_b_lines->init(six_choose_three_q_int, 0);

	type_a_line_BLT_idx = NEW_int(six_choose_three_q_int);
	
	// figure out type b) lines and type ii) points:

	for (h = 0; h < q + 1; h++) {

		
		a = BLT_line_idx[h];
		W->P3->unrank_line(Basis, W->Lines[a]);
		if (f_vvv) {
			cout << "BLT line " << h << " which is "
					<< W->Lines[a] << endl;
			Orbiter->Int_vec->print_integer_matrix_width(cout, Basis, 2, 4, 4,
					F->log10_of_q);
			cout << endl;
			}

		Orbiter->Int_vec->zero(Basis2, 3 * 6);
		Basis2[0] = 1;
		for (i = 0; i < 2; i++) {
			for (j = 0; j < 4; j++) {
				Basis2[(i + 1) * 6 + 2 + j] = Basis[i * 4 + j];
				}
			}
		if (f_v4) {
			cout << "embedded:" << endl;
			Orbiter->Int_vec->print_integer_matrix_width(cout, Basis2, 3, 6, 6,
					F->log10_of_q);
			cout << endl;
			}


		Orbiter->Int_vec->copy(Basis2, G63->M, 3 * 6);

		i = G63->rank_lint(0);
		if (f_v4) {
			cout << "This plane has rank " << i
					<< " and will be added as type b line" << endl;
			}
		type_b_lines->add_element(i);


		grassmann *G32;
		grassmann_embedded *Gre;

		
		G32 = NEW_OBJECT(grassmann);
		Gre = NEW_OBJECT(grassmann_embedded);
		G32->init(3, 2, F, verbose_level - 3);
		Gre->init(6, 3, G32, Basis2, verbose_level - 3);

		for (jj = 0; jj < Gre->degree; jj++) {
			Gre->unrank_lint(subspace_basis, jj, 0);
			Orbiter->Int_vec->copy(subspace_basis, P5->Grass_lines->M, 2 * 6);
			j = P5->Grass_lines->rank_lint(0);
			if (f_v4) {
				cout << "Subspace " << jj << " has a basis:" << endl;
				Orbiter->Int_vec->print_integer_matrix_width(cout, subspace_basis,
						2, 6, 6, F->log10_of_q);
				cout << "and has rank " << j << endl;
				}
			Orbiter->Int_vec->zero(subspace_basis + 2 * 6, 6);
			subspace_basis[2 * 6 + 0] = 1;
			rk = F->Linear_algebra->Gauss_easy(subspace_basis, 3, 6);
			if (rk <= 2) {
				if (f_v4) {
					cout << "This subspace contains P, "
							"so it is not interesting" << endl;
					}
				continue;
				}
			if (f_v4) {
				cout << "This subspace does not contain P, "
						"so it is interesting. Adding line " << j << endl;
				}
			type_ii_points->add_element(j);
			}

		FREE_OBJECT(G32);
		FREE_OBJECT(Gre);

		} // next h

	if (f_vv) {
		cout << "We found " << type_ii_points->k
				<< " type ii points." << endl;
		cout << "We found " << type_b_lines->k
				<< " type b lines." << endl;
		}
	if (f_vvv) {
		cout << "The " << type_ii_points->k
				<< " type ii points are:" << endl;
		type_ii_points->println();

		cout << "The " << type_b_lines->k
				<< " type b lines are:" << endl;
		type_b_lines->println();
		}
	
	



	// figure out type a) lines:



	for (i = 0; i < six_choose_three_q_int; i++) {
		
		G63->unrank_lint_here(Basis, i, 0);
		if (f_v4) {
			cout << "Subspace i=" << i << " / "
					<< six_choose_three_q_int << endl;
			Orbiter->Int_vec->print_integer_matrix_width(cout, Basis, 3, 6, 6, F->log10_of_q);
			cout << endl;
			}
		if (!F->Linear_algebra->is_totally_isotropic_wrt_symplectic_form(3, 6, Basis)) {
			if (f_v4) {
				cout << "is not totally isotropic" << endl;
				}
			continue;
			}

		// check if P is not contained:
		P5->unrank_point(Basis2, 0);
		c = F->Linear_algebra->is_subspace(6, 1, Basis2, 3, Basis, 0 /*verbose_level*/);
		if (c) {
			if (f_v4) {
				cout << "contains the point P" << endl;
				}
			continue;
			}

	 

		// check if \pi \cap P^\perp has rank 2:

		for (h = 0; h < 5; h++) {
			for (j = 0; j < 6; j++) {
				hh = h;
				if (h) {
					hh++;
					}
				if (j == hh) {
					Basis_Pperp[h * 6 + j] = 1;
					}
				else {
					Basis_Pperp[h * 6 + j] = 0;
					}
				}
			}
		//cout << "Basis Pperp:" << endl;
		//print_integer_matrix_width(cout, Basis_Pperp,
		//5, 6, 6, F->log10_of_q);
		//cout << endl;


		F->Linear_algebra->intersect_subspaces(6, 5, Basis_Pperp, 3, Basis,
			dim_intersection, Basis_intersection,
			0 /* verbose_level */);

		
		if (dim_intersection != 2) {
			continue;
			}

		// now we figure out if this line belongs to a BLT-plane.
		// Simply add P to the basis and rank to
		// determine which plane it would be.
		// Then check and see if this plane is a BLT-plane.

		Basis_intersection[2 * 6 + 0] = 1;
		Basis_intersection[2 * 6 + 1] = 0;
		Basis_intersection[2 * 6 + 2] = 0;
		Basis_intersection[2 * 6 + 3] = 0;
		Basis_intersection[2 * 6 + 4] = 0;
		Basis_intersection[2 * 6 + 5] = 0;


		Orbiter->Int_vec->copy(Basis_intersection, G63->M, 3 * 6);
		j = G63->rank_lint(0);
		
		if (type_b_lines->is_contained(j)) {


			if (f_v4) {
				cout << "Subspace i=" << i << " is totally "
						"isotropic and does not contain P." << endl;
				Orbiter->Int_vec->print_integer_matrix_width(cout,
						Basis, 3, 6, 6, F->log10_of_q);
				cout << endl;


				cout << "dim_intersection=" << dim_intersection << endl;
				cout << "Intersection:" << endl;
				Orbiter->Int_vec->print_integer_matrix_width(cout,
						Basis_intersection, dim_intersection,
						6, 6, F->log10_of_q);
				cout << endl;
			
				cout << "Added P" << endl;

				cout << "looking at" << endl;
				Orbiter->Int_vec->print_integer_matrix_width(cout,
						Basis_intersection, 3, 6, 6,
						F->log10_of_q);
				cout << endl;
			
				cout << "This plane has rank " << j << endl;


				cout << "This plane is contained in the BLT-set" << endl;
				cout << "The plane of rank " << i
						<< " and will be added as type a line" << endl;
				}			

			type_a_line_BLT_idx[type_a_lines->k] = type_b_lines->set_inv[j];
			type_a_lines->add_element(i);
			} // if 
		else {
			if (f_v4) {
				cout << "not type_b_lines->is_contained(j), "
						"we ignore this subspace" << endl;
				}
			}




		} // next i


	if (f_vv) {
		cout << "We found " << type_a_lines->k
				<< " type a) lines." << endl;
		}
	if (f_vvv) {
		cout << "The " << type_a_lines->k
				<< " type a) lines are:" << endl;
		type_a_lines->println();
		}

	
	if (f_vvv) {
		int cnt;
		cout << "type a) lines by BLT_idx:" << endl;
		for (h = 0; h < q + 1; h++) {
			cnt = 0;
			cout << "BLT_idx = " << h << " :" << endl;
			for (u = 0; u < type_a_lines->k; u++) {
				if (type_a_line_BLT_idx[u] != h) {
					continue;
					}
				i = type_a_lines->set[u];
				G63->unrank_lint_here(Basis, i, 0);
				cout << "cnt = " << cnt << " Subspace i=" << i
						<< " is totally isotropic and does "
								"not contain P." << endl;
				Orbiter->Int_vec->print_integer_matrix_width(cout, Basis,
						3, 6, 6, F->log10_of_q);
				cout << endl;
				cnt++;
				}
			}
		}

}

void knarr::incidence_matrix(int *&Inc,
		int &nb_points, int &nb_lines, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = FALSE;

	int i, j, a, b, c;

	int I, J, row, col, row0, col0, L1, L2;
	int *Basis_U, dim_U;
	int *Basis_V, dim_V;

	Basis_U = NEW_int(3 * 6);
	Basis_V = NEW_int(3 * 6);
	
	nb_points = type_i_points->k + type_ii_points->k + 1;
	nb_lines = type_a_lines->k + type_b_lines->k;

	cout << "nb_points=" << nb_points << endl;
	cout << "nb_lines=" << nb_lines << endl;
	
	cout << "Computing the incidence matrix..." << endl;
	
	Inc = NEW_int(nb_points * nb_lines);
	Orbiter->Int_vec->zero(Inc, nb_points * nb_lines);
	
	for (I = 0; I < 3; I++) {
		if (I == 0) {
			L1 = type_i_points->k;
			row0 = 0;
			}
		else if (I == 1) {
			L1 = type_ii_points->k;
			row0 = type_i_points->k;
			}
		else {
			L1 = 1;
			row0 = type_i_points->k + type_ii_points->k;
			}
		for (i = 0; i < L1; i++) {
			row = row0 + i;

			if (I == 0) {
				a = type_i_points->set[i];
				P5->unrank_point(Basis_U, a);
				dim_U = 1;
				}
			else if (I == 1) {
				a = type_ii_points->set[i];
				P5->Grass_lines->unrank_lint_here(Basis_U, a, 0);
				dim_U = 2;
				}
			else {
				a = 0;
				P5->unrank_point(Basis_U, a);
				dim_U = 1;
				}
			

			if (f_show) {
				cout << "I=" << I << " i=" << i;
				cout << " a=" << a << " row=" << row << endl;
				cout << "Basis_U:" << endl;
				Orbiter->Int_vec->print_integer_matrix_width(cout,
						Basis_U, dim_U, 6, 6, F->log10_of_q);
				cout << endl;
				}
			
			for (J = 0; J < 2; J++) {
				if (J == 0) {
					L2 = type_a_lines->k;
					col0 = 0;
					}
				else {
					L2 = type_b_lines->k;
					col0 = type_a_lines->k;
					}
				for (j = 0; j < L2; j++) {
					col = col0 + j;

					if (J == 0) {
						b = type_a_lines->set[j];
						G63->unrank_lint_here(Basis_V, b, 0);
						dim_V = 3;
						}
					else {
						b = type_b_lines->set[j];
						G63->unrank_lint_here(Basis_V, b, 0);
						dim_V = 3;
						}

					if (f_show) {
						cout << "J=" << J << " j=" << j; 
						cout << " b=" << b << " col=" << col << endl;
						cout << "Basis_V:" << endl;
						Orbiter->Int_vec->print_integer_matrix_width(cout,
								Basis_V, dim_V, 6, 6, F->log10_of_q);
						cout << endl;
						}

					c = F->Linear_algebra->is_subspace(6, dim_U,
							Basis_U, dim_V, Basis_V, 0 /*verbose_level*/);
					if (c) {
						Inc[row * nb_lines + col] = 1;
						}
					}
				}
			}
		}

	cout << "The incidence matrix has been computed" << endl;

#if 0
	cout << "The incidence matrix is" << endl;
	print_integer_matrix_width(cout, Inc,
			nb_points, nb_lines, nb_lines, 1);
	cout << endl;

	char fname[1000];

	snprintf(fname, 1000, "GQ_Knarr_BLT_%d_%d.inc", q, BLT_no);
	write_incidence_matrix_to_file(fname, Inc,
			nb_points, nb_lines, verbose_level);
	cout << "Written file " << fname
			<< " of size " << file_size(fname) << endl;
#endif

	FREE_int(Basis_U);
	FREE_int(Basis_V);
}

}}

