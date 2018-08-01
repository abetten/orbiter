// sarnak.C
//
// Anton Betten
// November 2, 2011
//
// creates the graphs described in the paper:
//
// @article {MR963118,
//    AUTHOR = {Lubotzky, A. and Phillips, R. and Sarnak, P.},
//     TITLE = {Ramanujan graphs},
//   JOURNAL = {Combinatorica},
//  FJOURNAL = {Combinatorica. An International Journal of the J\'anos Bolyai
//              Mathematical Society},
//    VOLUME = {8},
//      YEAR = {1988},
//    NUMBER = {3},
//     PAGES = {261--277},
//      ISSN = {0209-9683},
//     CODEN = {COMBDI},
//   MRCLASS = {05C75 (05C25 05C50)},
//  MRNUMBER = {963118 (89m:05099)},
//MRREVIEWER = {Dave Witte Morris},
//       DOI = {10.1007/BF02126799},
//       URL = {http://0-dx.doi.org.catalog.library.colostate.edu/10.1007/BF02126799},
//}

#include "orbiter.h"


INT t0;

void do_it(INT p, INT q, INT verbose_level);



int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_p = FALSE;
	INT p = 0;
	INT f_q = FALSE;
	INT q = 0;

	t0 = os_ticks();

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-p") == 0) {
			f_p = TRUE;
			p = atoi(argv[++i]);
			cout << "-p " << p << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	


	if (!f_p) {
		cout << "please specify -p <p>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please specify -q <q>" << endl;
		exit(1);
		}

	do_it(p, q, verbose_level);
	
}

void do_it(INT p, INT q, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, h, l, f_special = FALSE;



	l = Legendre(p, q, 0);
	cout << "Legendre(" << p << ", " << q << ")=" << l << endl;


	finite_field *F;
	action *A;
	INT f_semilinear = FALSE;
	INT f_basis = TRUE;

	F = new finite_field;
	F->init(q, 0);
	//F->init_override_polynomial(q, override_poly, verbose_level);

	A = new action;
	
	if (l == 1) {
		f_special = TRUE;

		cout << "Creating projective special linear group:" << endl; 
		A->init_projective_special_group(2, F, 
			f_semilinear, 
			f_basis, 
			verbose_level - 2);		
		}
	else {
		cout << "Creating projective linear group:" << endl; 
		A->init_projective_group(2, F, 
			f_semilinear, 
			f_basis, 
			verbose_level - 2);		
		}



	sims *Sims;

	Sims = A->Sims;	

	
	longinteger_object go;
	INT goi;
	Sims->group_order(go);

	cout << "found a group of order " << go << endl;
	goi = go.as_INT();
	cout << "found a group of order " << goi << endl;

	


	INT a0, a1, a2, a3;
	INT sqrt_p;

	INT *sqrt_mod_q;
	INT I;
	INT *A4;
	INT nb_A4 = 0;

	A4 = NEW_INT((p + 1) * 4);
	sqrt_mod_q = NEW_INT(q);
	for (i = 0; i < q; i++) {
		sqrt_mod_q[i] = -1;
		}
	for (i = 0; i < q; i++) {
		j = F->mult(i, i);
		sqrt_mod_q[j] = i;
		}
	cout << "sqrt_mod_q:" << endl;
	INT_vec_print(cout, sqrt_mod_q, q);
	cout << endl;

	sqrt_p = 0;
	for (i = 1; i < p; i++) {
		if (i * i > p) {
			sqrt_p = i - 1;
			break;
			}
		}
	cout << "p=" << p << endl;
	cout << "sqrt_p = " << sqrt_p << endl;

	
	for (I = 0; I < q; I++) {
		if (F->add(F->mult(I, I), 1) == 0) {
			break;
			}
		}
	if (I == q) {
		cout << "did not find I" << endl;
		exit(1);
		}
	cout << "I=" << I << endl;
	
	for (a0 = 1; a0 <= sqrt_p; a0++) {
		if (EVEN(a0)) {
			continue;
			}
		for (a1 = -sqrt_p; a1 <= sqrt_p; a1++) {
			if (ODD(a1)) {
				continue;
				}
			for (a2 = -sqrt_p; a2 <= sqrt_p; a2++) {
				if (ODD(a2)) {
					continue;
					}
				for (a3 = -sqrt_p; a3 <= sqrt_p; a3++) {
					if (ODD(a3)) {
						continue;
						}
					if (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 == p) {
						cout << "solution " << nb_A4 << " : " << a0 << ", " << a1 << ", " << a2 << ", " << a3 << ", " << endl;
						if (nb_A4 == p + 1) { 
							cout << "too many solutions" << endl;
							exit(1);
							}
						A4[nb_A4 * 4 + 0] = a0;
						A4[nb_A4 * 4 + 1] = a1;
						A4[nb_A4 * 4 + 2] = a2;
						A4[nb_A4 * 4 + 3] = a3;
						nb_A4++;
						}
					}
				}
			}
		} 

	cout << "nb_A4=" << nb_A4 << endl;
	if (nb_A4 != p + 1) { 
		cout << "nb_A4 != p + 1" << endl;
		exit(1);
		}

	INT_matrix_print(A4, nb_A4, 4);

	vector_ge *gens;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT M4[4];
	INT det; //, s, sv;
	
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);

	gens = new vector_ge;
	gens->init(A);
	gens->allocate(nb_A4);
	
	cout << "making connection set:" << endl;
	for (i = 0; i < nb_A4; i++) {

		if (f_vv) {
			cout << "making generator " << i << ":" << endl;
			}
		a0 = A4[i * 4 + 0];
		a1 = A4[i * 4 + 1];
		a2 = A4[i * 4 + 2];
		a3 = A4[i * 4 + 3];
		while (a0 < 0) {
			a0 += q;
			}
		while (a1 < 0) {
			a1 += q;
			}
		while (a2 < 0) {
			a2 += q;
			}
		while (a3 < 0) {
			a3 += q;
			}
		a0 = a0 % q;
		a1 = a1 % q;
		a2 = a2 % q;
		a3 = a3 % q;
		if (f_vv) {
			cout << "making generator " << i << ": a0=" << a0 << " a1=" << a1 << " a2=" << a2 << " a3=" << a3 << endl;
			}
		M4[0] = F->add(a0, F->mult(I, a1));
		M4[1] = F->add(a2, F->mult(I, a3));
		M4[2] = F->add(F->negate(a2), F->mult(I, a3));
		M4[3] = F->add(a0, F->negate(F->mult(I, a1)));

		if (f_vv) {
			cout << "M4=";
			INT_vec_print(cout, M4, 4);
			cout << endl;
			}

		if (f_special) {
			det = F->add(F->mult(M4[0], M4[3]), F->negate(F->mult(M4[1], M4[2])));

			if (f_vv) {
				cout << "det=" << det << endl;
				}
		
#if 0
			s = sqrt_mod_q[det];
			if (s == -1) {
				cout << "determinant is not a square" << endl;
				exit(1);
				}
			sv = F->inverse(s);
			if (f_vv) {
				cout << "det=" << det << " sqrt=" << s << " mutiplying by " << sv << endl;
				}
			for (j = 0; j < 4; j++) {
				M4[j] = F->mult(sv, M4[j]);
				}
			if (f_vv) {
				cout << "M4=";
				INT_vec_print(cout, M4, 4);
				cout << endl;
				}
#endif
			}

		A->make_element(Elt1, M4, verbose_level - 1);

		if (f_v) {
			cout << "s_" << i << "=" << endl;
			A->element_print_quick(Elt1, cout);
			}
	
		A->element_move(Elt1, gens->ith(i), 0);
		}
	

	INT *Adj;

	Adj = NEW_INT(goi * goi);

	INT_vec_zero(Adj, goi * goi);

	cout << "Computing the Cayley graph:" << endl;
	for (i = 0; i < goi; i++) {
		Sims->element_unrank_INT(i, Elt1);
		//cout << "i=" << i << endl;
		for (h = 0; h < nb_A4; h++) {
			A->element_mult(Elt1, gens->ith(h), Elt2, 0);
#if 0
			cout << "i=" << i << " h=" << h << endl;
			cout << "Elt1=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "g_h=" << endl;
			A->element_print_quick(gens->ith(h), cout);
			cout << "Elt2=" << endl;
			A->element_print_quick(Elt2, cout);
#endif
			j = Sims->element_rank_INT(Elt2);
			Adj[i * goi + j] = Adj[j * goi + i] = 1;
			if (i == 0) {
				cout << "edge " << i << " " << j << endl;
				}
			}
		}

	cout << "The adjacency matrix of a graph with " << goi << " vertices has been computed" << endl;
	//INT_matrix_print(Adj, goi, goi);

	INT k;
	k = 0;
	for (i = 0; i < goi; i++) {
		if (Adj[0 * goi + i]) {
			k++;
			}
		}
	cout << "k=" << k << endl;



	colored_graph *CG;
	BYTE fname[1000];

	CG = new colored_graph;
	CG->init_adjacency_no_colors(goi, Adj, verbose_level);

	sprintf(fname, "Sarnak_%ld_%ld.colored_graph", p, q);

	CG->save(fname, verbose_level);


	delete CG;
	delete gens;
	delete A;
	FREE_INT(A4);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	delete F;

}


