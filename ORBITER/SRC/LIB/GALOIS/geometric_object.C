// geometric_object.C
//
// Anton Betten
// November 18, 2014
//
//
// started from stuff that was in TOP_LEVEL/projective_space.C



#include "galois.h"

void do_cone_over(INT n, finite_field *F, 
	INT *set_in, INT set_size_in, INT *&set_out, INT &set_size_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P1;
	projective_space *P2;
	INT *v;
	INT d = n + 2;
	INT h, u, a, b, cnt;

	if (f_v) {
		cout << "do_cone_over" << endl;
		}
	P1 = new projective_space;
	P2 = new projective_space;
	
	P1->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);
	P2->init(n + 1, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	v = NEW_INT(d);

	set_size_out = 1 + F->q * set_size_in;
	set_out = NEW_INT(set_size_out);
	cnt = 0;
	
	// create the vertex:
	INT_vec_zero(v, d);
	v[d - 1] = 1;
	b = P2->rank_point(v);
	set_out[cnt++] = b;
	

	// for each point, create the generator
	// which is the line connecting the point and the vertex
	// since we have created the vertex already, 
	// we only need to create q points per line:
	
	for (h = 0; h < set_size_in; h++) {
		a = set_in[h];
		for (u = 0; u < F->q; u++) {
			P1->unrank_point(v, a);
			v[d - 1] = u;
			b = P2->rank_point(v);
			set_out[cnt++] = b;
			}
		}

	if (cnt != set_size_out) {
		cout << "do_cone_over cnt != set_size_out" << endl;
		exit(1);
		}

	FREE_INT(v);
	delete P1;
	delete P2;

}

void do_blocking_set_family_3(INT n, finite_field *F, 
	INT *set_in, INT set_size, 
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level)
{
	projective_space *P;
	INT h;
	INT q;

	q = F->q;
	if (n != 2) {
		cout << "do_blocking_set_family_3 we need n = 2" << endl;
		exit(1);
		}
	if (ODD(q)) {
		cout << "do_blocking_set_family_3 we need q even" << endl;
		exit(1);
		}
	if (set_size != q + 2) {
		cout << "do_blocking_set_family_3 we need set_size == q + 2" << endl;
		exit(1);
		}
	P = new projective_space;
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);


	INT *idx;
	INT p_idx[4];
	INT line[6];
	INT diag_pts[3];
	INT diag_line;
	INT nb, pt, sz;
	INT i, j;
	INT basis[6];

	fancy_set *S;

	S = new fancy_set;

	S->init(P->N_lines, 0);
	S->k = 0;

	idx = NEW_INT(set_size);

#if 1
	while (TRUE) {
		cout << "choosing random permutation" << endl;
		random_permutation(idx, set_size);

		cout << idx[0] << ", ";
		cout << idx[1] << ", ";
		cout << idx[2] << ", ";
		cout << idx[3] << endl;

		for (i = 0; i < 4; i++) {
			p_idx[i] = set_in[idx[i]];
			}

		line[0] = P->line_through_two_points(p_idx[0], p_idx[1]);
		line[1] = P->line_through_two_points(p_idx[0], p_idx[2]);
		line[2] = P->line_through_two_points(p_idx[0], p_idx[3]);
		line[3] = P->line_through_two_points(p_idx[1], p_idx[2]);
		line[4] = P->line_through_two_points(p_idx[1], p_idx[3]);
		line[5] = P->line_through_two_points(p_idx[2], p_idx[3]);
		diag_pts[0] = P->line_intersection(line[0], line[5]);
		diag_pts[1] = P->line_intersection(line[1], line[4]);
		diag_pts[2] = P->line_intersection(line[2], line[3]);
	
		diag_line = P->line_through_two_points(diag_pts[0], diag_pts[1]);	
		if (diag_line != P->line_through_two_points(diag_pts[0], diag_pts[2])) {
			cout << "diaginal points not collinear!" << endl;
			exit(1);
			}
		P->unrank_line(basis, diag_line);
		INT_matrix_print(basis, 2, 3);
		nb = 0;
		for (i = 0; i < set_size; i++) {
			pt = set_in[i];
			if (P->is_incident(pt, diag_line)) {
				nb++;
				}
			}
		cout << "nb=" << nb << endl;
		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
			break;
			}
		} // while 
#endif

#if 0
	INT fundamental_quadrangle[4] = {0,1,2,3};
	INT basis[6];
	
	for (i = 0; i < 4; i++) {
		if (!INT_vec_search_linear(set_in, set_size, fundamental_quadrangle[i], j)) {
			cout << "the point " << fundamental_quadrangle[i] << " is not contained in the hyperoval" << endl;
			exit(1);
			}
		idx[i] = j;
		}
	cout << "the fundamental quadrangle is contained, the positions are " << endl;
		cout << idx[0] << ", ";
		cout << idx[1] << ", ";
		cout << idx[2] << ", ";
		cout << idx[3] << endl;

		for (i = 0; i < 4; i++) {
			p_idx[i] = set_in[idx[i]];
			}

		line[0] = P->line_through_two_points(p_idx[0], p_idx[1]);
		line[1] = P->line_through_two_points(p_idx[0], p_idx[2]);
		line[2] = P->line_through_two_points(p_idx[0], p_idx[3]);
		line[3] = P->line_through_two_points(p_idx[1], p_idx[2]);
		line[4] = P->line_through_two_points(p_idx[1], p_idx[3]);
		line[5] = P->line_through_two_points(p_idx[2], p_idx[3]);
		diag_pts[0] = P->line_intersection(line[0], line[5]);
		diag_pts[1] = P->line_intersection(line[1], line[4]);
		diag_pts[2] = P->line_intersection(line[2], line[3]);
	
		diag_line = P->line_through_two_points(diag_pts[0], diag_pts[1]);	
		cout << "The diagonal line is " << diag_line << endl;

		P->unrank_line(basis, diag_line);
		INT_matrix_print(basis, 2, 3);
		
		if (diag_line != P->line_through_two_points(diag_pts[0], diag_pts[2])) {
			cout << "diaginal points not collinear!" << endl;
			exit(1);
			}
		nb = 0;
		for (i = 0; i < set_size; i++) {
			pt = set_in[i];
			if (P->Incidence[pt * P->N_lines + diag_line]) {
				nb++;
				}
			}
		cout << "nb=" << nb << endl;
		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
			}
		else {
			cout << "error: the diagonal line is not external" << endl;
			exit(1);
			}

#endif

	S->add_element(diag_line);
	for (i = 4; i < set_size; i++) {
		pt = set_in[idx[i]];
		for (j = 0; j < P->r; j++) {
			h = P->Lines_on_point[pt * P->r + j];
			if (!S->is_contained(h)) {
				S->add_element(h);
				}
			}
		}

	cout << "we created a blocking set of lines of size " << S->k << ":" << endl;
	INT_vec_print(cout, S->set, S->k);
	cout << endl;
	

	INT *pt_type;
	
	pt_type = NEW_INT(P->N_points);

	P->point_types(S->set, S->k, pt_type, 0);

	classify C;

	C.init(pt_type, P->N_points, FALSE, 0);

	
	cout << "the point types are:" << endl;
	C.print_naked(FALSE /*f_backwards*/);
	cout << endl;

#if 0
	for (i = 0; i <= P->N_points; i++) {
		if (pt_type[i]) {
			cout << i << "^" << pt_type[i] << " ";
			}
		}
	cout << endl;
#endif

	sz = ((q * q) >> 1) + ((3 * q) >> 1) - 4;

	if (S->k != sz) {
		cout << "the size does not match the expected size" << endl;
		exit(1);
		}

	cout << "the size is OK" << endl;

	the_set_out = NEW_INT(sz);
	set_size_out = sz;

	for (i = 0; i < sz; i++) {
		j = S->set[i];
		the_set_out[i] = P->Polarity_hyperplane_to_point[j];
		}
	
	
	
	delete P;
}

void create_hyperoval(finite_field *F, 
	INT f_translation, INT translation_exponent, 
	INT f_Segre, INT f_Payne, INT f_Cherowitzo, INT f_OKeefe_Penttila, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT n = 2;
	INT i, d;
	INT *v;
	INT q = F->q;

	d = n + 1;
	P = new projective_space;

	if (f_v) {
		cout << "create_hyperoval" << endl;
		}
	
	if (f_v) {
		cout << "create_hyperoval before P->init" << endl;
		}
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level /*MINIMUM(verbose_level - 1, 3)*/);
	if (f_v) {
		cout << "create_hyperoval after P->init" << endl;
		}

	v = NEW_INT(d);
	Pts = NEW_INT(P->N_points);

	if (f_translation) {
		P->create_translation_hyperoval(Pts, nb_pts, translation_exponent, verbose_level - 0);
		sprintf(fname, "hyperoval_translation_q%ld.txt", q);
		}
	else if (f_Segre) {
		P->create_Segre_hyperoval(Pts, nb_pts, verbose_level - 2);
		sprintf(fname, "hyperoval_Segre_q%ld.txt", q);
		}
	else if (f_Payne) {
		P->create_Payne_hyperoval(Pts, nb_pts, verbose_level - 2);
		sprintf(fname, "hyperoval_Payne_q%ld.txt", q);
		}
	else if (f_Cherowitzo) {
		P->create_Cherowitzo_hyperoval(Pts, nb_pts, verbose_level - 2);
		sprintf(fname, "hyperoval_Cherowitzo_q%ld.txt", q);
		}
	else if (f_OKeefe_Penttila) {
		P->create_OKeefe_Penttila_hyperoval_32(Pts, nb_pts, verbose_level - 2);
		sprintf(fname, "hyperoval_OKeefe_Penttila_q%ld.txt", q);
		}
	else {
		P->create_regular_hyperoval(Pts, nb_pts, verbose_level - 2);
		sprintf(fname, "hyperoval_regular_q%ld.txt", q);
		}
	
	if (f_v) {
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				INT_vec_print(cout, v, d);
				cout << endl;
				}
			}
		}

	if (!test_if_set_with_return_value(Pts, nb_pts)) {
		cout << "create_hyperoval the set is not a set, something is wrong" << endl;
		exit(1);
		}

	delete P;
	FREE_INT(v);
	//FREE_INT(L);
}

void create_subiaco_oval(finite_field *F, 
	INT f_short, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q = F->q;

	if (f_v) {
		cout << "create_subiaco_oval" << endl;
		}

	Subiaco_oval(F, Pts, nb_pts, f_short, verbose_level);
	if (f_short) {
		sprintf(fname, "oval_subiaco_short_q%ld.txt", q);
		}
	else {
		sprintf(fname, "oval_subiaco_long_q%ld.txt", q);
		}
	

	if (f_v) {
		INT i;
		INT n = 2, d = n + 1;
		INT *v;
		projective_space *P;

		v = NEW_INT(d);
		P = new projective_space;

	
		P->init(n, F, 
			FALSE /* f_init_incidence_structure */, 
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				INT_vec_print(cout, v, d);
				cout << endl;
				}
			}
		FREE_INT(v);
		delete P;
		}

	if (!test_if_set_with_return_value(Pts, nb_pts)) {
		cout << "create_subiaco_oval the set is not a set, something is wrong" << endl;
		exit(1);
		}

}

void create_subiaco_hyperoval(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q = F->q;

	if (f_v) {
		cout << "create_subiaco_hyperoval" << endl;
		}

	Subiaco_hyperoval(F, Pts, nb_pts, verbose_level);
	sprintf(fname, "subiaco_hyperoval_q%ld.txt", q);
	

	if (f_v) {
		INT i;
		INT n = 2, d = n + 1;
		INT *v;
		projective_space *P;

		v = NEW_INT(d);
		P = new projective_space;

	
		P->init(n, F, 
			FALSE /* f_init_incidence_structure */, 
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				INT_vec_print(cout, v, d);
				cout << endl;
				}
			}
		FREE_INT(v);
		delete P;
		}

	if (!test_if_set_with_return_value(Pts, nb_pts)) {
		cout << "create_subiaco_hyperoval the set is not a set, something is wrong" << endl;
		exit(1);
		}

}

void create_adelaide_hyperoval(subfield_structure *S, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	finite_field *F = S->Fq;
	INT q = F->q;

	if (f_v) {
		cout << "create_adelaide_hyperoval" << endl;
		}

	Adelaide_hyperoval(S, Pts, nb_pts, verbose_level);
	sprintf(fname, "adelaide_hyperoval_q%ld.txt", q);
	

	if (f_v) {
		INT i;
		INT n = 2, d = n + 1;
		INT *v;
		projective_space *P;

		v = NEW_INT(d);
		P = new projective_space;

	
		P->init(n, F, 
			FALSE /* f_init_incidence_structure */, 
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				INT_vec_print(cout, v, d);
				cout << endl;
				}
			}
		FREE_INT(v);
		delete P;
		}

	if (!test_if_set_with_return_value(Pts, nb_pts)) {
		cout << "create_adelaide_hyperoval the set is not a set, something is wrong" << endl;
		exit(1);
		}

}

void create_ovoid(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT n = 3, epsilon = -1;
	INT c1 = 1, c2 = 0, c3 = 0;
	INT i, j, d, h;
	INT *v, *w;
	INT q = F->q;

	d = n + 1;
	P = new projective_space;

	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = nb_pts_Qepsilon(epsilon, n, q);

	v = NEW_INT(n + 1);
	w = NEW_INT(n + 1);
	Pts = NEW_INT(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	choose_anisotropic_form(*F, c1, c2, c3, verbose_level);
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, i);
		for (h = 0; h < d; h++) {
			w[h] = v[h];
			}
		j = P->rank_point(w);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			INT_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points on the ovoid:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	//BYTE fname[1000];
	sprintf(fname, "ovoid_%ld.txt", q);
	//write_set_to_file(fname, L, N, verbose_level);

	delete P;
	FREE_INT(v);
	FREE_INT(w);
	//FREE_INT(L);
}

void create_Baer_substructure(INT n, finite_field *FQ, finite_field *Fq, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	projective_space *P2;
	INT q = Fq->q;
	INT Q = FQ->q;
	INT sz;
	INT *v;
	INT d = n + 1;
	INT i, j, a, b, index, f_is_in_subfield;

	//Q = q * q;
	P2 = new projective_space;

	P2->init(n, FQ, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level);

	if (q != i_power_j(FQ->p, FQ->e >> 1)) {
		cout << "q != i_power_j(FQ->p, FQ->e >> 1)" << endl;
		exit(1);
		}

	cout << "Q=" << Q << endl;
	cout << "q=" << q << endl;
	
	index = (Q - 1) / (q - 1);
	cout << "index=" << index << endl;
	
	v = NEW_INT(d);	
	Pts = NEW_INT(P2->N_points);
	sz = 0;
	for (i = 0; i < P2->N_points; i++) {
		PG_element_unrank_modified(*FQ, v, 1, d, i);
		for (j = 0; j < d; j++) {
			a = v[j];
			b = FQ->log_alpha(a);
			f_is_in_subfield = FALSE;
			if (a == 0 || (b % index) == 0) {
				f_is_in_subfield = TRUE;
				}
			if (!f_is_in_subfield) {
				break;
				}
			}
		if (j == d) {
			Pts[nb_pts++] = i;
			}
		}
	cout << "the Baer substructure PG(" << n << "," << q << ") inside PG(" << n << "," << Q << ") has size " << sz << ":" << endl;
	for (i = 0; i < sz; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;



	//BYTE fname[1000];
	sprintf(fname, "Baer_substructure_in_PG_%ld_%ld.txt", n, Q);
	//write_set_to_file(fname, S, sz, verbose_level);



	FREE_INT(v);
	//FREE_INT(S);
	delete P2;
}


void create_BLT_from_database(INT f_embedded, finite_field *F, INT BLT_k, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT epsilon = 0;
	INT n = 4;
	INT c1 = 0, c2 = 0, c3 = 0;
	INT d = 5;
	INT *BLT;
	INT *v;
	INT q = F->q;

	nb_pts = q + 1;

	BLT = BLT_representative(q, BLT_k);

	v = NEW_INT(d);
	Pts = NEW_INT(nb_pts);

	if (f_v) {
		cout << "i : orthogonal rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, BLT[i]);
		if (f_embedded) {
			PG_element_rank_modified(*F, v, 1, d, j);
			}
		else {
			j = BLT[i];
			}
		// recreate v:
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, BLT[i]);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : " << setw(4) << BLT[i] << " : ";
			INT_vec_print(cout, v, d);
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

	//BYTE fname[1000];
	if (f_embedded) {
		sprintf(fname, "BLT_%ld_%ld_embedded.txt", q, BLT_k);
		}
	else {
		sprintf(fname, "BLT_%ld_%ld.txt", q, BLT_k);
		}
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_INT(v);
	//FREE_INT(L);
	delete F;
}

void create_BLT(INT f_embedded, finite_field *FQ, finite_field *Fq, 
	INT f_Linear,
	INT f_Fisher,
	INT f_Mondello,
	INT f_FTWKB,
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT i, j;
	INT epsilon = 0;
	INT n = 4;
	//INT c1 = 0, c2 = 0, c3 = 0;
	//INT d = 5;
	//INT *Pts1;
	orthogonal *O;
	INT q = Fq->q;
	//INT *v;
	//BYTE BLT_label[1000];
	
	if (f_v) {
		cout << "create_BLT" << endl;
		}
	O = new orthogonal;
	if (f_v) {
		cout << "create_BLT before O->init" << endl;
		}
	O->init(epsilon, n + 1, Fq, verbose_level - 1);
	nb_pts = q + 1;

	//BLT = BLT_representative(q, BLT_k);

	//v = NEW_INT(d);
	//Pts1 = NEW_INT(nb_pts);
	Pts = NEW_INT(nb_pts);

	cout << "create_BLT currently disabled" << endl;
	exit(1);
#if 0
#if 0
	if (f_Linear) {
		strcpy(BLT_label, "Linear");
		create_Linear_BLT_set(Pts1, FQ, Fq, verbose_level - 1);
		}
	else if (f_Fisher) {
		strcpy(BLT_label, "Fi");
		create_Fisher_BLT_set(Pts1, FQ, Fq, verbose_level - 1);
		}
	else if (f_Mondello) {
		strcpy(BLT_label, "Mondello");
		create_Mondello_BLT_set(Pts1, FQ, Fq, verbose_level - 1);
		}
	else if (f_FTWKB) {
		strcpy(BLT_label, "FTWKB");
		create_FTWKB_BLT_set(O, Pts1, verbose_level - 1);
		}
	else {
		cout << "create_BLT no type" << endl;
		exit(1);
		}
#endif
	if (f_v) {
		cout << "i : orthogonal rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(*Fq, v, 1, epsilon, n, c1, c2, c3, Pts1[i]);
		if (f_embedded) {
			PG_element_rank_modified(*Fq, v, 1, d, j);
			}
		else {
			j = Pts1[i];
			}
		// recreate v:
		Q_epsilon_unrank(*Fq, v, 1, epsilon, n, c1, c2, c3, Pts1[i]);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : " << setw(4) << Pts1[i] << " : ";
			INT_vec_print(cout, v, d);
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

	//BYTE fname[1000];
	if (f_embedded) {
		sprintf(fname, "BLT_%s_%ld_embedded.txt", BLT_label, q);
		}
	else {
		sprintf(fname, "BLT_%s_%ld.txt", BLT_label, q);
		}
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_INT(Pts1);
	FREE_INT(v);
	//FREE_INT(L);
	delete O;
#endif
}

void create_orthogonal(INT epsilon, INT n, finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c1 = 1, c2 = 0, c3 = 0;
	INT i, j;
	INT d = n + 1;
	INT *v;

	nb_pts = nb_pts_Qepsilon(epsilon, n, F->q);

	v = NEW_INT(d);
	Pts = NEW_INT(nb_pts);

	if (epsilon == -1) {
		choose_anisotropic_form(*F, c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, i);
		PG_element_rank_modified(*F, v, 1, d, j);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			INT_vec_print(cout, v, d);
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

	//BYTE fname[1000];
	sprintf(fname, "Q%s_%ld_%ld.txt", plus_minus_letter(epsilon), n, F->q);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_INT(v);
	//FREE_INT(L);
}

void create_hermitian(INT n, finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT d = n + 1;
	INT *v;
	hermitian *H;

	H = new hermitian;
	H->init(F, d, verbose_level - 1);

	nb_pts = H->cnt_Sbar[d];

	v = NEW_INT(d);
	Pts = NEW_INT(nb_pts);

	if (f_v) {
		cout << "hermitian rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		H->Sbar_unrank(v, d, i, 0 /*verbose_level*/);
		PG_element_rank_modified(*F, v, 1, d, j);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			INT_vec_print(cout, v, d);
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

	//BYTE fname[1000];
	sprintf(fname, "H_%ld_%ld.txt", n, F->q);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_INT(v);
	delete H;
	//FREE_INT(L);
}


void create_twisted_cubic(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT n = 3;
	INT i, j, d, s, t;
	INT *v;
	INT v2[2];
	INT q = F->q;

	d = n + 1;
	P = new projective_space;

	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = q + 1;

	v = NEW_INT(n + 1);
	Pts = NEW_INT(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified(*F, v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		v[0] = F->mult(F->power(s, 3), F->power(t, 0));
		v[1] = F->mult(F->power(s, 2), F->power(t, 1));
		v[2] = F->mult(F->power(s, 1), F->power(t, 2));
		v[3] = F->mult(F->power(s, 0), F->power(t, 3));
		j = P->rank_point(v);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			INT_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points on the twisted cubic:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	//BYTE fname[1000];
	sprintf(fname, "twisted_cubic_%ld.txt", q);
	//write_set_to_file(fname, L, N, verbose_level);

	delete P;
	FREE_INT(v);
	//FREE_INT(L);
}



void create_ttp_code(finite_field *FQ, finite_field *Fq, 
	INT f_construction_A, INT f_hyperoval, INT f_construction_B, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	//INT q = Fq->q;
	INT i, j, d;
	INT *v;
	INT *H_subfield;
	INT m, n;
	INT f_elements_exponential = TRUE;
	const BYTE *symbol_for_print_subfield = "\\alpha";

	if (f_v) {
		cout << "create_ttp_code" << endl;
		}
	twisted_tensor_product_codes(
		H_subfield, m, n, 
		FQ, Fq, 
		f_construction_A, f_hyperoval, 
		f_construction_B,
		verbose_level - 2);
		// in GALOIS/tensor.C

	if (f_v) {
		cout << "H_subfield:" << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		print_integer_matrix_width(cout, H_subfield, m, n, n, 2);
		//f.latex_matrix(cout, f_elements_exponential, symbol_for_print_subfield, H_subfield, m, n);
		}
	
	d = m;
	P = new projective_space;

	
	P->init(d - 1, Fq, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = n;

	if (f_v) {
		cout << "H_subfield:" << endl;
		//print_integer_matrix_width(cout, H_subfield, m, n, n, 2);
		Fq->latex_matrix(cout, f_elements_exponential, symbol_for_print_subfield, H_subfield, m, n);
		}

	v = NEW_INT(d);
	Pts = NEW_INT(nb_pts);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < d; j++) {
			v[j] = H_subfield[j * n + i];
			}
		j = P->rank_point(v);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			INT_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points for the ttp code:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	//BYTE fname[1000];
	if (f_construction_A) {
		if (f_hyperoval) {
			sprintf(fname, "ttp_code_Ah_%ld.txt", Fq->q);
			}
		else {
			sprintf(fname, "ttp_code_A_%ld.txt", Fq->q);
			}
		}
	else if (f_construction_B) {
		sprintf(fname, "ttp_code_B_%ld.txt", Fq->q);
		}
	//write_set_to_file(fname, L, N, verbose_level);

	delete P;
	FREE_INT(v);
	FREE_INT(H_subfield);
}

void create_unital_XXq_YZq_ZYq(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P2;
	INT n = 2;
	INT i, rk, d;
	INT *v;

	d = n + 1;
	P2 = new projective_space;

	
	P2->init(2, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	v = NEW_INT(d);
	Pts = NEW_INT(P2->N_points);


	P2->create_unital_XXq_YZq_ZYq(Pts, nb_pts, verbose_level - 1);


	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		rk = Pts[i];
		P2->unrank_point(v, rk);
		if (f_v) {
			cout << setw(4) << i << " : ";
			INT_vec_print(cout, v, d);
			cout << " : " << setw(5) << rk << endl;
			}
		}


	sprintf(fname, "unital_XXq_YZq_ZYq_Q%ld.txt", F->q);

	delete P2;
	FREE_INT(v);
}


void create_desarguesian_line_spread_in_PG_3_q(finite_field *FQ, finite_field *Fq, 
	INT f_embedded_in_PG_4_q, 
	BYTE *fname, INT &nb_lines, INT *&Lines, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	projective_space *P1, *P3;
	//finite_field *FQ, *Fq;
	INT q = Fq->q;
	INT Q = q * q;
	INT j, h, rk, rk1, alpha, e, d;
	INT *w1, *w2, *v2;
	INT *components;
	INT *embedding;
	INT *pair_embedding;

	P1 = new projective_space;
	P3 = new projective_space;

	if (Q != FQ->q) {
		cout << "create_desarguesian_line_spread_in_PG_3_q Q != FQ->q" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "create_desarguesian_line_spread_in_PG_3_q" << endl;
		cout << "f_embedded_in_PG_4_q=" << f_embedded_in_PG_4_q << endl;
		}
	
	P1->init(1, FQ, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	if (f_embedded_in_PG_4_q) {
		P3->init(4, Fq, 
			TRUE /* f_init_incidence_structure */, 
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

		d = 5;
		}
	else {
		P3->init(3, Fq, 
			TRUE /* f_init_incidence_structure */, 
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

		d = 4;
		}



	FQ->subfield_embedding_2dimensional(*Fq, 
		components, embedding, pair_embedding, verbose_level - 3);

		// we think of FQ as two dimensional vector space 
		// over Fq with basis (1,alpha)
		// for i,j \in Fq, with x = i + j * alpha \in FQ, we have 
		// pair_embedding[i * q + j] = x;
		// also, 
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in Fq, embedding[i] is the element 
		// in FQ that corresponds to i 
		
		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]

	if (f_vv) {
		FQ->print_embedding(*Fq, 
			components, embedding, pair_embedding);
		}
	alpha = FQ->p;
	if (f_vv) {
		cout << "alpha=" << alpha << endl;
		//FQ->print(TRUE /* f_add_mult_table */);
		}


	nb_lines = Q + 1;
	Lines = NEW_INT(nb_lines);


	w1 = NEW_INT(d);
	w2 = NEW_INT(d);
	v2 = NEW_INT(2);
	
	e = FQ->e >> 1;
	if (f_vv) {
		cout << "e=" << e << endl;
		}


	INT a, a0, a1;
	INT b, b0, b1;
	
	if (f_v) {
		cout << "rk : w1,w2 : line rank" << endl;
		}
	for (rk = 0; rk < nb_lines; rk++) {
		if (f_vv) {
			cout << "rk=" << rk << endl;
			}
		P1->unrank_point(v2, rk);
			// w1[4] is the GF(q)-vector corresponding to the GF(q^2)-vector v[2]
			// w2[4] is the GF(q)-vector corresponding to the GF(q^2)-vector v[2] * alpha
			// where v[2] runs through the points of PG(1,q^2). 
			// That way, w1[4] and w2[4] are a GF(q)-basis for the 
			// 2-dimensional subspace v[2] (when viewed over GF(q)), 
			// which is an element of the regular spread.
		if (f_vv) {
			cout << "v2=";
			INT_vec_print(cout, v2, 2);
			cout << endl;
			}
						
		for (h = 0; h < 2; h++) {
			a = v2[h];
			a0 = components[a * 2 + 0];
			a1 = components[a * 2 + 1];
			b = FQ->mult(a, alpha);
			b0 = components[b * 2 + 0];
			b1 = components[b * 2 + 1];
			w1[2 * h + 0] = a0;
			w1[2 * h + 1] = a1;
			w2[2 * h + 0] = b0;
			w2[2 * h + 1] = b1;
			}
		if (f_embedded_in_PG_4_q) {
			w1[4] = 0;
			w2[4] = 0;
			}
		if (f_vv) {
			cout << "w1=";
			INT_vec_print(cout, w1, 4);
			cout << "w2=";
			INT_vec_print(cout, w2, 4);
			cout << endl;
			}

		for (j = 0; j < d; j++) {
			P3->Grass_lines->M[0 * d + j] = w1[j];
			P3->Grass_lines->M[1 * d + j] = w2[j];
			}
		if (f_vv) {
			cout << "before P3->Grass_lines->rank_INT:" << endl;
			INT_matrix_print(P3->Grass_lines->M, 2, 4);
			}
		rk1 = P3->Grass_lines->rank_INT(0 /* verbose_level*/);
		Lines[rk] = rk1;
		if (f_vv) {
			cout << setw(4) << rk << " : ";
			INT_vec_print(cout, w1, d);
			cout << ", ";
			INT_vec_print(cout, w2, d);
			cout << " : " << setw(5) << rk1 << endl;
			}
		}

	if (f_embedded_in_PG_4_q) {
		sprintf(fname, "desarguesian_line_spread_in_PG_3_%ld_embedded.txt", q);
		}
	else {
		sprintf(fname, "desarguesian_line_spread_in_PG_3_%ld.txt", q);
		}

	delete P1;
	delete P3;
	FREE_INT(w1);
	FREE_INT(w2);
	FREE_INT(v2);
	FREE_INT(components);
	FREE_INT(embedding);
	FREE_INT(pair_embedding);
}



void create_whole_space(INT n, finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT i; //, d;

	if (f_v) {
		cout << "create_whole_space" << endl;
		}
	//d = n + 1;
	P = new projective_space;

	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	Pts = NEW_INT(P->N_points);
	nb_pts = P->N_points;
	for (i = 0; i < P->N_points; i++) {
		Pts[i] = i;
		}

	sprintf(fname, "whole_space_PG_%ld_%ld.txt", n, F->q);
	
	delete P;
}

void create_hyperplane(INT n, finite_field *F, 
	INT pt, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT i, d, a;
	INT *v1;
	INT *v2;

	if (f_v) {
		cout << "create_hyperplane pt=" << pt << endl;
		}
	d = n + 1;
	P = new projective_space;
	v1 = NEW_INT(d);
	v2 = NEW_INT(d);
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	P->unrank_point(v1, pt);

	Pts = NEW_INT(P->N_points);
	nb_pts = 0;
	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v2, i);
		a = F->dot_product(d, v1, v2);
		if (a == 0) {
			Pts[nb_pts++] = i;
			if (f_v) {
				cout << setw(4) << nb_pts - 1 << " : ";
				INT_vec_print(cout, v2, d);
				cout << " : " << setw(5) << i << endl;
				}
			}
		}

	sprintf(fname, "hyperplane_PG_%ld_%ld_pt%ld.txt", n, F->q, pt);
	
	delete P;
	FREE_INT(v1);
	FREE_INT(v2);
}

void create_segre_variety(finite_field *F, INT a, INT b, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P1;
	projective_space *P2;
	projective_space *P3;
	INT i, j, d, N1, N2, rk;
	INT *v1;
	INT *v2;
	INT *v3;

	if (f_v) {
		cout << "create_segre_variety" << endl;
		cout << "a=" << a << " (projective)" << endl;
		cout << "b=" << b << " (projective)" << endl;
		}
	d = (a + 1) * (b + 1);
	if (f_v) {
		cout << "d=" << d << " (vector space dimension)" << endl;
		}
	P1 = new projective_space;
	P2 = new projective_space;
	P3 = new projective_space;
	v1 = NEW_INT(a + 1);
	v2 = NEW_INT(b + 1);
	v3 = NEW_INT(d);
	
	P1->init(a, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	P2->init(b, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	P3->init(d - 1, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);


	N1 = P1->N_points;
	N2 = P2->N_points;
	Pts = NEW_INT(N1 * N2);
	nb_pts = 0;
	for (i = 0; i < N1; i++) {
		P1->unrank_point(v1, i);
		for (j = 0; j < N2; j++) {
			P2->unrank_point(v2, j);
			F->mult_matrix_matrix(v1, v2, v3, a + 1, 1, b + 1);
			rk = P3->rank_point(v3);
			Pts[nb_pts++] = rk;
			if (f_v) {
				cout << setw(4) << nb_pts - 1 << " : " << endl;
				INT_matrix_print(v3, a + 1, b + 1);
				cout << " : " << setw(5) << rk << endl;
				}
			}
		}

	sprintf(fname, "segre_variety_%ld_%ld_%ld.txt", a, b, F->q);
	
	delete P1;
	delete P2;
	delete P3;
	FREE_INT(v1);
	FREE_INT(v2);
	FREE_INT(v3);
}

void create_Maruta_Hamada_arc(finite_field *F, 
	BYTE *fname, INT &nb_pts, INT *&Pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT N;

	if (f_v) {
		cout << "create_Maruta_Hamada_arc" << endl;
		}
	P = new projective_space;

	P->init(2, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	

	N = P->N_points;
	Pts = NEW_INT(N);

	P->create_Maruta_Hamada_arc2(Pts, nb_pts, verbose_level);

	sprintf(fname, "Maruta_Hamada_arc2_q%ld.txt", F->q);
	
	delete P;
	//FREE_INT(Pts);
}


