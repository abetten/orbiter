// blt_set2.C
// 
// Anton Betten
//
// started 8/13/2006
//
// moved here from blt_set.C Jan 23, 2016
//
//
//
//

#include "orbiter.h"
#include "discreta.h"

#include "blt.h"

void blt_set::find_free_points(INT *S, INT S_sz, 
	INT *&free_pts, INT *&free_pt_idx, INT &nb_free_pts, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *lines_on_pt;
	INT *Perp;
	INT i, j, a, b, h, f, fst, len, pt;
	classify C;

	if (f_v) {
		cout << "blt_set::find_free_points" << endl;
		}
	lines_on_pt = NEW_INT(S_sz * (q + 1));
	for (i = 0; i < S_sz; i++) {
		O->lines_on_point_by_line_rank(S[i], lines_on_pt + i * (q + 1), 0 /* verbose_level */);
		}

	if (f_vv) {
		cout << "blt_set::find_free_points Lines on partial BLT set:" << endl;
		INT_matrix_print(lines_on_pt, S_sz, q + 1);
		}

	Perp = NEW_INT(S_sz * (q + 1) * (q + 1));
	for (i = 0; i < S_sz; i++) {
		for (j = 0; j < q + 1; j++) {
			a = lines_on_pt[i * (q + 1) + j];
			O->points_on_line_by_line_rank(a, Perp + i * (q + 1) * (q + 1) + j * (q + 1), 0 /* verbose_level */);
			}
		}
	if (f_vv) {
		cout << "blt_set::find_free_points Perp:" << endl;
		INT_matrix_print(Perp, S_sz * (q + 1), q + 1);
		}
	

	C.init(Perp, S_sz * (q + 1) * (q + 1), TRUE, 0);

	C.print(FALSE /* f_reverse */);


	// find the points which are in Perp only once:
	f = C.second_type_first[0];
	nb_free_pts = C.second_type_len[0];
	if (f_v) {
		cout << "blt_set::find_free_points nb_free_pts=" << nb_free_pts << endl;
		}
	free_pts = NEW_INT(nb_free_pts);
	free_pt_idx = NEW_INT(O->nb_points);
	for (h = 0; h < O->nb_points; h++) {
		free_pt_idx[h] = -1;
		}
	
	for (h = 0; h < nb_free_pts; h++) {
		b = C.second_sorting_perm_inv[f + h];
		fst = C.type_first[b];
		len = C.type_len[b];
		if (len != 1) {
			cout << "blt_set::find_free_points len != 1" << endl;
			exit(1);
			}
		pt = C.data_sorted[fst];
		//cout << "h=" << h << " b=" << b << " len=" << len << " pt=" << pt << endl;
		free_pts[h] = pt;
		free_pt_idx[pt] = h;
		}

	FREE_INT(lines_on_pt);
	FREE_INT(Perp);

	if (f_v) {
		cout << "blt_set::find_free_points There are " << nb_free_pts << " free points" << endl;
		}
	if (f_v) {
		cout << "blt_set::find_free_points done" << endl;
		}
}

void blt_set::lifting_prepare_function_new(exact_cover *E, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_v3 = (verbose_level >= 3);
	INT i, j, a;
	
	if (f_v) {
		cout << "blt_set::lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}




	INT nb_free_points, nb_needed;
	INT *free_point_list; // [nb_free_points]
	INT *point_idx; // [nb_points_total]
		// point_idx[i] = index of a point in free_point_list 
		// or -1 if the point is in points_covered_by_starter


	nb_needed = q + 1 - starter_size;


	if (f_vv) {
		cout << "blt_set::lifting_prepare_function nb_needed=" << nb_needed << endl;
		cout << "blt_set::lifting_prepare_function nb_candidates=" << nb_candidates << endl;
		}

	if (f_v) {
		cout << "blt_set::lifting_prepare_function before find_free_points" << endl;
		}

	find_free_points(E->starter, starter_size, 
		free_point_list, point_idx, nb_free_points, 
		verbose_level - 2);

	if (f_v) {
		cout << "blt_set::lifting_prepare_function There are " << nb_free_points << " free points" << endl;
		}



	col_labels = NEW_INT(nb_candidates);


	INT_vec_copy(candidates, col_labels, nb_candidates);


	INT nb_rows = nb_free_points;
	INT nb_cols = nb_candidates;


	if (f_vv) {
		cout << "blt_set::lifting_prepare_function_new candidates: ";
		INT_vec_print(cout, candidates, nb_candidates);
		cout << " (nb_candidates=" << nb_candidates << ")" << endl;
		}




	if (E->f_lex) {
		INT nb_cols_before;

		nb_cols_before = nb_cols;
		E->lexorder_test(col_labels, nb_cols, Strong_gens->gens, 
			verbose_level - 2);
		if (f_v) {
			cout << "blt_set::lifting_prepare_function_new after lexorder test nb_candidates before: " << nb_cols_before << " reduced to  " << nb_cols << " (deleted " << nb_cols_before - nb_cols << ")" << endl;
			}
		}

	if (f_vv) {
		cout << "blt_set::lifting_prepare_function_new after lexorder test" << endl;
		cout << "blt_set::lifting_prepare_function_new nb_cols=" << nb_cols << endl;
		}

	INT *Pts1, *Pts2;

	Pts1 = NEW_INT(nb_free_points * 5);
	Pts2 = NEW_INT(nb_cols * 5);
	for (i = 0; i < nb_free_points; i++) {
		O->unrank_point(Pts1 + i * 5, 1, free_point_list[i], 0 /*verbose_level - 1*/);
		}
	for (i = 0; i < nb_cols; i++) {
		O->unrank_point(Pts2 + i * 5, 1, col_labels[i], 0 /*verbose_level - 1*/);
		}



	Dio = new diophant;
	Dio->open(nb_rows, nb_cols);
	Dio->sum = nb_needed;

	for (i = 0; i < nb_rows; i++) {
		Dio->type[i] = t_EQ;
		Dio->RHS[i] = 1;
		}

	Dio->fill_coefficient_matrix_with(0);
	if (f_vv) {
		cout << "blt_set::lifting_prepare_function_new initializing Inc" << endl;
		}


	for (i = 0; i < nb_free_points; i++) {
		for (j = 0; j < nb_cols; j++) {
			a = O->evaluate_bilinear_form(Pts1 + i * 5, Pts2 + j * 5, 1);
			if (a == 0) {
				Dio->Aij(i, j) = 1;
				}
			}
		}


	FREE_INT(free_point_list);
	FREE_INT(point_idx);
	FREE_INT(Pts1);
	FREE_INT(Pts2);
	if (f_v) {
		cout << "blt_set::lifting_prepare_function_new nb_free_points=" << nb_free_points << " nb_candidates=" << nb_candidates << endl;
		}

	if (f_v) {
		cout << "blt_set::lifting_prepare_function_new done" << endl;
		}
}

void blt_set::Law_71(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT set[100];
	INT q = 71;
	//matrix_group *M;
	//orthogonal *O;

	if (f_v) {
		cout << "Law_71" << endl;
		}

	action_on_orthogonal *AO = A->G.AO;
	orthogonal *O;

	//M = A->subaction->G.matrix_grp;
	//O = M->O;
	O = AO->O;
	//M = A->subaction->G.matrix_grp;
	//O = M->O;

	create_Law_71_BLT_set(O, set, verbose_level);
#if 0
	if (!G->check_conditions(cout, q + 1, set, verbose_level)) {
		cout << "the set is not a BLT set" << endl;
		exit(1);
		}
	cout << "BLT test passed" << endl;
#endif

	
	write_set_to_file("Law71.txt", set, q + 1, verbose_level);

#if 0
	r = G->open_database_and_identify_object(set, G->transporter, 
		G->f_use_implicit_fusion, verbose_level);
		
	cout << "Law_71 identified as r=" << r << endl;
#endif
}





void blt_set::report(isomorph &Iso, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];

	if (f_v) {
		cout << "blt_set::report" << endl;
		}
	sprintf(fname, "report_BLT_%ld.tex", q);

	{
	ofstream f(fname);
	INT f_book = TRUE;
	INT f_title = TRUE;
	BYTE title[1000];
	const BYTE *author = "Orbiter";
	INT f_toc = TRUE;
	INT f_landscape = FALSE;
	INT f_12pt = FALSE;
	INT f_enlarged_page = TRUE;
	INT f_pagenumbers = TRUE;

	sprintf(title, "BLT-sets of Q$(4,%ld)$", q);
	cout << "Writing file " << fname << " with " << Iso.Reps->count << " BLT-sets:" << endl;
	latex_head(f, f_book, f_title, 
		title, author, 
		f_toc, f_landscape, f_12pt, f_enlarged_page, f_pagenumbers, 
		NULL /* extra_praeamble */);

	f << "\\chapter{Summary}" << endl << endl;
	f << "There are " << Iso.Reps->count << " BLT-sets." << endl << endl;


	//Iso.setup_and_open_solution_database(verbose_level - 1);

	INT i, first, /* c,*/ id;
	INT u, v, h, rep, tt;
	longinteger_object go;
	INT data[1000];
	INT data2[1000];


	longinteger_object *Ago, *Ago_induced;

	Ago = new longinteger_object[Iso.Reps->count];
	Ago_induced = new longinteger_object[Iso.Reps->count];


	for (h = 0; h < Iso.Reps->count; h++) {
		if (f_v) {
			cout << "blt_set::report looking at representative h=" << h << endl;
			}
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);

		sims *Stab;
		
		Stab = Iso.Reps->stab[h];

		Iso.Reps->stab[h]->group_order(Ago[h]);
		//f << "Stabilizer has order $";
		//go.print_not_scientific(f);
		if (f_v) {
			cout << "blt_set::report computing induced action on the set (in data)" << endl;
			}
		Iso.induced_action_on_set(Stab, data, 2 /*verbose_level*/);
		if (f_v) {
			cout << "blt_set::report induced action on the set (in data) computed" << endl;
			}
		
			
		Iso.AA->group_order(Ago_induced[h]);
		}


	cout << "Computing intersection and plane invariants" << endl;
	INT **intersection_type;
	INT *highest_intersection_number;
	INT **intersection_matrix;
	INT *nb_planes;

	set_of_sets *Sos;
	set_of_sets *Sos2;
	set_of_sets *Sos3;

	decomposition *D2;
	decomposition *D3;

	grassmann *G;
	projective_space *P;
	//INT f_semilinear = TRUE;
	INT set_size = q + 1;

	P = new projective_space;
	
	if (f_v) {
		cout << "before P->init" << endl;
		}

#if 0
	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
#endif


	P->init(4, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level);

	if (f_v) {
		cout << "after P->init" << endl;
		}

	G = new grassmann;

	G->init(5, 3, F, 0 /*verbose_level - 2*/);


	longinteger_object **R;
	INT **Sos2_idx;
	INT **Sos3_idx;

	Sos = new set_of_sets[Iso.Reps->count];
	Sos2 = new set_of_sets[Iso.Reps->count];
	Sos3 = new set_of_sets[Iso.Reps->count];
	D2 = new decomposition[Iso.Reps->count];
	D3 = new decomposition[Iso.Reps->count];
	R = new plonginteger_object[Iso.Reps->count];
	Sos2_idx = NEW_PINT(Iso.Reps->count);
	Sos3_idx = NEW_PINT(Iso.Reps->count);

	if (f_v) {
		cout << "blt_set::report computing invariants" << endl;
		}
	intersection_type = NEW_PINT(Iso.Reps->count);
	highest_intersection_number = NEW_INT(Iso.Reps->count);
	intersection_matrix = NEW_PINT(Iso.Reps->count);
	nb_planes = NEW_INT(Iso.Reps->count);
	for (h = 0; h < Iso.Reps->count; h++) {
		if (f_v) {
			cout << "blt_set::report looking at representative h=" << h << endl;
			}
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);


		INT v5[5];

		for (i = 0; i < set_size; i++) {
			O->unrank_point(v5, 1, data[i], 0 /* verbose_level */);
			data2[i] = P->rank_point(v5);
			}


		if (f_v) {
			cout << "blt_set::report before P->plane_intersections" << endl;
			}
		P->plane_intersections(G, 
			data2, set_size, R[h], Sos[h], verbose_level);


		if (f_v) {
			cout << "blt_set::report before intersection_matrix" << endl;
			}
		Sos[h].intersection_matrix(
			intersection_type[h], highest_intersection_number[h], 
			intersection_matrix[h], nb_planes[h], 
			verbose_level);
		
		if (f_v) {
			cout << "blt_set::report before extract_largest_sets" << endl;
			}
		Sos[h].extract_largest_sets(Sos2[h], Sos2_idx[h], verbose_level);

		if (f_v) {
			cout << "blt_set::report before remove_sets_of_given_size" << endl;
			}
		Sos[h].remove_sets_of_given_size(3, Sos3[h], Sos3_idx[h], verbose_level);

		if (f_v) {
			cout << "blt_set::report before Sos2[h].compute_tdo_decomposition" << endl;
			}
		Sos2[h].compute_tdo_decomposition(D2[h], verbose_level);
		

		D2[h].get_row_scheme(verbose_level);
		D2[h].get_col_scheme(verbose_level);
		if (Sos3[h].nb_sets) {
			if (f_v) {
				cout << "blt_set::report before Sos3[h].compute_tdo_decomposition" << endl;
				}
			Sos3[h].compute_tdo_decomposition(D3[h], verbose_level);
			D3[h].get_row_scheme(verbose_level);
			D3[h].get_col_scheme(verbose_level);
			}
#if 0
		P->plane_intersection_invariant(G, 
			data2, set_size, 
			intersection_type[h], highest_intersection_number[h], 
			intersection_matrix[h], nb_planes[h], 
			verbose_level);
#endif
		
		}


	cout << "Computing intersection and plane invariants done" << endl;

	f << "\\chapter{Invariants}" << endl << endl;

	f << "\\chapter{The BLT-Sets}" << endl << endl;

	f << "\\clearpage" << endl << endl;


	for (h = 0; h < Iso.Reps->count; h++) {
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);



		f << "\\section{Isomorphism Type " << h << "}" << endl;
		f << "\\bigskip" << endl;

		if (Iso.Reps->stab[h]) {
			Iso.Reps->stab[h]->group_order(go);
			f << "Stabilizer has order $";
			go.print_not_scientific(f);
			f << "$\\\\" << endl;
			}
		else {
			//cout << endl;
			}

		INT a, j;
		f << "Plane intersection type is ";
		for (i = highest_intersection_number[h]; i >= 0; i--) {

			a = intersection_type[h][i];
			if (a == 0) 
				continue;
			f << "$" << i;
			if (a > 9) {
				f << "^{" << a << "}";
				}
			else if (a > 1) {
				f << "^" << a;
				}
#if 0
			if (i < nb_types - 1)
				f << ",\\,";
#endif
			f << "$ ";
			}
		f << "\\\\" << endl;
		f << "Plane invariant is ";

		if (nb_planes[h] < 10) {
			f << "$$";
			f << "\\left[" << endl;
			f << "\\begin{array}{*{" << nb_planes[h] << "}{c}}" << endl;
			for (i = 0; i < nb_planes[h]; i++) {
				for (j = 0; j < nb_planes[h]; j++) {
					f << intersection_matrix[h][i * nb_planes[h] + j];
					if (j < nb_planes[h] - 1) {
						f << " & ";
						}
					}
				f << "\\\\" << endl;
				}
			f << "\\end{array}" << endl;
			f << "\\right]" << endl;
			f << "$$" << endl;
			}
		else {
			f << "too big (" << nb_planes[h] << " planes)\\\\" << endl;
			}

		INT f_enter_math = FALSE;
		INT f_print_subscripts = TRUE;
		
		f << "$$" << endl;
		D2[h].print_row_decomposition_tex(
			f, f_enter_math, f_print_subscripts, verbose_level - 1);
		f << "\\quad" << endl;
		D2[h].print_column_decomposition_tex(
			f, f_enter_math, f_print_subscripts, verbose_level - 1);
		f << "$$" << endl;
		D2[h].Stack->print_classes_tex(f);
		
		if (Sos3[h].nb_sets) {
			f << "$$" << endl;

			D3[h].print_row_decomposition_tex(
				f, f_enter_math, f_print_subscripts, verbose_level - 1);
			f << "$$" << endl;
			f << "$$" << endl;
			D3[h].print_column_decomposition_tex(
				f, f_enter_math, f_print_subscripts, verbose_level - 1);
			f << "$$" << endl;
			D3[h].Stack->print_classes_tex(f);

			INT t, fst_col, fst, len, u, a;
			
			fst_col = D3[h].Stack->startCell[1];
			for (t = 0; t < D3[h].Stack->ht; t++) {
				if (!D3[h].Stack->is_col_class(t)) {
					continue;
					}
				f << "Column cell " << t << ":\\\\" << endl;
				len = D3[h].Stack->cellSize[t];
				fst = D3[h].Stack->startCell[t];
				INT *Cell;
				Cell = NEW_INT(len);
				for (u = 0; u < len; u++) {
					a = D3[h].Stack->pointList[fst + u] - fst_col;
					Cell[u] = a;
					}
				INT_vec_heapsort(Cell, len);
#if 0
				for (u = 0; u < len; u++) {
					a = Cell[u];
					b = Sos3_idx[h][a];
					f << a << " (rank = ";
					R[h][b].print_not_scientific(f);
					f << ") = ";
					G->unrank_longinteger(R[h][b], 0 /* verbose_level */);
					f << "$\\left[" << endl;
					f << "\\begin{array}{*{" << 5 << "}{c}}" << endl;
					for (i = 0; i < 3; i++) {
						for (j = 0; j < 5; j++) {
							c = G->M[i * 5 + j];
							f << c;
							if (j < 4) {
								f << "&";
								}
							}
						f << "\\\\" << endl;
						}
					f << "\\end{array}" << endl;
					f << "\\right]$\\\\" << endl;
					}
#endif
				FREE_INT(Cell);
				}
			}
		


		sims *Stab;
		
		Stab = Iso.Reps->stab[h];

		if (f_v) {
			cout << "blt_set::report computing induced action on the set (in data)" << endl;
			}
		Iso.induced_action_on_set(Stab, data, 0 /*verbose_level*/);
		
		longinteger_object go1;
			
		Iso.AA->group_order(go1);
		cout << "action " << Iso.AA->label << " computed, group order is " << go1 << endl;

		f << "Order of the group that is induced on the object is ";
		f << "$";
		go1.print_not_scientific(f);
		f << "$\\\\" << endl;
		
		{
		INT nb_ancestors;
		nb_ancestors = Iso.UF->count_ancestors();
		
		f << "Number of ancestors on $" << Iso.level << "$-sets is " << nb_ancestors << ".\\\\" << endl;

		INT *orbit_reps;
		INT nb_orbits;
		strong_generators *Strong_gens;
		//vector_ge SG;
		//INT *tl;
			
		Strong_gens = new strong_generators;
		//tl = NEW_INT(Iso.AA->base_len);
		Strong_gens->init_from_sims(Iso.AA->Sims, 0);
		//Iso.AA->Sims->extract_strong_generators_in_order(SG, tl, verbose_level);
		orbits_on_k_sets(Iso.AA, Iso.AA, Strong_gens /* SG, tl */, 
			Iso.level, orbit_reps, nb_orbits, verbose_level);

		f << "Number of orbits on $" << Iso.level << "$-sets is " << nb_orbits << ".\\\\" << endl;
		FREE_INT(orbit_reps);
		//FREE_INT(tl);
		delete Strong_gens;
		}

		schreier Orb;
		//longinteger_object go2;
		
		Iso.AA->compute_all_point_orbits(Orb, Stab->gens, verbose_level - 2);
		f << "With " << Orb.nb_orbits << " orbits on the object\\\\" << endl;

		classify C_ol;

		C_ol.init(Orb.orbit_len, Orb.nb_orbits, FALSE, 0);

		f << "Orbit lengths: $";
		//INT_vec_print(f, Orb.orbit_len, Orb.nb_orbits);
		C_ol.print_naked_tex(f, FALSE /* f_reverse */);
		f << "$ \\\\" << endl;
	
		tt = (target_size + 3) / 4;

		f << "The points by ranks:\\\\" << endl;
		f << "\\begin{center}" << endl;

		for (u = 0; u < 4; u++) {
			f << "\\begin{tabular}[t]{|c|c|}" << endl;
			f << "\\hline" << endl;
			f << "$i$ & Rank \\\\" << endl;
			f << "\\hline" << endl;
			for (i = 0; i < tt; i++) {
				v = u * tt + i;
				if (v < target_size) {
					f << "$" << v << "$ & $" << data[v] << "$ \\\\" << endl;
					}
				}
			f << "\\hline" << endl;
			f << "\\end{tabular}" << endl;
			}
		f << "\\end{center}" << endl; 

		f << "The points:\\\\" << endl;
		INT v5[5];
		for (i = 0; i < target_size; i++) {
			O->unrank_point(v5, 1, data[i], 0 /* verbose_level */);
			//Grass->unrank_INT(data[i], 0/*verbose_level - 4*/);
			if ((i % 4) == 0) {
				if (i) {
					f << "$$" << endl;
					}
				f << "$$" << endl;
				}
			//f << "\\left[" << endl;
			//f << "\\begin{array}{c}" << endl;
			f << "P_{" << i /*data[i]*/ << "}=";
			INT_vec_print(f, v5, 5);
#if 0
			for (u = 0; u < 5; u++) {
				for (v = 0; v < n; v++) {
					f << Grass->M[u * n + v];
					}
				f << "\\\\" << endl;
				}
#endif
			//f << "\\end{array}" << endl;
			//f << "\\right]" << endl;
			}
		f << "$$" << endl;


		longinteger_object so;

		Stab->group_order(so);
		f << "Stabilizer of order ";
		so.print_not_scientific(f);
		f << " is generated by:\\\\" << endl;
		for (i = 0; i < Stab->gens.len; i++) {
		
			INT *fp, n;
		
			fp = NEW_INT(A->degree);
			n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
			//cout << "with " << n << " fixed points" << endl;
			FREE_INT(fp);

			f << "$$ g_{" << i + 1 << "}=" << endl;
			A->element_print_latex(Stab->gens.ith(i), f);
			f << "$$" << endl << "with " << n << " fixed points" << endl;
			}



		//report_stabilizer(Iso, f, h /* orbit */, 0 /* verbose_level */);


		}


	BYTE prefix[1000];
	BYTE label_of_structure_plural[1000];

	sprintf(prefix, "BLT_%ld", q);
	sprintf(label_of_structure_plural, "BLT-Sets");
	isomorph_report_data_in_source_code_inside_tex(Iso, 
		prefix, label_of_structure_plural, f, verbose_level);



	//Iso.close_solution_database(verbose_level - 1);



	latex_foot(f);
	//FREE_INT(Rk_of_span);
	delete G;
	delete P;
	delete [] Ago;
	delete [] Ago_induced;
	}

	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	if (f_v) {
		cout << "blt_set::report done" << endl;
		}

}

void blt_set::subset_orbits(isomorph &Iso, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];

	if (f_v) {
		cout << "blt_set::subset_orbits" << endl;
		cout << "A->elt_size_in_INT=" << A->elt_size_in_INT << endl;
		}
	sprintf(fname, "report_BLT_%ld_subset_orbits.tex", q);


	Iso.load_table_of_solutions(verbose_level);
	
	Iso.depth_completed = Iso.level /*- 2*/;

	Iso.gen->recreate_schreier_vectors_up_to_level(Iso.level - 1, TRUE /* f_compact */, verbose_level);

	INT i;
	
	if (f_v) {
		for (i = 0; i <= Iso.level + 1; i++) {
			cout << "gen->first_oracle_node_at_level[" << i << "]=" << Iso.gen->first_oracle_node_at_level[i] << endl;
			}
		cout << "Iso.depth_completed=" << Iso.depth_completed << endl;
		}
	Iso.iso_test_init2(verbose_level);


	{
	ofstream f(fname);
	INT f_book = TRUE;
	INT f_title = TRUE;
	BYTE title[1000];
	const BYTE *author = "Orbiter";
	INT f_toc = TRUE;
	INT f_landscape = FALSE;
	INT f_12pt = FALSE;
	INT f_enlarged_page = TRUE;
	INT f_pagenumbers = TRUE;

	sprintf(title, "BLT-sets of Q$(4,%ld)$", q);
	cout << "Writing file " << fname << " with " << Iso.Reps->count << " BLT-sets:" << endl;
	latex_head(f, f_book, f_title, 
		title, author, 
		f_toc, f_landscape, f_12pt, f_enlarged_page, f_pagenumbers, 
		NULL /* extra_praeamble */);

	f << "\\chapter{Summary}" << endl << endl;
	f << "There are " << Iso.Reps->count << " BLT-sets." << endl << endl;


	Iso.setup_and_open_solution_database(verbose_level - 1);

	INT h, rep, first, id;
	longinteger_object go;
	INT data[1000];
	//INT data2[1000];

	for (h = 0; h < Iso.Reps->count; h++) {
		rep = Iso.Reps->rep[h];
		first = Iso.orbit_fst[rep];
		//c = Iso.starter_number[first];
		id = Iso.orbit_perm[first];		
		Iso.load_solution(id, data);



		f << "\\section{Isomorphism Type " << h << "}" << endl;
		f << "\\bigskip" << endl;

		INT_vec_print(cout, data, Iso.size);
		cout << endl;

		sims *Stab;
		
		Stab = Iso.Reps->stab[h];

		if (f_v) {
			cout << "blt_set::subset_orbits computing induced action on the set (in data)" << endl;
			}
		Iso.induced_action_on_set(Stab, data, 0 /*verbose_level*/);
		
		cout << "data after induced_action_on_set:" << endl;
		INT_vec_print(cout, data, Iso.size);
		cout << endl;
		
		longinteger_object go1;
			
		Iso.AA->group_order(go1);
		cout << "action " << Iso.AA->label << " computed, group order is " << go1 << endl;

		f << "Order of the group that is induced on the object is ";
		f << "$";
		go1.print_not_scientific(f);
		f << "$\\\\" << endl;

		{
		INT *orbit_reps;
		INT nb_orbits;
		//vector_ge SG;
		//INT *tl;
		strong_generators *Strong_gens;
		
		Strong_gens = new strong_generators;
		Strong_gens->init_from_sims(Iso.AA->Sims, 0);
		//tl = NEW_INT(Iso.AA->base_len);
		//Iso.AA->Sims->extract_strong_generators_in_order(SG, tl, verbose_level);
		orbits_on_k_sets(Iso.AA, Iso.AA, Strong_gens /* SG, tl */, 
			Iso.level, orbit_reps, nb_orbits, verbose_level);

		cout << "Orbit reps: nb_orbits=" << nb_orbits << endl;
		INT_matrix_print(orbit_reps, nb_orbits, Iso.level);

		f << "Number of orbits on $" << Iso.level << "$-sets is " << nb_orbits << ".\\\\" << endl;

		INT *rearranged_set;
		INT *transporter;
		INT u;
		INT case_nb;
		INT f_implicit_fusion = FALSE;
		INT cnt_special_orbits;
		INT f_vv = FALSE;
		INT idx;
		
		rearranged_set = NEW_INT(Iso.size);
		transporter = NEW_INT(A->elt_size_in_INT);

		cnt_special_orbits = 0;
		for (u = 0; u < nb_orbits; u++) {
			cout << "orbit " << u << ":" << endl;
			INT_vec_print(cout, orbit_reps + u * Iso.level, Iso.level);
			cout << endl;



			rearrange_subset(Iso.size, Iso.level, data, orbit_reps + u * Iso.level, rearranged_set, 0/*verbose_level - 3*/);
				// in GALOIS/sorting.C


			//INT_vec_print(cout, rearranged_set, Iso.size);
			//cout << endl;
			INT f_failure_to_find_point, f_found;

			A->element_one(transporter, 0);
			case_nb = Iso.trace_set(rearranged_set, transporter, 
				f_implicit_fusion, f_failure_to_find_point, 0 /*verbose_level - 2*/);


			f_found = Iso.find_extension_easy_new(rearranged_set, case_nb, idx, 0 /* verbose_level */);
#if 0
			f_found = Iso.identify_solution_relaxed(prefix, transporter, 
				f_implicit_fusion, orbit_no0, f_failure_to_find_point, 3 /*verbose_level*/);
#endif

			cout << "case_nb=" << case_nb << endl;
			if (f_failure_to_find_point) {
				cout << "blt_set::subset_orbits f_failure_to_find_point" << endl;
				exit(1);
				}	
			if (!f_found) {
				if (f_vv) {
					cout << "blt_set::subset_orbits not found" << endl;
					}
				continue;
				}
			cnt_special_orbits++;
			} // next u

		f << "Number of special orbits on $" << Iso.level << "$-sets is " << cnt_special_orbits << ".\\\\" << endl;

		FREE_INT(rearranged_set);
		FREE_INT(transporter);
		FREE_INT(orbit_reps);
		//FREE_INT(tl);
		delete Strong_gens;
		}

		}

	Iso.close_solution_database(verbose_level - 1);



	latex_foot(f);
	//FREE_INT(Rk_of_span);
	}

	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	if (f_v) {
		cout << "blt_set::subset_orbits done" << endl;
		}
}


// ####################################################################################
// global functions:
// ####################################################################################



void print_set(INT len, INT *S, void *data)
{
	blt_set *Gen = (blt_set *) data;
	
	//print_vector(ost, S, len);
	Gen->print(S, len);
}

INT check_conditions(INT len, INT *S, void *data, INT verbose_level)
{
	blt_set *Gen = (blt_set *) data;
	return Gen->check_conditions(len, S, verbose_level);
}

void blt_set_lifting_prepare_function_new(exact_cover *EC, INT starter_case, 
	INT *candidates, INT nb_candidates, strong_generators *Strong_gens, 
	diophant *&Dio, INT *&col_labels, 
	INT &f_ruled_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	blt_set *B = (blt_set *) EC->user_data;

	if (f_v) {
		cout << "blt_set_lifting_prepare_function_new nb_candidates=" << nb_candidates << endl;
		}

	B->lifting_prepare_function_new(EC, starter_case, 
		candidates, nb_candidates, Strong_gens, 
		Dio, col_labels, f_ruled_out, 
		verbose_level);


	if (f_v) {
		cout << "blt_set_lifting_prepare_function_new after lifting_prepare_function_new" << endl;
		}

	if (f_v) {
		cout << "blt_set_lifting_prepare_function_new nb_rows=" << Dio->m << " nb_cols=" << Dio->n << endl;
		}

	if (f_v) {
		cout << "blt_set_lifting_prepare_function_new done" << endl;
		}
}



void early_test_func_callback(INT *S, INT len, 
	INT *candidates, INT nb_candidates, 
	INT *good_candidates, INT &nb_good_candidates, 
	void *data, INT verbose_level)
{
	blt_set *BLT = (blt_set *) data;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "early_test_func for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	BLT->early_test_func(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "early_test_func done" << endl;
		}
}

INT check_function_incremental_callback(INT len, INT *S, void *data, INT verbose_level)
{
	blt_set *BLT = (blt_set *) data;
	INT f_OK;
	
	f_OK = BLT->check_function_incremental(len, S, verbose_level);
	return f_OK; 
}



void callback_report(isomorph *Iso, void *data, INT verbose_level)
{
	blt_set *Gen = (blt_set *) data;
	
	Gen->report(*Iso, verbose_level);
}

void callback_subset_orbits(isomorph *Iso, void *data, INT verbose_level)
{
	blt_set *Gen = (blt_set *) data;
	
	Gen->subset_orbits(*Iso, verbose_level);
}






