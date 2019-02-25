// packing2.C
// 
// Anton Betten
// Feb 6, 2013
//
// moved here from packing.C: Apr 25, 2016
// 
//
//

#include "orbiter.h"

namespace orbiter {
namespace top_level {


void packing::compute_list_of_lines_from_packing(
		int *list_of_lines, int *packing, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;
	
	if (f_v) {
		cout << "packing::compute_list_of_lines_from_packing" << endl;
		}
	for (i = 0; i < size_of_packing; i++) {
		a = packing[i];
		int_vec_copy(Spread_tables->spread_table + a * spread_size,
				list_of_lines + i * spread_size, spread_size);
#if 0
		for (j = 0; j < spread_size; j++) {
			b = Spread_table[a * spread_size + j];
			list_of_lines[i * spread_size + j] = b;
			}
#endif
		}
	if (f_v) {
		cout << "packing::compute_list_of_lines_from_packing done" << endl;
		}
}

void packing::compute_klein_invariants(
		isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int data[1000];
	int *list_of_lines;
	int orbit, id;
	int f_split, split_r, split_m;

	if (f_v) {
		cout << "packing::compute_klein_invariants" << endl;
		}
	f_split = f_split_klein;
	split_r = split_klein_r;
	split_m = split_klein_m;
	list_of_lines = NEW_int(size_of_packing * spread_size);
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
	
		if (f_split && (orbit % split_m) != split_r) {
			continue;
			}
		
		if (f_v) {
			cout << "packing::compute_klein_invariants orbit "
					<< orbit << " / " << Iso->Reps->count << endl;
			}
		
		char fname[1000];
	
		klein_invariants_fname(fname, Iso->prefix_invariants, orbit);
		cout << "file size of " << fname << " is " << file_size(fname) << endl;
		if (file_size(fname) > 0) {
			if (f_v) {
				cout << "file " << fname << " exists, skipping" << endl;
				}
			continue;
			}
		id = Iso->orbit_perm[Iso->orbit_fst[Iso->Reps->rep[orbit]]];
	
		Iso->load_solution(id, data);
		if (f_vv) {
			cout << "read representative of orbit " << orbit
					<< " (id=" << id << ")" << endl;
			int_vec_print(cout, data, Iso->size);
			cout << endl;
			}
		compute_list_of_lines_from_packing(list_of_lines,
				data, verbose_level - 2);
		if (f_v3) {
			cout << "read representative of orbit " << orbit
					<< " (id=" << id << ") list of lines:" << endl;
			int_matrix_print(list_of_lines,
					size_of_packing, spread_size);
			cout << endl;
			}

		save_klein_invariants(Iso->prefix_invariants, 
			orbit, 
			list_of_lines,
			size_of_packing * spread_size,
			verbose_level - 2);
		} // next orbit

	FREE_int(list_of_lines);
	
	if (f_v) {
		cout << "packing::compute_klein_invariants done" << endl;
		}
}

void packing::klein_invariants_fname(
		char *fname, char *prefix, int iso_cnt)
{
	sprintf(fname, "%s%d_klein_invariant.bin", prefix, iso_cnt);
}

void packing::save_klein_invariants(char *prefix, 
	int iso_cnt, 
	int *data, int data_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object *R;
	int **Pts_on_plane;
	int *nb_pts_on_plane;
	int nb_planes;
	int i, j;

	if (f_v) {
		cout << "packing::save_klein_invariants" << endl;
		}
	
	compute_plane_intersections(data, data_size, 
		R,
		Pts_on_plane, 
		nb_pts_on_plane, 
		nb_planes, 
		verbose_level - 2);

	Vector v;

	v.m_l(3);
	v.m_ii(0, nb_planes);
	v.s_i(1).change_to_vector();
	v.s_i(2).change_to_vector();

	v.s_i(1).as_vector().m_l(nb_planes);
	v.s_i(2).as_vector().m_l(nb_planes);
	for (i = 0; i < nb_planes; i++) {
		v.s_i(1).as_vector().m_ii(i, R[i].as_int());
		//v.s_i(1).as_vector().s_i(i).change_to_longinteger();
		//v.s_i(1).as_vector().s_i(i).as_longinteger().allocate(1, R[i].rep());
		v.s_i(2).as_vector().s_i(i).change_to_vector();
		v.s_i(2).as_vector().s_i(i).as_vector().m_l(nb_pts_on_plane[i]);
		for (j = 0; j < nb_pts_on_plane[i]; j++) {
			v.s_i(2).as_vector().s_i(i).as_vector().m_ii(j, Pts_on_plane[i][j]);
			}
		}

	char fname[1000];
	
	klein_invariants_fname(fname, prefix, iso_cnt);
	v.save_file(fname);

	delete [] R;
	for (i = 0; i < nb_planes; i++) {
		FREE_int(Pts_on_plane[i]);
		}
	FREE_pint(Pts_on_plane);
	FREE_int(nb_pts_on_plane);

	if (f_v) {
		cout << "packing::save_klein_invariants done" << endl;
		}
}


void packing::compute_plane_intersections(
	int *data, int data_size,
	longinteger_object *&R,
	int **&Pts_on_plane, 
	int *&nb_pts_on_plane, 
	int &nb_planes, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P5;
	int i;
	int N;
	int *the_set_out;
	int set_size;
	grassmann *Gr;

	if (f_v) {
		cout << "packing::compute_plane_intersections" << endl;
		}
	set_size = data_size;

	P5 = NEW_OBJECT(projective_space);
	
	P5->init(5, T->F, 
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	the_set_out = NEW_int(set_size);
	
	if (f_v) {
		cout << "packing::compute_plane_intersections "
				"before P3->klein_correspondence" << endl;
		}
	P3->klein_correspondence(P5, 
		data, set_size, the_set_out, 0/*verbose_level*/);



	if (f_v) {
		cout << "packing::compute_plane_intersections "
				"after P3->klein_correspondence" << endl;
		}

	

	N = P5->nb_rk_k_subspaces_as_int(3);
	if (f_v) {
		cout << "packing::compute_plane_intersections "
				"N = " << N << endl;
		}

	

	Gr = NEW_OBJECT(grassmann);

	Gr->init(6, 3, F, 0 /* verbose_level */);

	if (f_v) {
		cout << "packing::compute_plane_intersections "
				"before plane_intersection_type_fast" << endl;
		}
	P5->plane_intersection_type_slow(Gr, the_set_out, set_size, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes, 
		verbose_level /*- 3*/);

	if (f_v) {
		cout << "packing::compute_plane_intersections: "
				"We found " << nb_planes << " planes." << endl;
#if 1
		for (i = 0; i < nb_planes; i++) {
			cout << setw(3) << i << " : " << R[i] 
				<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
			int_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
			cout << endl; 
			}
#endif
		}


	FREE_int(the_set_out);
	FREE_OBJECT(Gr);
	FREE_OBJECT(P5);
	if (f_v) {
		cout << "packing::compute_plane_intersections done" << endl;
		}
}

void packing::report(isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	char fname[1000];

	if (f_v) {
		cout << "packing::report" << endl;
		}

	sprintf(fname, "packing_report_q%d.tex", (int)q);

	{
	ofstream f(fname);

	cout << "Writing file " << fname << " with "
			<< Iso->Reps->count << " spreads:" << endl;

	report_whole(Iso, f, verbose_level);

	} // close file f
	if (f_v) {
		cout << "packing::report written file " << fname
				<< " of size " << file_size(fname) << endl;
		}
	


	if (f_v) {
		cout << "packing::report done" << endl;
		}
}

void packing::report_whole(isomorph *Iso,
		ofstream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	report_title_page(Iso, f, verbose_level);

	f << "\\chapter{Summary}" << endl << endl;
	f << "There are " << Iso->Reps->count
			<< " packings of PG$(3," << q << ")$." << endl << endl;




	invariants_packing *inv = NULL;
	


	inv = NEW_OBJECT(invariants_packing);


	if (f_v) {
		cout << "packing::report loading and "
				"computing invariants" << endl;
		}

	inv->init(Iso, this, verbose_level);


	if (f_v) {
		cout << "packing::report loading and "
				"computing invariants done" << endl;
		}

	
	classify C_ago;

	
	C_ago.init(inv->Ago_int, Iso->Reps->count, FALSE, 0);
	f << "Classification by Ago: ";
	C_ago.print_naked_tex(f, TRUE /*f_backwards*/);
	f << "\\\\" << endl;

	f << "\\chapter{Invariants: Types of Packing}" << endl << endl;



	inv->make_table(Iso, f, FALSE, FALSE, verbose_level);

	f << "\\clearpage" << endl << endl;

	inv->make_table(Iso, f, TRUE, FALSE, verbose_level);

	f << "\\clearpage" << endl << endl;

	inv->make_table(Iso, f, FALSE, TRUE, verbose_level);

	f << "\\clearpage" << endl << endl;


	report_packings_by_ago(Iso, f, inv, C_ago, verbose_level);


	
	report_extra_stuff(Iso, f, verbose_level);
	

	latex_foot(f);
	if (inv) {
		FREE_OBJECT(inv);
		}
}

void packing::report_title_page(
		isomorph *Iso, ofstream &f, int verbose_level)
{
	int f_book = TRUE;
	int f_title = TRUE;
	char title[1000];
	const char *author = "Anton Betten";
	int f_toc = TRUE;
	int f_landscape = FALSE;
	int f_12pt = FALSE;
	int f_enlarged_page = TRUE;
	int f_pagenumbers = TRUE;

	sprintf(title, "The Packings of PG$(%d,%d)$", (int)3, (int)q);
	latex_head(f, f_book, f_title, 
		title, author, 
		f_toc, f_landscape, f_12pt,
		f_enlarged_page, f_pagenumbers,
		NULL /* extra_praeamble */);


#if 0
	char prefix[1000];
	char label_of_structure_plural[1000];

	sprintf(prefix, "packing_PG_3_%d", q);
	sprintf(label_of_structure_plural, "Packings");
	isomorph_report_data_in_source_code_inside_tex(*Iso, 
		prefix, label_of_structure_plural, f, 
		verbose_level);
#endif


}

void packing::report_packings_by_ago(
	isomorph *Iso, ofstream &f,
	invariants_packing *inv, classify &C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing::report_packings_by_ago" << endl;
		}

	f << "\\chapter{The Packings of PG$(3," << q << ")$}"
			<< endl << endl;

	f << "\\clearpage" << endl << endl;

	
	int u, v, a, cnt, fst, length, t, vv;
	int *set;

	cnt = 0;
	for (u = C_ago.nb_types - 1; u >= 0; u--) {
		fst = C_ago.type_first[u];
		length = C_ago.type_len[u];
		t = C_ago.data_sorted[fst];


		f << "\\section{Packings with a Group of Order "
				"$" << t << "$}" << endl;

		f << "There are " << length << " packings with an "
				"automorphism group of order $" << t << "$.\\\\" << endl;

		f << endl;
		f << "\\bigskip" << endl;
		f << endl;

		if (length > 100) {
			f << "Too many packings to list.\\\\" << endl;
			continue;
			}
		
		set = NEW_int(length);
		
		for (v = 0; v < length; v++) {
			vv = fst + v;
			a = C_ago.sorting_perm_inv[vv];
			set[v] = a;
			}

		int_vec_heapsort(set, length);

		for (v = 0; v < length; v++, cnt++) {

			int orbit;

			orbit = set[v];


	//for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
			report_isomorphism_type(Iso, f, 
				orbit, inv, verbose_level);
		
		//} // next orbit
			} // next v

		FREE_int(set);
		} // next u

	
	if (f_v) {
		cout << "packing::report_packings_by_ago done" << endl;
		}
}


void packing::report_isomorphism_type(
	isomorph *Iso, ofstream &f,
	int orbit, invariants_packing *inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, id, rep, first, c;
	int packing[1000];
	int spread_iso_type[1000];
	int dual_packing[1000];
	int data[1000];
	longinteger_object go;
	int *list_of_lines;

	if (f_v) {
		cout << "packing::report_isomorphism_type" << endl;
		}

	list_of_lines = NEW_int(size_of_packing * spread_size);

	rep = Iso->Reps->rep[orbit];
	first = Iso->orbit_fst[rep];
	c = Iso->starter_number[first];
	id = Iso->orbit_perm[first];		
	Iso->load_solution(id, data);

	
	for (i = 0; i < Iso->size; i++) {
		packing[i] = data[i];
		dual_packing[i] = Spread_tables->dual_spread_idx[packing[i]];
		}

	compute_list_of_lines_from_packing(
			list_of_lines, data, verbose_level - 1);


	f << "\\subsection*{Isomorphism Type " << orbit << "}" << endl;
	f << "\\bigskip" << endl;

	for (i = 0; i < Iso->size; i++) {
		spread_iso_type[i] = Spread_tables->spread_iso_type[packing[i]];
		}
	f << "spread : isotype : dualspread \\\\" << endl;
	for (i = 0; i < Iso->size; i++) {
		f << packing[i];
		f << " : ";
		f << spread_iso_type[i];
		f << " : ";
		f << Spread_tables->dual_spread_idx[packing[i]];
		f << "\\\\" << endl;
		}
	//f << "\\\\" << endl;
	f << "\\bigskip" << endl;

	classify C_iso;

	C_iso.init(spread_iso_type, Iso->size, FALSE, 0);
	f << "Classification by isomorphism type of spreads: ";
	C_iso.print_naked_tex(f, FALSE /*f_backwards*/);
	f << "\\\\" << endl;
		
		
	int dual_idx;
#if 0
	int f_implicit_fusion = TRUE;
	dual_idx = Iso->identify_database_is_open(
			dual_packing, f_implicit_fusion, verbose_level - 2);
#endif

	dual_idx = inv->Dual_idx[orbit];
	f << "The dual packing belongs to isomorphism type "
			<< dual_idx << "\\\\" << endl;
	f << "\\bigskip" << endl;

#if 0
	f << "Stabilizer has order $";
	inv->Ago[orbit].print_not_scientific(f);
	f << "$. The group that is induced has order $";
	inv->Ago_induced[orbit].print_not_scientific(f);
	f << "$\\\\" << endl;
#endif


#if 0
	f << "Plane type of Klein-image is $($ ";
	inv->Inv[orbit].C->print_naked_tex(f, TRUE /*f_backwards*/);
	f << " $)$" << endl << endl;
	f << "\\bigskip" << endl << endl;
#endif

	sims *Stab;
		
	Stab = Iso->Reps->stab[orbit];

	Stab->group_order(go);
	f << "Stabilizer has order $";
	go.print_not_scientific(f);
	f << "$\\\\" << endl;

	report_stabilizer(*Iso, f, orbit, verbose_level);

	if (f_v) {
		cout << "packing::report computing induced "
				"action on the set (in data)" << endl;
		}
	Iso->induced_action_on_set_basic(Stab, data, verbose_level - 2);
		// at the bottom of isomorph_testing.C
	if (f_v) {
		longinteger_object go;
			
		Iso->AA->group_order(go);
		cout << "action " << Iso->AA->label << " computed, "
				"group order is " << go << endl;
		}

	report_stabilizer_in_action(*Iso, f, orbit, verbose_level);

	if (go.as_int() > 2) {
		report_stabilizer_in_action_gap(*Iso, orbit, verbose_level);
		}

	schreier Orb;
	//longinteger_object go;
		
	Iso->AA->compute_all_point_orbits(Orb,
			Stab->gens, verbose_level - 2);
	//cout << "Computed all orbits on the set, "
	//"found " << Orb.nb_orbits << " orbits" << endl;
	//cout << "orbit lengths: ";
	//int_vec_print(cout, Orb.orbit_len, Orb.nb_orbits);
	//cout << endl;

	classify C;


	C.init(Orb.orbit_len, Orb.nb_orbits, FALSE, 0);


	f << "\\bigskip" << endl;

	f << "There are $" << Orb.nb_orbits
			<< "$ orbits on the set.\\\\" << endl;
	f << "The orbit type is $[$ ";
	C.print_naked_tex(f, FALSE /*f_backwards*/);
	f << " $]$\\\\" << endl;
	f << "\\bigskip" << endl;
	
	report_klein_invariants(Iso,
			f, orbit, inv, verbose_level);
		

	report_packing_as_table(Iso,
			f, orbit, inv, list_of_lines, verbose_level);

	FREE_int(list_of_lines);


	if (f_v) {
		cout << "packing::report_isomorphism_type done" << endl;
		}
}

void packing::report_packing_as_table(
	isomorph *Iso, ofstream &f,
	int orbit, invariants_packing *inv, int *list_of_lines,
	int verbose_level)
{
#if 1
	{
	int nb_points;
	int *the_spread;

	nb_points = T->Grass->nb_points_covered(0 /*verbose_level*/);

	cout << "nb_points=" << nb_points << endl;
	the_spread = NEW_int(spread_size * nb_points);

	f << "The lines of the packing are "
			"(each row corresponds to a spread):" << endl;
	f << "$$" << endl;
	f << "\\left[" << endl;
	int_matrix_print_tex(f, list_of_lines, size_of_packing, spread_size);
	f << "\\right]" << endl;
	f << "$$" << endl;

	if (T->Sing) {
		int i, j, a, b;
		f << "$$" << endl;
		f << "\\left[" << endl;
		f << "\\begin{array}{*{" << spread_size << "}{c}}" << endl;
		for (i = 0; i < size_of_packing; i++) {
			for (j = 0; j < spread_size; j++) {
				a = list_of_lines[i * spread_size + j];
				b = T->Sing->line_orbit_inv[a];
				f << T->Sing->line_orbit_label_tex[b];
				if (j < spread_size - 1) {
					f << " & ";
					}
				}
			f << "\\\\" << endl;
			}
		//int_matrix_print_tex(f,
		//list_of_lines, size_of_packing, spread_size);
		f << "\\end{array}" << endl;
		f << "\\right]" << endl;
		f << "$$" << endl;
		}
#if 0
	int u, v, a, j;
	for (u = 0; u < size_of_packing; u++) {
		f << "Spread $" << u << "$ is spread number $"
				<< packing[u] << "$.\\\\" << endl;
		f << "Lines of the spread are:" << endl;
		f << "$$" << endl;
		for (v = 0; v < spread_size; v++) {
			if (v && (v % 5) == 0) {
				f << "$$" << endl;
				f << "$$" << endl;
				}
			a = list_of_lines[u * spread_size + v];
			T->Grass->unrank_int(a, 0/*verbose_level - 4*/);
			f << "L_{" << a << "}=";
			f << "\\left[" << endl;
			f << "\\begin{array}{c}" << endl;
			for (i = 0; i < T->k; i++) {
				for (j = 0; j < T->n; j++) {
					f << T->Grass->M[i * T->n + j];
					}
				f << "\\\\" << endl;
				}
			f << "\\end{array}" << endl;
			f << "\\right]" << endl;
			} // next v
		f << "$$" << endl;
		for (v = 0; v < spread_size; v++) {
			a = list_of_lines[u * spread_size + v];
			T->Grass->unrank_int(a, 0/*verbose_level - 4*/);
			T->Grass->points_covered(
					the_spread + v * nb_points,
					0 /* verbose_level*/);
			}
		f << "The partition of the point set is:\\\\" << endl;
		f << "$$" << endl;
		f << "\\left[" << endl;
		int_matrix_print_tex(f,
				the_spread, spread_size, nb_points);
		f << "\\right]" << endl;
		f << "$$" << endl;
		} // next u
#endif
	FREE_int(the_spread);
	}
#endif
}

void packing::report_klein_invariants(
	isomorph *Iso, ofstream &f,
	int orbit, invariants_packing *inv,
	int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

		// klein invariants:
		{
			char fname_klein[1000];
			Vector V;
			
			klein_invariants_fname(fname_klein,
					Iso->prefix_invariants, orbit);
			if (file_size(fname_klein) > 0) {
				if (f_vv) {
					cout << "packing::report loading "
							"file " << fname_klein << endl;
					}
				V.load_file(fname_klein);
				inv->Inv[orbit].init_klein_invariants(
						V, verbose_level - 1);
				// free, so that we don't use that much memory:
				V.freeself();
		
				inv->Inv[orbit].compute_decomposition(verbose_level - 1);
				f << "\\bigskip" << endl << endl;
				if (file_size(inv->Inv[orbit].fname_row_scheme) < 1000) {
					f << "\\[" << endl;
					cout << "copying file "
							<< inv->Inv[orbit].fname_row_scheme
							<< " in" << endl;
					copy_file_to_ostream(f,
							inv->Inv[orbit].fname_row_scheme);
					//f << "\\input "
					//<< inv->Inv[orbit].fname_row_scheme << endl;
					f << "\\]" << endl;
					f << "\\[" << endl;
					copy_file_to_ostream(f,
							inv->Inv[orbit].fname_col_scheme);
					//f << "\\input "
					//<< inv->Inv[orbit].fname_col_scheme << endl;
					f << "\\]" << endl;
					}
				else {
					f << "The TDO decomposition is "
							"too large to print\\\\" << endl;
					}
				}
		}
}

void packing::report_stabilizer(isomorph &Iso,
		ofstream &f, int orbit, int verbose_level)
{
	sims *Stab;
	longinteger_object go;
	int i;

	Stab = Iso.Reps->stab[orbit];
	Stab->group_order(go);

	f << "The stabilizer of order $" << go
			<< "$ is generated by:\\\\" << endl;
	f << "$$" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		//int *fp, n, ord;
		
		//fp = NEW_int(A->degree);
		//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		//FREE_int(fp);

		//ord = A->element_order(Stab->gens.ith(i));

		//f << "$$ g_{" << i + 1 << "}=" << endl;
		T->A->element_print_latex(Stab->gens.ith(i), f);
		//f << "$$" << endl;
		//f << "of order $" << ord << "$ and with " << n
		//<< " fixed points" << endl;
		}
	f << "$$" << endl;
}

void packing::report_stabilizer_in_action(
		isomorph &Iso, ofstream &f, int orbit,
		int verbose_level)
{
	sims *Stab;
	longinteger_object go;
	int i;

	Stab = Iso.Reps->stab[orbit];
	Stab->group_order(go);

	f << "The stabilizer generators in their action "
			"on the spreads of the packing are:\\\\" << endl;
	//f << "$$" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = FALSE;

		//int *fp, n, ord;
		
		//fp = NEW_int(A->degree);
		//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		//FREE_int(fp);

		//ord = A->element_order(Stab->gens.ith(i));

		f << "$";
		Iso.AA->element_print_as_permutation_with_offset(
			Stab->gens.ith(i), f,
			offset, f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one, 0 /* verbose_level */);
		f << "$\\\\" << endl;
		//f << "of order $" << ord << "$ and with " << n
		//<< " fixed points" << endl;
		}
	//f << "$$" << endl;
}

void packing::report_stabilizer_in_action_gap(
		isomorph &Iso, int orbit,
		int verbose_level)
{
	sims *Stab;
	longinteger_object go;
	int i;
	char fname[1000];


	sprintf(fname, "group_%d.g", orbit);
	{
	ofstream fp(fname);


	Stab = Iso.Reps->stab[orbit];
	Stab->group_order(go);

	//f << "The stabilizer generators in their action on "
	//"the spreads of the packing are:\\\\" << endl;
	//f << "$$" << endl;
	for (i = 0; i < Stab->gens.len; i++) {
		
		int offset = 1;
		int f_do_it_anyway_even_for_big_degree = TRUE;
		int f_print_cycles_of_length_one = FALSE;

		//int *fp, n, ord;
		
		//fp = NEW_int(A->degree);
		//n = A->find_fixed_points(Stab->gens.ith(i), fp, 0);
		//cout << "with " << n << " fixed points" << endl;
		//FREE_int(fp);

		//ord = A->element_order(Stab->gens.ith(i));

		fp << "g" << i + 1 << " := ";
		Iso.AA->element_print_as_permutation_with_offset(
			Stab->gens.ith(i), fp,
			offset, f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one, 0 /* verbose_level */);
		fp << ";" << endl;
		//f << "of order $" << ord << "$ and with " << n
		//<< " fixed points" << endl;
		}
	fp << "G := Group([";
	for (i = 0; i < Stab->gens.len; i++) {
		fp << "g" << i + 1;
		if (i < Stab->gens.len - 1) {
			fp << ",";
			}
		}
	fp << "]);" << endl;
	}

}

void packing::report_extra_stuff(
		isomorph *Iso, ofstream &f,
		int verbose_level)
{
	f << "\\chapter{The Field GF$(" << q << ")$}" << endl << endl;
	
	T->F->cheat_sheet(f, verbose_level - 1);


	f << "\\chapter{The Points and Lines of "
			"PG$(3," << q << ")$}" << endl << endl;

	//f << "\\clearpage" << endl << endl;

	{
	int nb_points;
	int nb_lines;
	int v[4];
	int i, j, u;

	nb_points = P3->N_points;
	nb_lines = generalized_binomial(4, 2, q);

	f << "PG$(3," << q << ")$ has " << nb_points
			<< " points:\\\\" << endl;
	for (i = 0; i < nb_points; i++) {
		P3->unrank_point(v, i);
		f << "$P_{" << i << "}=";
		int_vec_print_fully(f, v, 4);
		f << "$\\\\" << endl;
		}
	f << endl;
	f << "\\bigskip" << endl;
	f << endl;
	f << "PG$(3," << q << ")$ has " << nb_lines
			<< " lines:\\\\" << endl;
	for (u = 0; u < nb_lines; u++) {
		T->Grass->unrank_int(u, 0 /* verbose_level*/);
		f << "$L_{" << u << "}=";
		f << "\\left[" << endl;
		f << "\\begin{array}{c}" << endl;
		for (i = 0; i < T->k; i++) {
			for (j = 0; j < T->n; j++) {
				f << T->Grass->M[i * T->n + j];
				}
			f << "\\\\" << endl;
			}
		f << "\\end{array}" << endl;
		f << "\\right]" << endl;
		f << " = \\{" << endl;
		for (i = 0; i < P3->k; i++) {
			f << P3->Lines[u * P3->k + i];
			if (i < P3->k - 1) {
				f << ", ";
				}
			}
		f << "\\}" << endl;
		f << "$\\\\" << endl;
		}

#if 1
	f << "\\chapter{The Spreads of PG$(3," << q << ")$}"
			<< endl << endl;

	f << "PG$(3," << q << ")$ has " << Spread_tables->nb_spreads
			<< " labeled spreads\\\\" << endl;

	for (u = 0; u < Spread_tables->nb_spreads; u++) {
		f << "Spread " << u << " is $";
		int_vec_print_fully(f,
				Spread_tables->spread_table + u * spread_size, spread_size);
		f << "$ isomorphism type "
				<< Spread_tables->spread_iso_type[u] << "\\\\" << endl;
		
		}
#endif

	}
}

}}

