// orthogonal_group.cpp
// 
// Anton Betten
// 10/17/2007
//
// 
//
//

#include "orbiter.h"

using namespace std;
using namespace orbiter;

// global data:

int t0; // the system time when the program started

void usage(int argc, char **argv);
int main(int argc, char **argv);
void do_it(int epsilon, int n, int q,
		int f_draw_tree,
		int f_orbit_of, int orbit_of_idx,
		int f_report, int f_sylow,
		int verbose_level);

void usage(int argc, char **argv)
{
	cout << "usage: " << argv[0] << " [options]" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <n>                   : verbose level n" << endl;
	cout << "-epsilon <epsilon>       : set form type epsilon" << endl;
	cout << "-d <d>                   : set dimension d" << endl;
	cout << "-q <q>                   : set field size q" << endl;
}



int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_epsilon = FALSE;
	int epsilon = 0;
	int f_d = FALSE;
	int d = 0;
	int f_q = FALSE;
	int q = 0;
	int f_draw_tree = FALSE;
	int f_orbit_of = FALSE;
	int orbit_of_idx = 0;
	int f_report = FALSE;
	int f_sylow = FALSE;
	os_interface Os;

	t0 = Os.os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-h") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-help") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
			}
		else if (strcmp(argv[i], "-sylow") == 0) {
			f_sylow = TRUE;
			cout << "-sylow" << endl;
			}
		else if (strcmp(argv[i], "-f_draw_tree") == 0) {
			f_draw_tree = TRUE;
			cout << "-f_draw_tree " << endl;
			}
		else if (strcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = TRUE;
			orbit_of_idx = atoi(argv[++i]);
			cout << "-orbit_of " << orbit_of_idx << endl;
			}
		}
	if (!f_epsilon) {
		cout << "please use -epsilon <epsilon>" << endl;
		usage(argc, argv);
		exit(1);
		}	
	if (!f_d) {
		cout << "please use -d <d>" << endl;
		usage(argc, argv);
		exit(1);
		}	
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		usage(argc, argv);
		exit(1);
		}	
	do_it(epsilon, d, q,
			f_draw_tree,
			f_orbit_of, orbit_of_idx,
			f_report, f_sylow,
			verbose_level);

	the_end_quietly(t0);
}

void do_it(int epsilon, int n, int q,
		int f_draw_tree,
		int f_orbit_of, int orbit_of_idx,
		int f_report, int f_sylow,
		int verbose_level)
{
	finite_field *F;
	action *A;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int p, h, i, j, a;
	int *v;
	number_theory_domain NT;
	
	A = NEW_OBJECT(action);
	NT.is_prime_power(q, p, h);
	if (h > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;
	
	v = NEW_int(n);

	
	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	A->init_orthogonal_group(epsilon, n, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, verbose_level);
	

	if (!A->f_has_strong_generators) {
		cout << "action does not have strong generators" << endl;
		exit(1);
		}
	strong_generators *SG;
	longinteger_object go;
	action_on_orthogonal *AO = A->G.AO;
	orthogonal *O = AO->O;

	SG = A->Strong_gens;
	SG->group_order(go);

	cout << "The group " << A->label << " has order "
			<< go << " and permutation degree "
			<< A->degree << endl;
	cout << "The points on which the group acts are:" << endl;
	if (A->degree < 1000) {
		for (i = 0; i < A->degree; i++) {
			O->unrank_point(v, 1 /* stride */, i, 0 /* verbose_level */);
			cout << i << " / " << A->degree << " : ";
			int_vec_print(cout, v, n);
			cout << endl;
		}
	} else {
		cout << "Too many to print" << endl;
	}
	cout << "Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		if (A->degree < 1000) {
			cout << "as permutation: " << endl;
			A->element_print_as_permutation(
					SG->gens->ith(i), cout);
			cout << endl;
		}
	}
	if (A->degree < 1000) {
		cout << "Generators are:" << endl;
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
			}
		cout << "Generators in compact permutation form are:" << endl;
		cout << SG->gens->len << " " << A->degree << endl;
		for (i = 0; i < SG->gens->len; i++) {
			for (j = 0; j < A->degree; j++) {
				a = A->element_image_of(j,
						SG->gens->ith(i), 0 /* verbose_level */);
				cout << a << " ";
				}
			cout << endl;
			}
		cout << "-1" << endl;
	}


	if (f_orbit_of) {

		schreier *Sch;
		Sch = NEW_OBJECT(schreier);

		cout << "computing orbit of point " << orbit_of_idx << ":" << endl;

		//A->all_point_orbits(*Sch, verbose_level);

		Sch->init(A, verbose_level - 2);
		if (!A->f_has_strong_generators) {
			cout << "action::all_point_orbits !f_has_strong_generators" << endl;
			exit(1);
			}
		Sch->init_generators(*A->Strong_gens->gens /* *strong_generators */, verbose_level - 2);
		Sch->initialize_tables();
		Sch->compute_point_orbit(orbit_of_idx, verbose_level);


		cout << "computing orbit of point done." << endl;

		char fname_tree_mask[1000];

		sprintf(fname_tree_mask, "%s_orbit_%%d.layered_graph", A->label);

		Sch->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);

		int orbit_idx = 0;
		schreier *shallow_tree;

		cout << "computing shallow Schreier tree:" << endl;

#if 0
		enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
				//shallow_schreier_tree_standard;
				//shallow_schreier_tree_Seress_deterministic;
				shallow_schreier_tree_Seress_randomized;
				//shallow_schreier_tree_Sajeeb;
#endif
		int f_randomized = TRUE;

		Sch->shallow_tree_generators(orbit_idx,
				f_randomized,
				shallow_tree,
				verbose_level);
		cout << "computing shallow Schreier tree done." << endl;

		sprintf(fname_tree_mask, "%s_orbit_%%d_shallow.layered_graph", A->label);

		shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);

		if (f_draw_tree) {

			cout << "drawing the Schreier tree" << endl;

			char fname_tree[1000];
			char fname_report[1000];
			int xmax = 2000000;
			int ymax = 1000000;
			int f_circletext = TRUE;
			int rad = 18000;
			int f_embedded = FALSE;
			int f_sideways = TRUE;
			double scale = 0.35;
			double line_width = 1.0;

			sprintf(fname_tree, "%s_tree", A->label);
			sprintf(fname_report, "%s_orbit_report.tex", A->label);


			Sch->draw_tree(fname_tree, 0 /* orbit_no*/,
					xmax, ymax, f_circletext, rad,
					f_embedded, f_sideways,
					scale, line_width,
					FALSE /* f_has_point_labels */, NULL /*  *point_labels */,
					verbose_level);


			{
			ofstream fp(fname_report);
			latex_interface L;


			L.head_easy(fp);

			SG->print_generators_tex(fp);

			//fp << "Schreier tree:" << endl;
			fp << "\\input " << fname_tree << ".tex" << endl;


			if (q == 3 && n == 5) {
				int u[] = { // singular vectors
						0,0,1,1,0,
						1,2,0,2,1,
						0,0,0,1,0,
						1,0,0,2,1,
						1,1,2,0,2
				};
				int v[] = { // v is orthogonal to u
						0,1,2,0,2,
						2,0,1,2,2,
						1,0,0,0,0,
						2,0,2,0,1,
						0,2,2,0,2
				};
				int w[] = {
						1,1,1,1,0
				};
				int *Mtx;

				Mtx = NEW_int(6 * 25);
				for (i = 0; i < 5; i++) {
					cout << "creating Siegel transformation " << i << " / 5:" << endl;
					F->Siegel_Transformation(0 /*epsilon */, n - 1,
							1 /*form_c1*/, 0 /*form_c2*/, 0 /*form_c3*/,
							Mtx + i * 25, v + i * 5, u + i * 5, verbose_level);
					int_matrix_print(Mtx + i * 25, 5, 5);
					cout << endl;
				}
				O->make_orthogonal_reflection(Mtx + 5 * 25, w, verbose_level - 1);
				int_matrix_print(Mtx + 5 * 25, 5, 5);
				cout << endl;
				cout << "generators for O(5,3) are:" << endl;
				int_matrix_print(Mtx, 6, 25);

				vector_ge *gens;
				gens = NEW_OBJECT(vector_ge);
				gens->init_from_data(A, Mtx,
						6 /* nb_elements */, 25 /* elt_size */, verbose_level);
				gens->print(cout);
				schreier *Sch2;
				Sch2 = NEW_OBJECT(schreier);

				cout << "computing orbits on points:" << endl;
				Sch2->init(A, verbose_level - 2);
				Sch2->init_generators(*gens, verbose_level - 2);
				Sch2->compute_all_point_orbits(verbose_level);

				char fname_tree2[1000];

				sprintf(fname_tree2, "%s_tree2", A->label);
				Sch2->draw_tree(fname_tree2, 0 /* orbit_no*/,
						xmax, ymax, f_circletext, rad,
						f_embedded, f_sideways,
						scale, line_width,
						FALSE /* f_has_point_labels */, NULL /*  *point_labels */,
						verbose_level);

				longinteger_object go;
				A->group_order(go);

				gens->print_generators_tex(go, fp);


				//fp << "Schreier tree:" << endl;
				fp << "\\input " << fname_tree2 << ".tex" << endl;



			}


			L.foot(fp);

			} // end fname_report
		} // if (f_tree)
	} // if (f_orbit_of)


	if (f_report) {
		char fname[1000];
		char title[1000];
		const char *author = "Orbiter";
		const char *extras_for_preamble = "";

		sprintf(fname, "%s_report.tex", A->label);
		sprintf(title, "The group $%s$", A->label_tex);

		{
			ofstream fp(fname);
			latex_interface L;
			//latex_head_easy(fp);
			L.head(fp,
				FALSE /* f_book */, TRUE /* f_title */,
				title, author,
				FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
				TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
				extras_for_preamble);

			sims *H;
			int *Elt;
			longinteger_object go;

			H = A->Strong_gens->create_sims(verbose_level);

			Elt = NEW_int(A->elt_size_in_int);
			H->group_order(go);

			fp << "\\section{The Group $" << A->label_tex << "$}" << endl;


			H->group_order(go);

			fp << "\\noindent The order of the group $"
					<< A->label_tex
					<< "$ is " << go << "\\\\" << endl;

			fp << "\\noindent The field ${\\mathbb F}_{"
					<< F->q
					<< "}$ :\\\\" << endl;
			F->cheat_sheet(fp, verbose_level);


			fp << "\\noindent The group acts on a set of size "
					<< A->degree << "\\\\" << endl;

			if (A->degree < 1000) {

				A->print_points(fp);
			}

			//cout << "Order H = " << H->group_order_int() << "\\\\" << endl;


			SG->print_generators_tex(fp);
			//LG->report(fp, f_sylow, verbose_level);

			A->report(fp, TRUE /*f_sims*/, H,
					TRUE /* f_strong_gens */, A->Strong_gens, verbose_level);

			A->report_basic_orbits(fp);

			if (f_sylow) {
				sylow_structure *Syl;

				Syl = NEW_OBJECT(sylow_structure);
				Syl->init(H, verbose_level);
				Syl->report(fp);

				A->report_conjugacy_classes_and_normalizers(fp,
						verbose_level);
			}


			FREE_int(Elt);
			FREE_OBJECT(H);
			L.foot(fp);
		}
	}




	FREE_int(v);
	FREE_OBJECT(A);
	FREE_OBJECT(F);


}


