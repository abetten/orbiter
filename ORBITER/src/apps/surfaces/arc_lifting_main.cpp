// arc_lifting_main.C
// 
// Anton Betten, Fatma Karaoglu
//
// February 15, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace orbiter;
using namespace orbiter::top_level;


int t0 = 0;


int main(int argc, const char **argv);
void lift_single_arc(int *arc, int arc_size,
		surface_with_action *Surf_A, int verbose_level);
void classify_arcs_and_do_arc_lifting(int argc, const char **argv,
		surface_with_action *Surf_A, int verbose_level);


int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;
	int f_arc = FALSE;
	const char *the_arc_text = NULL;
	int f_classify = FALSE;
	int i;

	t0 = os_ticks();


	surface *Surf;
	surface_with_action *Surf_A;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	Surf = NEW_OBJECT(surface);
	Surf_A = NEW_OBJECT(surface_with_action);


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-verbose_level " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-arc") == 0) {
			f_arc = TRUE;
			the_arc_text = argv[++i];
			cout << "-arc " << the_arc_text << endl;
			}
		else if (strcmp(argv[i], "-classify") == 0) {
			f_classify = TRUE;
			cout << "-classify " << endl;
			}
		}


	int f_v = (verbose_level >= 1);
		
	F->init(q, 0);

	if (f_v) {
		cout << "before Surf->init" << endl;
		}
	Surf->init(F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf->init" << endl;
		}

	int f_semilinear;

	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


#if 0
	if (f_v) {
		cout << "before Surf->init_large_polynomial_domains" << endl;
		}
	Surf->init_large_polynomial_domains(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf->init_large_polynomial_domains" << endl;
		}
#endif


	if (f_v) {
		cout << "before Surf_A->init" << endl;
		}
	Surf_A->init(Surf, f_semilinear, 0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->init" << endl;
		}
	


	if (f_v) {
		cout << "before Surf_A->Classify_trihedral_"
				"pairs->classify" << endl;
		}
	Surf_A->Classify_trihedral_pairs->classify(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->Classify_trihedral_"
				"pairs->classify" << endl;
		}


	

	if (f_arc) {
		int *arc;
		int arc_size;

		int_vec_scan(the_arc_text, arc, arc_size);
		
		if (f_v) {
			cout << "before lift_single_arc" << endl;
			}
		lift_single_arc(arc, arc_size, Surf_A, verbose_level);
		if (f_v) {
			cout << "after lift_single_arc" << endl;
			}
		}
	
	if (f_classify) {
		if (f_v) {
			cout << "before classify_arcs_and_do_arc_lifting" << endl;
			}
		classify_arcs_and_do_arc_lifting(argc, argv,
				Surf_A, verbose_level);
		}

	the_end(t0);
	//the_end_quietly(t0);
	
}


void lift_single_arc(int *arc, int arc_size,
		surface_with_action *Surf_A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	surface *Surf;
	finite_field *F;
	char fname_arc_lifting[1000];

	F = Surf_A->F;
	q = F->q;
	Surf = Surf_A->Surf;


	if (f_v) {
		cout << "lift_single_arc q=" << q << endl;
		cout << "Lifting the arc ";
		int_vec_print(cout, arc, arc_size);
		cout << endl;
		}


	if (arc_size != 6) {
		cout << "lift_single_arc arc_size != 6" << endl;
		exit(1);
		}

	{
	char title[10000];
	char author[10000];
	int i;
	
	sprintf(title, "Lifting a single arc over GF(%d) ", q);
	sprintf(author, "");

	sprintf(fname_arc_lifting, "single_arc_lifting_q%d_arc", q);

	for (i = 0; i < 6; i++) {
		sprintf(fname_arc_lifting + strlen(fname_arc_lifting), "_%d",
				arc[i]);
		}
	sprintf(fname_arc_lifting + strlen(fname_arc_lifting), ".tex");
	ofstream fp(fname_arc_lifting);


	latex_head(fp,
		FALSE /* f_book */,
		TRUE /* f_title */,
		title, author, 
		FALSE /*f_toc */,
		FALSE /* f_landscape */,
		FALSE /* f_12pt */,
		TRUE /*f_enlarged_page */,
		TRUE /* f_pagenumbers*/,
		NULL /* extra_praeamble */);


	fp << "We are lifting the arc ";
	int_vec_print(fp, arc, arc_size);
	fp << endl;

	fp << "consisting of the following points:\\\\" << endl;
	display_table_of_projective_points(fp, F, arc, 6, 3);
	



	cout << "classify_arcs_and_do_arc_lifting before Surf_A->list_"
			"orbits_on_trihedra_type1" << endl;
	Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type1(fp);

	cout << "classify_arcs_and_do_arc_lifting before Surf_A->list_"
			"orbits_on_trihedra_type2" << endl;
	Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type2(fp);

	cout << "classify_arcs_and_do_arc_lifting before Surf_A->print_"
			"trihedral_pairs no stabs" << endl;
	Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
			FALSE /* f_with_stabilizers */);

	cout << "classify_arcs_and_do_arc_lifting before Surf_A->print_"
			"trihedral_pairs with stabs" << endl;
	Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
			TRUE /* f_with_stabilizers */);



	arc_lifting *AL;

	AL = NEW_OBJECT(arc_lifting);


	if (f_v) {
		cout << "lift_single_arc before AL->create_surface" << endl;
		}
	AL->create_surface(Surf_A, arc, verbose_level);
	if (f_v) {
		cout << "lift_single_arc after AL->create_surface" << endl;
		}

	AL->print(fp);


	FREE_OBJECT(AL);
	
	latex_foot(fp);
	} // fp

	cout << "Written file " << fname_arc_lifting << " of size "
			<< file_size(fname_arc_lifting) << endl;

	if (f_v) {
		cout << "lift_single_arc done" << endl;
		}
}

void classify_arcs_and_do_arc_lifting(int argc, const char **argv,
		surface_with_action *Surf_A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	surface *Surf;
	finite_field *F;
	int i, j, arc_idx;
	
	char fname_arc_lifting[10000];

	F = Surf_A->F;
	q = F->q;
	Surf = Surf_A->Surf;


	six_arcs_not_on_a_conic *Six_arcs;

	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);
	

	// classify six arcs not on a conic:

	if (f_v) {
		cout << "before Six_arcs->init" << endl;
		}
	Six_arcs->init(F, Surf->P2, 
		argc, argv, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "after Six_arcs->init" << endl;
		}



	{
	char title[10000];
	char author[10000];
	sprintf(title, "Arc lifting over GF(%d) ", q);
	sprintf(author, "");

	sprintf(fname_arc_lifting, "arc_lifting_q%d.tex", q);
	ofstream fp(fname_arc_lifting);


	latex_head(fp,
		FALSE /* f_book */,
		TRUE /* f_title */,
		title, author, 
		FALSE /*f_toc */,
		FALSE /* f_landscape */,
		FALSE /* f_12pt */,
		TRUE /*f_enlarged_page */,
		TRUE /* f_pagenumbers*/,
		NULL /* extra_praeamble */);


	if (f_v) {
		cout << "do_arc_lifting q=" << q << endl;
		}



	Six_arcs->report_latex(fp);

	if (f_v) {
		Surf->print_polynomial_domains(fp);
		Surf->print_line_labelling(fp);


		cout << "classify_arcs_and_do_arc_lifting "
				"before Surf->print_Steiner_and_Eckardt" << endl;
		Surf->print_Steiner_and_Eckardt(fp);
		cout << "classify_arcs_and_do_arc_lifting "
				"after Surf->print_Steiner_and_Eckardt" << endl;

		cout << "classify_arcs_and_do_arc_lifting "
				"before Surf->print_clebsch_P" << endl;
		Surf->print_clebsch_P(fp);
		cout << "classify_arcs_and_do_arc_lifting "
				"after Surf->print_clebsch_P" << endl;
	


		cout << "classify_arcs_and_do_arc_lifting "
				"before Surf_A->list_orbits_on_trihedra_type1" << endl;
		Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type1(fp);

		cout << "classify_arcs_and_do_arc_lifting "
				"before Surf_A->list_orbits_on_trihedra_type2" << endl;
		Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type2(fp);

		cout << "classify_arcs_and_do_arc_lifting "
				"before Surf_A->print_trihedral_pairs no stabs" << endl;
		Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
				FALSE /* f_with_stabilizers */);

		cout << "classify_arcs_and_do_arc_lifting "
				"before Surf_A->print_trihedral_pairs with stabs" << endl;
		Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
				TRUE /* f_with_stabilizers */);

		}



	char fname_base[1000];
	sprintf(fname_base, "arcs_q%d", q);

	if (q < 20) {
		cout << "before Gen->gen->draw_poset_full" << endl;
		Six_arcs->Gen->gen->draw_poset(
			fname_base,
			6 /* depth */, 0 /* data */,
			TRUE /* f_embedded */,
			FALSE /* f_sideways */,
			verbose_level);
		}


	int *f_deleted; // [Six_arcs->nb_arcs_not_on_conic]
	int *Arc_identify; //[Six_arcs->nb_arcs_not_on_conic *
				// Six_arcs->nb_arcs_not_on_conic]
	int *Arc_identify_nb; // [Six_arcs->nb_arcs_not_on_conic]
	int Arc6[6];
	int nb_surfaces;

	nb_surfaces = 0;

	f_deleted = NEW_int(Six_arcs->nb_arcs_not_on_conic);
	Arc_identify = NEW_int(Six_arcs->nb_arcs_not_on_conic *
			Six_arcs->nb_arcs_not_on_conic);
	Arc_identify_nb = NEW_int(Six_arcs->nb_arcs_not_on_conic);

	int_vec_zero(f_deleted, Six_arcs->nb_arcs_not_on_conic);
	int_vec_zero(Arc_identify_nb, Six_arcs->nb_arcs_not_on_conic);

	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {


		if (f_deleted[arc_idx]) {
			continue;
			}
		

		if (f_v) {
			cout << "classify_arcs_and_do_arc_lifting extending arc "
					<< arc_idx << " / "
					<< Six_arcs->nb_arcs_not_on_conic << ":" << endl;
			}

		fp << "\\clearpage\n\\section{Extending arc " << arc_idx
				<< " / " << Six_arcs->nb_arcs_not_on_conic << "}" << endl;

		Six_arcs->Gen->gen->get_set_by_level(
				6 /* level */,
				Six_arcs->Not_on_conic_idx[arc_idx],
				Arc6);
		
		{
		set_and_stabilizer *The_arc;

		The_arc = Six_arcs->Gen->gen->get_set_and_stabilizer(
				6 /* level */,
				Six_arcs->Not_on_conic_idx[arc_idx],
				0 /* verbose_level */);
		
		
		fp << "Arc " << arc_idx << " / "
				<< Six_arcs->nb_arcs_not_on_conic << " is: ";
		fp << "$$" << endl;
		//int_vec_print(fp, Arc6, 6);
		The_arc->print_set_tex(fp);
		fp << "$$" << endl;

		display_table_of_projective_points(fp, F, 
			The_arc->data, 6, 3);


		fp << "The stabilizer is the following group:\\\\" << endl;
		The_arc->Strong_gens->print_generators_tex(fp);

		FREE_OBJECT(The_arc);
		}

		char arc_label[1000];
		char arc_label_short[1000];
		
		sprintf(arc_label, "%d / %d",
				arc_idx, Six_arcs->nb_arcs_not_on_conic);
		sprintf(arc_label_short, "Arc%d", arc_idx);
		
		if (f_v) {
			cout << "classify_arcs_and_do_arc_lifting "
					"before do_arc_lifting" << endl;
			}

		Surf_A->arc_lifting_and_classify(
			TRUE /* f_log_fp */,
			fp,
			Arc6,
			arc_label,
			arc_label_short,
			nb_surfaces, 
			Six_arcs, 
			Arc_identify_nb, 
			Arc_identify, 
			f_deleted, 
			verbose_level);

		if (f_v) {
			cout << "classify_arcs_and_do_arc_lifting "
					"after do_arc_lifting" << endl;
			}
		

		
		nb_surfaces++;
		} // next arc_idx

	cout << "We found " << nb_surfaces << " surfaces" << endl;


	cout << "decomposition matrix:" << endl;
	for (i = 0; i < nb_surfaces; i++) {
		for (j = 0; j < Arc_identify_nb[i]; j++) {
			cout << Arc_identify[i * Six_arcs->nb_arcs_not_on_conic + j];
			if (j < Arc_identify_nb[i] - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}
	int *Decomp;
	int a;

	Decomp = NEW_int(Six_arcs->nb_arcs_not_on_conic * nb_surfaces);
	int_vec_zero(Decomp, Six_arcs->nb_arcs_not_on_conic * nb_surfaces);
	for (i = 0; i < nb_surfaces; i++) {
		for (j = 0; j < Arc_identify_nb[i]; j++) {
			a = Arc_identify[i * Six_arcs->nb_arcs_not_on_conic + j];
			Decomp[a * nb_surfaces + i]++;
			}
		}
	
	cout << "decomposition matrix:" << endl;
	cout << "$$" << endl;
	print_integer_matrix_with_standard_labels(cout, Decomp,
			Six_arcs->nb_arcs_not_on_conic, nb_surfaces,
			TRUE /* f_tex */);
	cout << "$$" << endl;
	
	fp << "Decomposition matrix:" << endl;
	//fp << "$$" << endl;
	//print_integer_matrix_with_standard_labels(fp, Decomp,
	//nb_arcs_not_on_conic, nb_surfaces, TRUE /* f_tex */);
	print_integer_matrix_tex_block_by_block(fp, Decomp,
			Six_arcs->nb_arcs_not_on_conic, nb_surfaces, 25);
	//fp << "$$" << endl;



	FREE_int(Decomp);
	FREE_int(f_deleted);
	FREE_int(Arc_identify);
	FREE_int(Arc_identify_nb);

	latex_foot(fp);
	} // fp

	cout << "Written file " << fname_arc_lifting << " of size "
			<< file_size(fname_arc_lifting) << endl;
	//delete Gen;
	//delete F;

	cout << "classify_arcs_and_do_arc_lifting done" << endl;
}


