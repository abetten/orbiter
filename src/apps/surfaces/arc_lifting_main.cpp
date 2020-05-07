// arc_lifting_main.cpp
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

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


int t0 = 0;


int main(int argc, const char **argv);
void lift_single_arc(long int *arc, int arc_size,
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
	os_interface Os;

	t0 = Os.os_ticks();


	surface_domain *Surf;
	surface_with_action *Surf_A;
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	Surf = NEW_OBJECT(surface_domain);
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
	Surf->init(F, verbose_level - 5);
	if (f_v) {
		cout << "after Surf->init" << endl;
	}

	int f_semilinear;
	number_theory_domain NT;

	if (NT.is_prime(q)) {
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
		cout << "before Surf_A->Classify_trihedral_pairs->classify" << endl;
	}
	Surf_A->Classify_trihedral_pairs->classify(0 /*verbose_level*/);
	if (f_v) {
		cout << "after Surf_A->Classify_trihedral_pairs->classify" << endl;
	}


	

	if (f_arc) {
		long int *arc;
		int arc_size;

		lint_vec_scan(the_arc_text, arc, arc_size);
		
		if (f_v) {
			cout << "before lift_single_arc" << endl;
			}
		lift_single_arc(arc, arc_size, Surf_A, verbose_level);
		if (f_v) {
			cout << "after lift_single_arc" << endl;
		}
	}
	

	the_end(t0);
	//the_end_quietly(t0);
	
}


void lift_single_arc(long int *arc, int arc_size,
		surface_with_action *Surf_A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q;
	surface_domain *Surf;
	finite_field *F;
	char fname_arc_lifting[1000];

	F = Surf_A->F;
	q = F->q;
	Surf = Surf_A->Surf;


	if (f_v) {
		cout << "lift_single_arc q=" << q << endl;
		cout << "Lifting the arc ";
		lint_vec_print(cout, arc, arc_size);
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
		sprintf(fname_arc_lifting + strlen(fname_arc_lifting), "_%ld",
				arc[i]);
		}
	sprintf(fname_arc_lifting + strlen(fname_arc_lifting), ".tex");
	ofstream fp(fname_arc_lifting);
	latex_interface L;


	L.head(fp,
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
	lint_vec_print(fp, arc, arc_size);
	fp << endl;

	fp << "consisting of the following points:\\\\" << endl;
	F->display_table_of_projective_points(fp, arc, 6, 3);
	



	cout << "classify_arcs_and_do_arc_lifting before Surf_A->list_orbits_on_trihedra_type1" << endl;
	Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type1(fp);

	cout << "classify_arcs_and_do_arc_lifting before Surf_A->list_orbits_on_trihedra_type2" << endl;
	Surf_A->Classify_trihedral_pairs->list_orbits_on_trihedra_type2(fp);

	cout << "classify_arcs_and_do_arc_lifting before Surf_A->print_trihedral_pairs no stabs" << endl;
	Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
			FALSE /* f_with_stabilizers */);

	cout << "classify_arcs_and_do_arc_lifting before Surf_A->print_trihedral_pairs with stabs" << endl;
	Surf_A->Classify_trihedral_pairs->print_trihedral_pairs(fp,
			TRUE /* f_with_stabilizers */);



	arc_lifting *AL;

	AL = NEW_OBJECT(arc_lifting);


	if (f_v) {
		cout << "lift_single_arc before AL->create_surface_and_group" << endl;
		}
	AL->create_surface_and_group(Surf_A, arc, verbose_level);
	if (f_v) {
		cout << "lift_single_arc after AL->create_surface_and_group" << endl;
		}

	AL->print(fp);


	FREE_OBJECT(AL);
	
	L.foot(fp);
	} // fp
	file_io Fio;

	cout << "Written file " << fname_arc_lifting << " of size "
			<< Fio.file_size(fname_arc_lifting) << endl;

	if (f_v) {
		cout << "lift_single_arc done" << endl;
		}
}

