// k_arc_generator.cpp
// 
// Anton Betten
//
// May 14, 2018
//


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

k_arc_generator::k_arc_generator()
{
	null();
}

k_arc_generator::~k_arc_generator()
{
	freeself();
}

void k_arc_generator::null()
{
	F = NULL;
	P2 = NULL;
	Gen = NULL;
	d = 0;
	sz = 0;
	line_type = NULL;
	k_arc_idx = NULL;
}

void k_arc_generator::freeself()
{
	if (Gen) {
		FREE_OBJECT(Gen);
		}
	if (line_type) {
		FREE_int(line_type);
		}
	if (k_arc_idx) {
		FREE_int(k_arc_idx);
		}
	null();
}

void k_arc_generator::init(
	finite_field *F, projective_space *P2,
	int d, int sz, 
	int argc, const char **argv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "k_arc_generator::init "
				"d=" << d << " sz=" << sz << endl;
		}

	k_arc_generator::d = d;
	k_arc_generator::sz = sz;
	k_arc_generator::F = F;
	k_arc_generator::P2 = P2;
	
	line_type = NEW_int(P2->N_lines);

	sprintf(base_fname, "arcs_q%d_d%d", F->q, d);

	
	Gen = NEW_OBJECT(arc_generator);

	Gen->f_poly = FALSE;

	Gen->d = d; // we will classify d-arcs


	Gen->ECA = NEW_OBJECT(exact_cover_arguments);
	Gen->IA = NEW_OBJECT(isomorph_arguments);

	Gen->ECA->f_has_solution_prefix = TRUE;
	Gen->ECA->solution_prefix = "";

	Gen->ECA->f_has_base_fname = TRUE;
	Gen->ECA->base_fname = base_fname;
	
	if (f_v) {
		cout << "k_arc_generator::init "
				"before Gen->init" << endl;
		}
	Gen->init(F, 
		"ARCS/" /* Gen->ECA->input_prefix */,
		base_fname /* Gen->ECA->base_fname */,
		sz /* Gen->ECA->starter_size */, 
		argc, argv, 
		verbose_level - 2);
	if (f_v) {
		cout << "k_arc_generator::init "
				"after Gen->init" << endl;
		}



	//cout << "before Gen->main" << endl;
	//Gen->main(Gen->verbose_level);

	if (f_v) {
		cout << "k_arc_generator::init "
				"Classifying 6-arcs "
				"for q=" << F->q << endl;
		}
	
	Gen->compute_starter(verbose_level - 1);


	nb_orbits = Gen->gen->nb_orbits_at_level(sz);

	if (f_v) {
		cout << "k_arc_generator::init We found " << nb_orbits
				<< " isomorphism types of (" << sz << "," << d
				<< ")-arcs" << endl;
		}


	
	int *Arc;
	int h, j;
	

	nb_k_arcs = 0;

	Arc = NEW_int(sz);
	k_arc_idx = NEW_int(nb_orbits);	
	
	if (f_v) {
		cout << "k_arc_generator::init "
				"testing the arcs" << endl;
		}

	for (h = 0; h < nb_orbits; h++) {

		if (f_v) {
			cout << "k_arc_generator::init testing arc " << h
					<< " / " << nb_orbits << " : "; // << endl;
			}

		
		Gen->gen->get_set_by_level(sz, h, Arc);
		
		compute_line_type(Arc, sz, 0 /* verbose_level */);

		if (f_v) {
			classify C;

			C.init(line_type, P2->N_lines, FALSE, 0);
			C.print_naked(TRUE);
			//cout << endl;
		}

		for (j = 0; j < P2->N_lines; j++) {
			if (line_type[j] == d) {
				break;
				}
			}

		if (j < P2->N_lines) {
			k_arc_idx[nb_k_arcs++] = h;
			if (f_v) {
				cout << " is good" << endl;
			}
			}
		else {
			if (f_v) {
				cout << " is bad" << endl;
			}
		}


		}
	FREE_int(Arc);

	if (f_v) {
		cout << "We found " << nb_k_arcs << " isomorphism types of ("
				<< sz << "," << d << ")-arcs, out of a total of "
				<< nb_orbits << " isomorphism types of arcs" << endl;
		}
	



	if (f_v) {
		cout << "k_arc_generator::init done" << endl;
		}

}

void k_arc_generator::compute_line_type(int *set, int len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b;

	if (f_v) {
		cout << "k_arc_generator::compute_line_type" << endl;
		}

	if (P2->Lines_on_point == 0) {
		cout << "k_arc_generator::compute_line_type "
				"P->Lines_on_point == 0" << endl;
		exit(1);
		}
	int_vec_zero(line_type, P2->N_lines);
	for (i = 0; i < len; i++) {
		a = set[i];
		for (j = 0; j < P2->r; j++) {
			b = P2->Lines_on_point[a * P2->r + j];
			line_type[b]++;
			}
		}
	
}

}}


