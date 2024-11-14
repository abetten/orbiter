/*
 * nauty_interface_for_combo.cpp
 *
 *  Created on: Aug 24, 2024
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


nauty_interface_for_combo::nauty_interface_for_combo()
{
}

nauty_interface_for_combo::~nauty_interface_for_combo()
{
}



void nauty_interface_for_combo::run_nauty_for_combo(
		canonical_form_classification::any_combinatorial_object *Any_combo,
		int f_compute_canonical_form,
		int f_save_nauty_input_graphs,
		data_structures::bitvector *&Canonical_form,
		l1_interfaces::nauty_output *&NO,
		canonical_form_classification::encoded_combinatorial_object *&Enc,
		int verbose_level)
// called from
// nauty_interface_for_OwCF::run_nauty_basic
// classification_of_objects::process_object
// nauty_interface_with_group::set_stabilizer_of_object
// classify_using_canonical_forms::find_object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo" << endl;
	}
	//int L;
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::nauty_interface Nau;

	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo "
				"before Any_combo->encode_incma" << endl;
	}
	Any_combo->encode_incma(Enc, verbose_level - 1);
	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo "
				"after Any_combo->encode_incma" << endl;
	}
	if (verbose_level > 2) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo Incma not shown" << endl;
		//Enc->print_incma();
	}



	if (f_save_nauty_input_graphs) {

		// save Levi graph in DIMACS format:

		if (f_v) {
			cout << "nauty_interface_for_combo::run_nauty_for_combo "
					"saving Levi graph in DIMACS format" << endl;
		}


		graph_theory::colored_graph *CG;
		string graph_label;
		static int run_nauty_graph_counter = 0;


		graph_label = Any_combo->label + "_run_nauty_graph_" + std::to_string(run_nauty_graph_counter);

		run_nauty_graph_counter++;

		Enc->create_Levi_graph(CG, graph_label, verbose_level);

		//Enc->print_incma();

		if (f_v) {
			cout << "nauty_interface_for_combo::run_nauty_for_combo "
					"writing file " << graph_label << endl;
		}
		CG->save_DIMACS(graph_label, verbose_level);

		FREE_OBJECT(CG);

		if (f_v) {
			cout << "nauty_interface_for_combo::run_nauty_for_combo "
					"saving Levi graph in DIMACS format done" << endl;
		}

	}


	NO = NEW_OBJECT(l1_interfaces::nauty_output);


	//L = Enc->nb_rows * Enc->nb_cols;

	if (verbose_level > 5) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo "
				"before NO->nauty_output_allocate" << endl;
	}

	NO->nauty_output_allocate(
			Enc->canonical_labeling_len,
			Enc->invariant_set_start,
			Enc->invariant_set_size,
			verbose_level - 2);

	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo "
				"before Nau.Levi_graph" << endl;
	}
	int t0, t1, dt, tps;
	double delta_t_in_sec;
	orbiter_kernel_system::os_interface Os;

	tps = Os.os_ticks_per_second();
	t0 = Os.os_ticks();


	Nau.Levi_graph(
		Enc,
		NO,
		verbose_level);

	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo "
				"after Nau.Levi_graph" << endl;
	}

	//Int_vec_copy_to_lint(NO->Base, NO->Base_lint, NO->Base_length);

	t1 = Os.os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) dt / (double) tps;

	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo "
				"Ago=" << *NO->Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
	}
	if (verbose_level > 5) {
		int h;
		//int degree = nb_rows +  nb_cols;

		for (h = 0; h < NO->Aut_counter; h++) {
			cout << "aut generator " << h << " / " << NO->Aut_counter << " : " << endl;
			//Combi.perm_print(cout, Aut + h * degree, degree);
			cout << endl;
		}
	}




	if (f_compute_canonical_form) {

		if (f_v) {
			cout << "nauty_interface_for_combo::run_nauty_for_combo "
					"before Enc->compute_canonical_form" << endl;
		}


		Enc->compute_canonical_form(
				Canonical_form,
				NO->canonical_labeling, verbose_level);

		if (f_v) {
			cout << "nauty_interface_for_combo::run_nauty_for_combo "
					"after Enc->compute_canonical_form" << endl;
		}

	}


	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo done" << endl;
	}


}

void nauty_interface_for_combo::run_nauty_for_combo_basic(
		canonical_form_classification::any_combinatorial_object *Any_combo,
		l1_interfaces::nauty_output *&NO,
		int verbose_level)
// called from
// classify_using_canonical_forms::orderly_test
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo_basic"
				<< endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	data_structures::bitvector *Canonical_form;

	canonical_form_classification::encoded_combinatorial_object *Enc;

	int f_save_nauty_input_graphs = false;

	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo_basic "
				"before run_nauty_for_OwCF" << endl;
	}
	run_nauty_for_combo(
			Any_combo,
			false /* f_compute_canonical_form */,
			f_save_nauty_input_graphs,
			Canonical_form,
			NO,
			Enc,
			verbose_level);
	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo_basic "
				"after run_nauty_for_combo" << endl;
	}

	FREE_OBJECT(Enc);

	if (f_v) {
		cout << "nauty_interface_for_combo::run_nauty_for_combo_basic done" << endl;
	}
}

}}}


