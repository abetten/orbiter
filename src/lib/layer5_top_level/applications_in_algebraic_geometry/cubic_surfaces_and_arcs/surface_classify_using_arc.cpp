/*
 * surface_classify_using_arc.cpp
 *
 *  Created on: Jul 16, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_arcs {



surface_classify_using_arc::surface_classify_using_arc()
{
	Record_birth();
	PA = NULL;
	Surf_A = NULL;

	//A = NULL;
	//nice_gens = NULL;

	Six_arcs = NULL;
	Descr = NULL;

	transporter = NULL;


	nb_surfaces = 0;
	SCAL = NULL;

	Arc_identify_nb = 0;
	Arc_identify = NULL;
	f_deleted = NULL;
	Decomp = NULL;

}



surface_classify_using_arc::~surface_classify_using_arc()
{
	Record_death();
#if 0
	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}
#endif
	if (transporter) {
		FREE_int(transporter);
	}
	if (SCAL) {
		FREE_OBJECTS(SCAL);
	}

	if (Decomp) {
		FREE_int(Decomp);
	}
	if (f_deleted) {
		FREE_int(f_deleted);
	}
	if (Arc_identify) {
		FREE_int(Arc_identify);
	}
	if (Arc_identify_nb) {
		FREE_int(Arc_identify_nb);
	}

}


void surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs(
		std::string &Control_six_arcs_label,
		projective_geometry::projective_space_with_action *PA,
		int f_test_nb_Eckardt_points, int nb_E,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, arc_idx;
	algebra::number_theory::number_theory_domain NT;




	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	surface_classify_using_arc::PA = PA;
	surface_classify_using_arc::Surf_A = PA->Surf_A;



	Descr = NEW_OBJECT(apps_geometry::arc_generator_description);
	Descr->f_d = true;
	Descr->d = 2;
	Descr->f_target_size = true;
	Descr->target_size = 6;
	Descr->f_control = true;
	Descr->control_label.assign(Control_six_arcs_label);



	// classify six arcs not on a conic:

	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Six_arcs->init" << endl;
	}
	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);
	Six_arcs->init(
			Descr,
			Surf_A->PA,
			f_test_nb_Eckardt_points, nb_E,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"after Six_arcs->init" << endl;
	}



	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"before extending the arcs" << endl;
	}


	nb_surfaces = 0;

	f_deleted = NEW_int(Six_arcs->nb_arcs_not_on_conic);
	Arc_identify = NEW_int(Six_arcs->nb_arcs_not_on_conic *
			Six_arcs->nb_arcs_not_on_conic);
	Arc_identify_nb = NEW_int(Six_arcs->nb_arcs_not_on_conic);

	Int_vec_zero(f_deleted, Six_arcs->nb_arcs_not_on_conic);
	Int_vec_zero(Arc_identify_nb, Six_arcs->nb_arcs_not_on_conic);

	transporter = NEW_int(Surf_A->A->elt_size_in_int);



	SCAL = NEW_OBJECTS(surface_create_by_arc_lifting, Six_arcs->nb_arcs_not_on_conic);


	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {


		if (f_deleted[arc_idx]) {
			continue;
		}

		if (f_v) {
			cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
					"before SCAL[nb_surfaces].init, nb_surfaces = " << nb_surfaces << endl;
		}

		SCAL[nb_surfaces].init(arc_idx, this, verbose_level);

		if (f_v) {
			cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
					"after SCAL[nb_surfaces].init nb_surfaces = " << nb_surfaces << endl;
		}

		nb_surfaces++;

	} // next arc_idx

	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"after extending the arcs" << endl;

		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"We found " << nb_surfaces << " surfaces" << endl;


		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"decomposition matrix:" << endl;
		for (i = 0; i < nb_surfaces; i++) {
			for (j = 0; j < Arc_identify_nb[i]; j++) {
				cout << Arc_identify[i * Six_arcs->nb_arcs_not_on_conic + j];
				if (j < Arc_identify_nb[i] - 1) {
					cout << ", ";
				}
			}
			cout << endl;
		}
	}

	int a;

	Decomp = NEW_int(Six_arcs->nb_arcs_not_on_conic * nb_surfaces);
	Int_vec_zero(Decomp, Six_arcs->nb_arcs_not_on_conic * nb_surfaces);
	for (i = 0; i < nb_surfaces; i++) {
		for (j = 0; j < Arc_identify_nb[i]; j++) {
			a = Arc_identify[i * Six_arcs->nb_arcs_not_on_conic + j];
			Decomp[a * nb_surfaces + i]++;
		}
	}



	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs done" << endl;
	}
}


void surface_classify_using_arc::report(
		other::graphics::layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_using_arc::report" << endl;
	}

	algebra::field_theory::finite_field *F;



	F = Surf_A->PA->F;

	string fname_arc_lifting;

	fname_arc_lifting = "arc_lifting_q" + std::to_string(F->q) + ".tex";

	{
		string title, author, extra_praeamble;

		title = "Arc lifting over GF(" + std::to_string(F->q) + ") ";



		ofstream fp(fname_arc_lifting);
		other::l1_interfaces::latex_interface L;


		L.head(fp,
			false /* f_book */,
			true /* f_title */,
			title, author,
			false /*f_toc */,
			false /* f_landscape */,
			false /* f_12pt */,
			true /*f_enlarged_page */,
			true /* f_pagenumbers*/,
			extra_praeamble /* extra_praeamble */);


		report2(fp, Opt, verbose_level);


		L.foot(fp);

	} // fp

	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_arc_lifting << " of size "
			<< Fio.file_size(fname_arc_lifting) << endl;


	if (f_v) {
		cout << "surface_classify_using_arc::report done" << endl;
	}
}


void surface_classify_using_arc::report2(
		std::ostream &ost,
		other::graphics::layered_graph_draw_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_using_arc::report2" << endl;
	}

	//surface_domain *Surf;
	algebra::field_theory::finite_field *F;

	F = Surf_A->PA->F;
	//Surf = Surf_A->Surf;

	if (f_v) {
		cout << "surface_classify_using_arc::report2 q=" << F->q << endl;
	}



	report_decomposition_matrix(ost, verbose_level);


	int surface_idx;

	for (surface_idx = 0;
			surface_idx < nb_surfaces;
			surface_idx++) {



		ost << "Surface $" << surface_idx << "$ "
				"is associated with the following arcs: $";
		Int_vec_print(ost,
			Arc_identify + surface_idx * Six_arcs->nb_arcs_not_on_conic,
			Arc_identify_nb[surface_idx]);
		ost << "$\\\\" << endl;



	}


	//fp << "\\clearpage" << endl << endl;



	ost << "\\section*{Six-Arcs}" << endl << endl;


	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"before Six_arcs->report_latex" << endl;
	}
	Six_arcs->report_latex(ost);
	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"after Six_arcs->report_latex" << endl;
	}

#if 0
	fname_base = "arcs_q" + std::to_string(F->q);

	if (F->q < 20) {
		cout << "before Gen->gen->draw_poset_full" << endl;
		Six_arcs->Gen->gen->draw_poset(
			fname_base,
			6 /* depth */, 0 /* data */,
			true /* f_embedded */,
			false /* f_sideways */,
			100 /* rad */,
			verbose_level);
	}
#endif


	ost << "\\section*{Double Triplets}" << endl << endl;

	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"before Surf_A->report_double_triplets" << endl;
	}
	if (f_v) {
		Surf_A->report_double_triplets(ost);
	}
	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"after Surf_A->report_double_triplets" << endl;
	}




	ost << "\\section*{Summary of Surfaces}" << endl << endl;




	for (surface_idx = 0;
			surface_idx < nb_surfaces;
			surface_idx++) {



		ost << "\\subsection*{Surface $" << surface_idx << "$ "
				"of " << nb_surfaces << "}" << endl << endl;


		if (f_v) {
			cout << "surface_classify_using_arc::report2 "
					"before SCAL[" << surface_idx << "].report_summary" << endl;
		}
		SCAL[surface_idx].report_summary(ost, verbose_level);
		if (f_v) {
			cout << "surface_classify_using_arc::report2 "
					"after SCAL[" << surface_idx << "].report_summary" << endl;
		}


		ost << "The following " << Arc_identify_nb[surface_idx]
			<< " arcs are involved with surface " <<   nb_surfaces << ": $";
		Int_vec_print(ost,
			Arc_identify + surface_idx * Six_arcs->nb_arcs_not_on_conic,
			Arc_identify_nb[surface_idx]);
		ost << "$\\\\" << endl;



	}



	ost << "\\section*{List of Surfaces}" << endl << endl;




	for (surface_idx = 0;
			surface_idx < nb_surfaces;
			surface_idx++) {



		ost << "\\subsection*{Surface $" << surface_idx << "$ "
				"of " << nb_surfaces << "}" << endl << endl;


		if (f_v) {
			cout << "surface_classify_using_arc::report2 "
					"before SCAL[" << surface_idx << "].report" << endl;
		}
		SCAL[surface_idx].report(ost, Opt, verbose_level);
		if (f_v) {
			cout << "surface_classify_using_arc::report2 "
					"after SCAL[" << surface_idx << "].report" << endl;
		}


		ost << "The following " << Arc_identify_nb[surface_idx]
			<< " arcs are involved with surface " <<   nb_surfaces << ": $";
		Int_vec_print(ost,
			Arc_identify + surface_idx * Six_arcs->nb_arcs_not_on_conic,
			Arc_identify_nb[surface_idx]);
		ost << "$\\\\" << endl;



	}


	ost << "\\section*{Double Triplets: Details}" << endl << endl;
	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"before Surf_A->report_double_triplets_detailed" << endl;
	}
	if (f_v) {
		Surf_A->report_double_triplets_detailed(ost);
	}
	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"after Surf_A->report_double_triplets_detailed" << endl;
	}


	ost << "\\section*{Basics}" << endl << endl;

	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"before Surf_A->report_basics_and_trihedral_pair" << endl;
	}
	if (f_v) {
		Surf_A->report_basics(ost);
	}
	if (f_v) {
		cout << "surface_classify_using_arc::report2 "
				"after Surf_A->report_basics_and_trihedral_pair" << endl;
	}

}


void surface_classify_using_arc::report_decomposition_matrix(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "surface_classify_using_arc::report_decomposition_matrix" << endl;
	}

	ost << "\\section*{Decomposition Matrix Arcs vs Surfaces}" << endl << endl;



	cout << "surface_classify_using_arc::report_decomposition_matrix "
			"decomposition matrix:" << endl;
	cout << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(cout, Decomp,
			Six_arcs->nb_arcs_not_on_conic, nb_surfaces,
			true /* f_tex */);
	cout << "$$" << endl;

	ost << "Decomposition matrix:" << endl;
	//fp << "$$" << endl;
	//print_integer_matrix_with_standard_labels(fp, Decomp,
	//nb_arcs_not_on_conic, nb_surfaces, true /* f_tex */);
	L.print_integer_matrix_tex_block_by_block(ost, Decomp,
			Six_arcs->nb_arcs_not_on_conic, nb_surfaces, 25);
	//fp << "$$" << endl;

	other::orbiter_kernel_system::file_io Fio;
	string fname_decomposition;

	fname_decomposition = "surfaces_q" + std::to_string(Surf_A->PA->F->q) + "_decomposition_matrix.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_decomposition, Decomp,
			Six_arcs->nb_arcs_not_on_conic, nb_surfaces);

	cout << "Written file " << fname_decomposition << " of size "
			<< Fio.file_size(fname_decomposition) << endl;

	if (f_v) {
		cout << "surface_classify_using_arc::report_decomposition_matrix done" << endl;
	}

}

}}}}


