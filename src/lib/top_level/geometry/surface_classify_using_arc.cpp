/*
 * surface_classify_using_arc.cpp
 *
 *  Created on: Jul 16, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


surface_classify_using_arc::surface_classify_using_arc()
{
	Surf_A = NULL;

	A = NULL;
	nice_gens = NULL;

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
	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}
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


void surface_classify_using_arc::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_using_arc::report" << endl;
	}

	surface_domain *Surf;
	finite_field *F;

	char fname_arc_lifting[1000];


	F = Surf_A->F;
	Surf = Surf_A->Surf;


	{
		char title[1000];
		char author[1000];
		snprintf(title, 1000, "Arc lifting over GF(%d) ", F->q);
		strcpy(author, "");

		snprintf(fname_arc_lifting, 1000, "arc_lifting_q%d.tex", F->q);
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


		if (f_v) {
			cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs q=" << F->q << endl;
		}



		fp << "\\section*{Basics}" << endl << endl;

		if (f_v) {
			cout << "surface_classify_using_arc::report before Surf_A->report_basics_and_trihedral_pair" << endl;
		}
		if (f_v) {
			Surf_A->report_basics_and_trihedral_pair(fp);
		}
		if (f_v) {
			cout << "surface_classify_using_arc::report after Surf_A->report_basics_and_trihedral_pair" << endl;
		}

		fp << "\\section*{Six-Arcs}" << endl << endl;


		if (f_v) {
			cout << "surface_classify_using_arc::report before Six_arcs->report_latex" << endl;
		}
		Six_arcs->report_latex(fp);
		if (f_v) {
			cout << "surface_classify_using_arc::report after Six_arcs->report_latex" << endl;
		}


		char fname_base[1000];
		snprintf(fname_base, 1000, "arcs_q%d", F->q);

		if (F->q < 20) {
			cout << "before Gen->gen->draw_poset_full" << endl;
			Six_arcs->Gen->gen->draw_poset(
				fname_base,
				6 /* depth */, 0 /* data */,
				TRUE /* f_embedded */,
				FALSE /* f_sideways */,
				100 /* rad */,
				verbose_level);
		}


		fp << "\\section*{List of Surfaces}" << endl << endl;



		int surface_idx;

		for (surface_idx = 0;
				surface_idx < nb_surfaces;
				surface_idx++) {



			fp << "\\section*{Surface $" << surface_idx << "$ of " << nb_surfaces << "}" << endl << endl;


			if (f_v) {
				cout << "surface_classify_using_arc::report before SCAL[" << surface_idx << "].report" << endl;
			}
			SCAL[surface_idx].report(fp, verbose_level);
			if (f_v) {
				cout << "surface_classify_using_arc::report after SCAL[" << surface_idx << "].report" << endl;
			}


			fp << "The following " << Arc_identify_nb[surface_idx]
				<< " arcs are involved with surface " <<   nb_surfaces << ": $";
			int_vec_print(fp,
				Arc_identify + surface_idx * Six_arcs->nb_arcs_not_on_conic,
				Arc_identify_nb[surface_idx]);
			fp << "$\\\\" << endl;



		}


		fp << "\\section*{Decomposition Matrix Surfaces vs Arcs}" << endl << endl;



		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"decomposition matrix:" << endl;
		cout << "$$" << endl;
		L.print_integer_matrix_with_standard_labels(cout, Decomp,
				Six_arcs->nb_arcs_not_on_conic, nb_surfaces,
				TRUE /* f_tex */);
		cout << "$$" << endl;

		fp << "Decomposition matrix:" << endl;
		//fp << "$$" << endl;
		//print_integer_matrix_with_standard_labels(fp, Decomp,
		//nb_arcs_not_on_conic, nb_surfaces, TRUE /* f_tex */);
		L.print_integer_matrix_tex_block_by_block(fp, Decomp,
				Six_arcs->nb_arcs_not_on_conic, nb_surfaces, 25);
		//fp << "$$" << endl;

		file_io Fio;
		char fname_decomposition[1000];

		sprintf(fname_decomposition, "surfaces_q%d_decomposition_matrix.csv", F->q);

		Fio.int_matrix_write_csv(fname_decomposition, Decomp,
				Six_arcs->nb_arcs_not_on_conic, nb_surfaces);
		cout << "Written file " << fname_decomposition << " of size "
				<< Fio.file_size(fname_decomposition) << endl;




		L.foot(fp);

	} // fp

	file_io Fio;

	cout << "Written file " << fname_arc_lifting << " of size "
			<< Fio.file_size(fname_arc_lifting) << endl;


	if (f_v) {
		cout << "surface_classify_using_arc::report done" << endl;
	}
}

void surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs(
		poset_classification_control *Control_six_arcs,
		surface_with_action *Surf_A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	surface_domain *Surf;
	finite_field *F;
	int i, j, arc_idx;
	number_theory_domain NT;

	int f_semilinear = TRUE;

	surface_classify_using_arc::Surf_A = Surf_A;


	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs" << endl;
	}

	F = Surf_A->F;
	Surf = Surf_A->Surf;

	A = NEW_OBJECT(action);


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}


	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"before A->init_projective_group" << endl;
	}
	A->init_projective_group(3, F,
			f_semilinear,
			TRUE /*f_basis*/, TRUE /* f_init_sims */,
			nice_gens,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"after A->init_projective_group" << endl;
	}



	Descr = NEW_OBJECT(arc_generator_description);
	Descr->F = F;
	Descr->f_q = TRUE;
	Descr->q = F->q;
	Descr->f_n = TRUE;
	Descr->n = 3;
	Descr->f_d = TRUE;
	Descr->d = 2;
	Descr->f_target_size = TRUE;
	Descr->target_size = 6;
	Descr->Control = Control_six_arcs;



	// classify six arcs not on a conic:

	if (f_v) {
		cout << "surface_classify_using_arc::classify_surfaces_through_arcs_and_trihedral_pairs "
				"before Six_arcs->init" << endl;
	}
	Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);
	Six_arcs->init(
			Descr,
			A,
			Surf->P2,
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

	int_vec_zero(f_deleted, Six_arcs->nb_arcs_not_on_conic);
	int_vec_zero(Arc_identify_nb, Six_arcs->nb_arcs_not_on_conic);

	transporter = NEW_int(Surf_A->A->elt_size_in_int);



	SCAL = NEW_OBJECTS(surface_create_by_arc_lifting, Six_arcs->nb_arcs_not_on_conic);


	for (arc_idx = 0;
			arc_idx < Six_arcs->nb_arcs_not_on_conic;
			arc_idx++) {


		if (f_deleted[arc_idx]) {
			continue;
		}

		SCAL->init(arc_idx, this, verbose_level);

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
	int_vec_zero(Decomp, Six_arcs->nb_arcs_not_on_conic * nb_surfaces);
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





}}

