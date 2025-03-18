/*
 * orbits_on_polynomials.cpp
 *
 *  Created on: Nov 28, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {


orbits_on_polynomials::orbits_on_polynomials()
{
	Record_birth();
	LG = NULL;
	degree_of_poly = 0;

	F = NULL;
	A = NULL;
	n = 0;
	// go;
	HPD = NULL;
	A2 = NULL;
	Elt1 = Elt2 = Elt3 = NULL;
	f_has_Sch = false;
	Sch = NULL;
	// full_go

	f_has_Orb = false;
	Orb = NULL;

	//fname_base
	//fname_csv
	//fname_report
	T = NULL;
	Nb_pts = NULL;

}

orbits_on_polynomials::~orbits_on_polynomials()
{
	Record_death();
}

void orbits_on_polynomials::init(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;



	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}


	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";


	//Sch = new schreier;
	//A2->all_point_orbits(*Sch, verbose_level);


	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"before A->Strong_gens->compute_all_point_orbits_schreier" << endl;
	}

	f_has_Sch = true;

	int print_interval = 10000;

	Sch = A->Strong_gens->compute_all_point_orbits_schreier(
			A2, print_interval, verbose_level - 2);

	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"after A->Strong_gens->compute_all_point_orbits_schreier" << endl;
	}



	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"before Sch->write_orbit_summary" << endl;
	}
	Sch->write_orbit_summary(
			fname_csv,
			A /*default_action*/,
			go,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"after Sch->write_orbit_summary" << endl;
	}



	A->group_order(full_go);
	T = NEW_OBJECT(data_structures_groups::orbit_transversal);

	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"before T->init_from_schreier" << endl;
	}

	T->init_from_schreier(
			Sch,
			A,
			full_go,
			verbose_level);

	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"after T->init_from_schreier" << endl;
	}



	Sch->print_orbit_reps(cout);


	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"before compute_points" << endl;
	}
	compute_points(verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"after compute_points" << endl;
	}


#if 0
	if (f_recognize) {
		long int *Rank;
		int len;
		int i;

		int *Idx;


		cout << "orbits_on_polynomials::init recognition:" << endl;
		Lint_vec_scan(recognize_text, Rank, len);

		Idx = NEW_int(len);

		for (i = 0; i < len; i++) {
			//cout << "recognizing object " << i << " / " << len << " which is " << Rank[i] << endl;
			int orbit_idx;
			orbit_idx = Sch->orbit_number(Rank[i]);
			Idx[i] = orbit_idx;
			cout << "recognizing object " << i << " / " << len << ", point "
					<< Rank[i] << " lies in orbit " << orbit_idx << endl;
		}
		orbiter_kernel_system::file_io Fio;
		std::string fname;

		fname = fname_base + "_recognition.csv";

		string label;

		label = "Idx";
		Fio.int_vec_write_csv(Idx, len, fname, label);

		FREE_lint(Rank);

	}
#endif



	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);



	if (f_v) {
		cout << "orbits_on_polynomials::init done" << endl;
	}
}


void orbits_on_polynomials::orbit_of_one_polynomial(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		algebra::expression_parser::symbolic_object_builder *Symbol,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::HPD = HPD;



	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	if (f_v) {
		cout << "n = " << n << endl;
	}


	degree_of_poly = HPD->degree;
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}


	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level - 2);

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	fname_base =  "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)
			+ "_q" + std::to_string(F->q);
	fname_csv = fname_base + ".csv";


	//Sch = new schreier;
	//A2->all_point_orbits(*Sch, verbose_level);



	if (Symbol->Formula_vector->len != 1) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial len != 1" << endl;
		exit(1);
	}


	int *eqn;
	int eqn_sz;

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"before get_multipoly" << endl;
	}
	Symbol->Formula_vector->V[0].get_multipoly(HPD,
			eqn, eqn_sz, verbose_level - 1);

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"after get_multipoly" << endl;
	}
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"eqn = ";
		Int_vec_print(cout, eqn, eqn_sz);
		cout << endl;
	}




	// compute the orbit of the equation under the stabilizer of the set of points:


	f_has_Orb = true;

	Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);

	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"before Orb->init" << endl;
	}
	Orb->init(
			A, F,
			A2->G.OnHP,
		LG->Strong_gens /* A->Strong_gens*/, eqn,
		verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"after Orb->init" << endl;
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"found an orbit of length " << Orb->used_length << endl;
	}

#if 0
	int *transporter;
	int Idx[]= {0,648,6480,12312,64800};
	int i;
	int Nb;

	Nb = sizeof(Idx) / sizeof(int);

	transporter = NEW_int(A->elt_size_in_int);

	for (i = 0; i < Nb; i++) {

		if (f_v) {
			cout << "orbits_on_polynomials::orbit_of_one_polynomial "
					"before Orb->get_transporter" << endl;

		}
		Orb->get_transporter(
				Idx[i],
				transporter, verbose_level);

		if (f_v) {
			cout << "orbits_on_polynomials::orbit_of_one_polynomial "
					"after Orb->get_transporter" << endl;

		}

		cout << "i=" << i << " / " << Nb << " Idx[i] = " << Idx[i] << "transporter=" << endl;
		A->Group_element->element_print(transporter, cout);
		cout << endl;
	}
#endif

#if 0
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"before Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic = Orb->stabilizer_orbit_rep(
			pt_stab_order, verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial "
				"after Orb->stabilizer_orbit_rep" << endl;
	}
	Stab_gens_quartic->print_generators_tex(cout);

	FREE_OBJECT(SG_pt_stab);
	FREE_OBJECT(Orb);
	FREE_OBJECT(AonHPD);
#endif




#if 0
	if (f_recognize) {
		long int *Rank;
		int len;
		int i;

		int *Idx;


		cout << "orbits_on_polynomials::orbit_of_one_polynomial recognition:" << endl;
		Lint_vec_scan(recognize_text, Rank, len);

		Idx = NEW_int(len);

		for (i = 0; i < len; i++) {
			//cout << "recognizing object " << i << " / " << len << " which is " << Rank[i] << endl;
			int orbit_idx;
			orbit_idx = Sch->orbit_number(Rank[i]);
			Idx[i] = orbit_idx;
			cout << "recognizing object " << i << " / " << len << ", point "
					<< Rank[i] << " lies in orbit " << orbit_idx << endl;
		}
		orbiter_kernel_system::file_io Fio;
		std::string fname;

		fname = fname_base + "_recognition.csv";

		string label;

		label = "Idx";
		Fio.int_vec_write_csv(Idx, len, fname, label);

		FREE_lint(Rank);

	}
#endif



	//FREE_int(Elt1);
	//FREE_int(Elt2);
	//FREE_int(Elt3);



	if (f_v) {
		cout << "orbits_on_polynomials::orbit_of_one_polynomial done" << endl;
	}
}


void orbits_on_polynomials::compute_points(
		int verbose_level)
{
	int *coeff;
	int i;

	coeff = NEW_int(HPD->get_nb_monomials());
	Nb_pts = NEW_int(T->nb_orbits);


	for (i = 0; i < T->nb_orbits; i++) {

		algebra::ring_theory::longinteger_object go;
		T->Reps[i].Strong_gens->group_order(go);

		cout << i << " : ";
		Lint_vec_print(cout, T->Reps[i].data, T->Reps[i].sz);
		cout << " : ";
		cout << go;

		cout << " : ";

		HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

		std::vector<long int> Pts;

		HPD->enumerate_points(coeff, Pts, verbose_level);

		Points.push_back(Pts);
		Nb_pts[i] = Pts.size();
	}
	FREE_int(coeff);

}

void orbits_on_polynomials::report(
		int verbose_level)
// used to create a projective_geometry::projective_space_with_action
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::report" << endl;
	}
	cout << "orbit reps:" << endl;

	string title, author, extra_praeamble;

	fname_report = "poly_orbits_d" + std::to_string(degree_of_poly)
			+ "_n" + std::to_string(n - 1)+ "_q" + std::to_string(F->q) + ".tex";

	title = "Varieties of degree " + std::to_string(degree_of_poly)
			+ " in PG(" + std::to_string(n - 1)+ "," + std::to_string(F->q) + ")";

	author.assign("Orbiter");

	{
		ofstream ost(fname_report);
		other::l1_interfaces::latex_interface L;

		L.head(ost,
				false /* f_book*/,
				true /* f_title */,
				title, author,
				false /* f_toc */,
				false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);

		ost << "\\small" << endl;
		ost << "\\arraycolsep=2pt" << endl;
		ost << "\\parindent=0pt" << endl;
		ost << "$q = " << F->q << "$\\\\" << endl;
		ost << "$n = " << n - 1 << "$\\\\" << endl;
		ost << "degree of poly $ = " << degree_of_poly << "$\\\\" << endl;

		ost << "\\clearpage" << endl << endl;


		// summary table:

		ost << "\\section*{The Varieties of degree $" << degree_of_poly
				<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, summary}" << endl;

#if 0
		T->print_table_latex(
				f,
				true /* f_has_callback */,
				polynomial_orbits_callback_print_function2,
				HPD /* callback_data */,
				true /* f_has_callback */,
				polynomial_orbits_callback_print_function,
				HPD /* callback_data */,
				verbose_level);
#else
		int *coeff;
		int i;

		coeff = NEW_int(HPD->get_nb_monomials());
		//Nb_pts = NEW_int(T->nb_orbits);


#if 0
		// compute the group of the surface:
		projective_geometry::projective_space_with_action *PA;
		int f_semilinear;
		number_theory::number_theory_domain NT;

		if (NT.is_prime(F->q)) {
			f_semilinear = false;
		}
		else {
			f_semilinear = true;
		}

		PA = NEW_OBJECT(projective_geometry::projective_space_with_action);

		if (f_v) {
			cout << "group_theoretic_activity::do_cubic_surface_properties before PA->init" << endl;
		}
		PA->init(
			F, n - 1 /*n*/, f_semilinear,
			true /* f_init_incidence_structure */,
			verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_cubic_surface_properties after PA->init" << endl;
		}
#endif






		other::data_structures::tally T_nb_pts;
		int h, j, f, l, a;

		T_nb_pts.init(Nb_pts, T->nb_orbits, false, 0);

		for (h = T_nb_pts.nb_types - 1; h >= 0; h--) {

			f = T_nb_pts.type_first[h];
			l = T_nb_pts.type_len[h];
			a = T_nb_pts.data_sorted[f];

			ost << "\\subsection*{Objects with " << a << " Points}" << endl;

			ost << "There are " << l << " objects with " << a << " Points: \\\\" << endl;

			int *Idx;
			int len;

			T_nb_pts.get_class_by_value(Idx, len, a, 0 /*verbose_level*/);


			other::data_structures::sorting Sorting;

			Sorting.int_vec_heapsort(Idx, l);

			ost << "orbit : rep : go : poly : Pts\\\\" << endl;
			for (j = 0; j < l; j++) {

				//i = T_nb_pts.sorting_perm_inv[f + j];

				i = Idx[j];

				algebra::ring_theory::longinteger_object go;
				T->Reps[i].Strong_gens->group_order(go);

				ost << i << " : ";
				Lint_vec_print(ost, T->Reps[i].data, T->Reps[i].sz);
				ost << " : ";
				ost << go;

				ost << " : ";

				HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

				int nb_pts;

				nb_pts = Nb_pts[i];

				//ost << nb_pts;
				//ost << " : ";

				ost << T->Reps[i].data[0] << "=$";
				HPD->print_equation_tex(ost, coeff);
				//int_vec_print(f, coeff, HPD->get_nb_monomials());
				//cout << " = ";
				//HPD->print_equation_str(ost, coeff);

				//f << " & ";
				//Reps[i].Strong_gens->print_generators_tex(f);
				ost << "$";

				ost << " : ";

				int u;
				long int *set;
				//groups::strong_generators *Sg;

				set = NEW_lint(nb_pts);
				for (u = 0; u < nb_pts; u++) {
					set[u] = Points[i][u];
				}

				for (u = 0; u < nb_pts; u++) {
					ost << set[u];
					if (u < nb_pts - 1) {
						ost << ",";
					}
				}

#if 0
				PA->compute_group_of_set(set, nb_pts,
						Sg,
						verbose_level);

				ost << " : go=";
				ring_theory::longinteger_object go1;
				Sg->group_order(go1);
				ost << go1;
#endif

				ost << "\\\\" << endl;

				FREE_lint(set);
			}

			FREE_int(Idx);

		}
		//FREE_OBJECT(PA);

#endif

		FREE_int(coeff);



		L.foot(ost);

	}
	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_report << " of size " << Fio.file_size(fname_report) << endl;
	if (f_v) {
		cout << "orbits_on_polynomials::report done" << endl;
	}

}

void orbits_on_polynomials::report_detailed_list(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::report_detailed_list" << endl;
	}
	// detailed listing:



	other::data_structures::tally T1;

	T1.init(Nb_pts, T->nb_orbits, false, 0);
	ost << "Distribution of the number of points: $";
	T1.print_bare_tex(ost, true);
	ost << "$\\\\" << endl;

#if 0
	ost << "\\section{The Varieties of degree $" << degree_of_poly
			<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, "
					"detailed listing}" << endl;
	{
		int fst, l, a, r;
		ring_theory::longinteger_object go, go1;
		ring_theory::longinteger_domain D;
		int *coeff;
		int *line_type;
		long int *Pts;
		int nb_pts;
		int *Kernel;
		int *v;
		int i;
		//int h, pt, orbit_idx;

		A->group_order(go);
		Pts = NEW_lint(P->Subspaces->N_points);
		coeff = NEW_int(HPD->get_nb_monomials());
		line_type = NEW_int(P->Subspaces->N_lines);
		Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
		v = NEW_int(n);

		for (i = 0; i < Sch->nb_orbits; i++) {
			ost << "\\subsection*{Orbit " << i << " / "
					<< Sch->nb_orbits << "}" << endl;
			fst = Sch->orbit_first[i];
			l = Sch->orbit_len[i];

			D.integral_division_by_int(go, l, go1, r);
			a = Sch->orbit[fst];
			HPD->unrank_coeff_vector(coeff, a);


			vector<long int> Points;

			HPD->enumerate_points(coeff, Points, verbose_level);

			nb_pts = Points.size();
			Pts = NEW_lint(nb_pts);
			for (int u = 0; u < nb_pts; u++) {
				Pts[u] = Points[u];
			}

			ost << "stab order " << go1 << "\\\\" << endl;
			ost << "orbit length = " << l << "\\\\" << endl;
			ost << "orbit rep = " << a << "\\\\" << endl;
			ost << "number of points = " << nb_pts << "\\\\" << endl;

			ost << "$";
			Int_vec_print(ost, coeff, HPD->get_nb_monomials());
			ost << " = ";
			HPD->print_equation(ost, coeff);
			ost << "$\\\\" << endl;


			cout << "We found " << nb_pts << " points in the variety" << endl;
			cout << "They are : ";
			Lint_vec_print(cout, Pts, nb_pts);
			cout << endl;
			P->Reporting->print_set_numerical(cout, Pts, nb_pts);

			F->Io->display_table_of_projective_points(
					ost, Pts, nb_pts, n);

			P->Subspaces->line_intersection_type(Pts, nb_pts,
					line_type, 0 /* verbose_level */);

			ost << "The line type is: ";

			stringstream sstr;
			Int_vec_print_classified_str(sstr,
					line_type, P->Subspaces->N_lines,
					true /* f_backwards*/);
			string s = sstr.str();
			ost << "$" << s << "$\\\\" << endl;
			//int_vec_print_classified(line_type, HPD->P->N_lines);
			//cout << "after int_vec_print_classified" << endl;

			ost << "The stabilizer is generated by:" << endl;
			T->Reps[i].Strong_gens->print_generators_tex(ost);
		} // next i

		FREE_lint(Pts);
		FREE_int(coeff);
		FREE_int(line_type);
		FREE_int(Kernel);
		FREE_int(v);
		}
#endif

	if (f_v) {
		cout << "orbits_on_polynomials::report_detailed_list done" << endl;
	}
}


void orbits_on_polynomials::export_something(
		std::string &what, int data1,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::export_something" << endl;
		cout << "orbits_on_polynomials::export_something this = " << this << endl;
	}

	other::data_structures::string_tools ST;

	string fname_base;

	fname_base = "orbits_" + A2->label;

	if (f_v) {
		cout << "orbits_on_polynomials::export_something "
				"before export_something_worker" << endl;
	}
	export_something_worker(fname_base, what, data1, fname, verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::export_something "
				"after export_something_worker" << endl;
	}

	if (f_v) {
		cout << "orbits_on_polynomials::export_something done" << endl;
	}

}

void orbits_on_polynomials::export_something_worker(
		std::string &fname_base,
		std::string &what, int data1,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::export_something_worker" << endl;
	}

	other::data_structures::string_tools ST;
	other::orbiter_kernel_system::file_io Fio;


	if (ST.stringcmp(what, "orbit") == 0) {

		if (f_v) {
			cout << "orbits_on_polynomials::export_something_worker orbit" << endl;
		}

		if (f_has_Sch) {

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Sch" << endl;
			}

			fname = fname_base + "_orbit_" + std::to_string(data1) + ".csv";

			int orbit_idx = data1;
			std::vector<int> Orb;
			int *Pts;
			int i;

			Sch->get_orbit_in_order(Orb,
					orbit_idx, verbose_level);

			Pts = NEW_int(Orb.size());
			for (i = 0; i < Orb.size(); i++) {
				Pts[i] = Orb[i];
			}



			Fio.Csv_file_support->int_matrix_write_csv(
					fname, Pts, 1, Orb.size());

			FREE_int(Pts);
		}
		else if (f_has_Orb) {

			if (f_v) {
				cout << "orbits_on_polynomials::export_something_worker f_has_Orb" << endl;
			}

			fname = fname_base + "_orbit_" + std::to_string(data1) + ".csv";

			std::string *Table;
			std::string *Headings;
			int nb_rows, nb_cols;

			Orb->get_table(
					Table, Headings,
					nb_rows, nb_cols,
					verbose_level);

			Fio.Csv_file_support->write_table_of_strings_with_col_headings(
					fname,
					nb_rows, nb_cols, Table,
					Headings,
					verbose_level);
		}
		else {
			cout << "orbits_on_polynomials::export_something_worker neither f_has_Sch nor f_has_Orb" << endl;
			exit(1);
		}

		cout << "orbits_on_polynomials::export_something_worker "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	else {
		cout << "orbits_on_polynomials::export_something_worker "
				"unrecognized export target: " << what << endl;
	}

	if (f_v) {
		cout << "orbits_on_polynomials::export_something_worker done" << endl;
	}

}



}}}

