/*
 * surface_repository.cpp
 *
 *  Created on: Feb 17, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {



surface_repository::surface_repository()
{
	Record_birth();
	Wedge = NULL;
	nb_surfaces = 0;
	SaS = NULL;
	SOA = NULL;
	Lines = NULL;
	Eqn = NULL;
}


surface_repository::~surface_repository()
{
	Record_death();
	if (SaS) {
		int i;

		for (i = 0; i < nb_surfaces; i++) {
			FREE_OBJECT(SaS[i]);
		}
		FREE_pvoid((void **) SaS);
	}
	if (SOA) {
		int i;

		for (i = 0; i < nb_surfaces; i++) {
			FREE_OBJECT(SOA[i]->SO);
			SOA[i]->SO = NULL;
			FREE_OBJECT(SOA[i]);
		}
		FREE_pvoid((void **) SOA);
	}
	if (Lines) {
		FREE_lint(Lines);
	}
	if (Eqn) {
		FREE_int(Eqn);
	}


}


void surface_repository::init(
		surface_classify_wedge *Wedge, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_repository::init" << endl;
	}


	surface_repository::Wedge = Wedge;

	int orbit_index;

	nb_surfaces = Wedge->Surfaces->nb_orbits;

	if (f_v) {
		cout << "surface_repository::init nb_surfaces = " << nb_surfaces << endl;
	}

	//SaS = NEW_OBJECTS(data_structures_groups::set_and_stabilizer, nb_surfaces);
	SaS = (data_structures_groups::set_and_stabilizer **) NEW_pvoid(nb_surfaces);
	SOA = (cubic_surfaces_in_general::surface_object_with_group **) NEW_pvoid(nb_surfaces);

	Lines = NEW_lint(nb_surfaces * 27);
	Eqn = NEW_int(nb_surfaces * 20);

	if (f_v) {
		cout << "surface_repository::init processing "
				<< Wedge->Surfaces->nb_orbits << " surfaces" << endl;
	}
	for (orbit_index = 0;
			orbit_index < nb_surfaces;
			orbit_index++) {


		if (f_v) {
			cout << "surface_repository::init processing "
					"before init_one_surface, "
					"orbit_index = " << orbit_index << " / " << nb_surfaces << endl;
		}

		init_one_surface(
				orbit_index, verbose_level - 2);

		if (f_v) {
			cout << "surface_repository::init processing "
					"after init_one_surface, "
					"orbit_index = " << orbit_index << " / " << nb_surfaces << endl;
		}
	}
	if (f_v) {
		cout << "surface_repository::init processing "
				<< Wedge->Surfaces->nb_orbits << " surfaces done" << endl;
	}

	if (f_v) {
		cout << "surface_repository::init done" << endl;
	}
}

void surface_repository::init_one_surface(
		int orbit_index, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index << " / " << nb_surfaces << endl;
	}

	//long int Lines[27];
	int the_equation[20];

	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index
				<< " / " << Wedge->Surfaces->nb_orbits << endl;
	}

	SaS[orbit_index] = Wedge->Surfaces->get_set_and_stabilizer(
			orbit_index, 0 /* verbose_level */);

	Lint_vec_copy(SaS[orbit_index]->data, Lines + orbit_index * 27, 27);

	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index
				<< " / " << Wedge->Surfaces->nb_orbits
				<< " before Surf->build_cubic_surface_from_lines" << endl;
	}
	Wedge->Surf->build_cubic_surface_from_lines(
			27,
			Lines + orbit_index * 27,
			the_equation, verbose_level - 2);
	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index
				<< " / " << Wedge->Surfaces->nb_orbits
				<< " after Surf->build_cubic_surface_from_lines" << endl;
	}

	Wedge->F->Projective_space_basic->PG_element_normalize_from_front(
			the_equation, 1, 20);

	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index
				<< " / " << Wedge->Surfaces->nb_orbits
				<< " equation: " << endl;
		Int_vec_print(cout, the_equation, 20);
	}
	Int_vec_copy(the_equation, Eqn + orbit_index * 20, 20);

	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index
				<< " / " << Wedge->Surfaces->nb_orbits
				<< " testing the generators on the equation" << endl;
	}
	int ret;

	ret = SaS[orbit_index]->Strong_gens->test_if_they_stabilize_the_equation(
			the_equation,
			Wedge->Surf->PolynomialDomains->Poly3_4,
			verbose_level);

	if (!ret) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index
				<< " / " << Wedge->Surfaces->nb_orbits
				<< " the generators do not fix the equation" << endl;
		exit(1);
	}
	else {
		if (f_v) {
			cout << "surface_repository::init_one_surface "
					"orbit_index = " << orbit_index
					<< " / " << Wedge->Surfaces->nb_orbits
					<< " the generators fix the equation, good." << endl;
		}

	}



	geometry::algebraic_geometry::surface_object *SO;
	int *equation;

	equation = Eqn + orbit_index * 20;

	string label_txt;
	string label_tex;

	label_txt = "surface_q" + std::to_string(Wedge->q) + "_iso" + std::to_string(orbit_index);
	label_tex = "surface\\_q" + std::to_string(Wedge->q) + "\\_iso" + std::to_string(orbit_index);

	SO = NEW_OBJECT(geometry::algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"before SO->init_with_27_lines "
				"orbit_index = " << orbit_index << endl;
	}
	SO->init_with_27_lines(
			Wedge->Surf,
			Lines + orbit_index * 27,
			equation,
			label_txt, label_tex,
			true /*f_find_double_six_and_rearrange_lines*/,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"after SO->init_with_27_lines "
				"orbit_index = " << orbit_index << endl;
	}



	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"before SO->compute_properties "
				"orbit_index = " << orbit_index << endl;
	}
	SO->compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"after SO->compute_properties "
				"orbit_index = " << orbit_index << endl;
	}





	SOA[orbit_index] = NEW_OBJECT(cubic_surfaces_in_general::surface_object_with_group);

	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"before SOA->init_surface_object "
				"orbit_index = " << orbit_index << endl;
	}
	SOA[orbit_index]->init_surface_object(
			Wedge->Surf_A, SO,
			SaS[orbit_index]->Strong_gens,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"after SOA->init_surface_object "
				"orbit_index = " << orbit_index << endl;
	}




	if (f_v) {
		cout << "surface_repository::init_one_surface "
				"orbit_index = " << orbit_index << " / " << nb_surfaces
				<< " done" << endl;
	}
}


void surface_repository::generate_source_code(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	std::string fname;
	std::string fname_base;
	int orbit_index;
	int i, j;

	if (f_v) {
		cout << "surface_repository::generate_source_code" << endl;
	}
	fname_base = Wedge->fname_base;
	fname = Wedge->fname_base + ".cpp";

	{
		ofstream f(fname);

		other::orbiter_kernel_system::os_interface Os;
		string str;

		Os.get_date(str);


		f << "// file " << fname << endl;
		f << "// created by Orbiter" << endl;
		f << "// date " << str << endl;
		f << "static int " << Wedge->fname_base << "_nb_reps = "
				<< nb_surfaces << ";" << endl;
		f << "static int " << fname_base << "_size = "
				<< Wedge->Surf->PolynomialDomains->nb_monomials << ";" << endl;



		if (f_v) {
			cout << "surface_repository::generate_source_code "
					"preparing reps" << endl;
		}
		f << "// the equations:" << endl;
		f << "static int " << fname_base << "_reps[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_surfaces;
				orbit_index++) {


			int *equation;

			equation = Eqn + orbit_index * 20;
			f << "\t";
			for (i = 0; i < Wedge->Surf->PolynomialDomains->nb_monomials; i++) {
				f << equation[i];
				f << ", ";
			}
			f << endl;

		}
		f << "};" << endl;



		if (f_v) {
			cout << "surface_repository::generate_source_code "
					"preparing stab_order" << endl;
		}
		f << "// the stabilizer orders:" << endl;
		f << "static const char *" << fname_base << "_stab_order[] = {" << endl;
		for (orbit_index = 0;
				orbit_index < nb_surfaces;
				orbit_index++) {

			algebra::ring_theory::longinteger_object ago;

			SaS[orbit_index]->Strong_gens->group_order(ago);

			f << "\t\"";

			ago.print_not_scientific(f);
			f << "\"," << endl;

		}
		f << "};" << endl;


		if (f_v) {
			cout << "surface_repository::generate_source_code "
					"preparing nb_E" << endl;
		}
		f << "// the number of Eckardt points:" << endl;
		f << "static int " << fname_base << "_nb_E[] = { " << endl << "\t";
		for (orbit_index = 0;
				orbit_index < nb_surfaces;
				orbit_index++) {

#if 0
			long int *Pts;
			int nb_pts;
			other::data_structures::set_of_sets *pts_on_lines;
			int nb_E;

			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"orbit_index = " << orbit_index << endl;
			}

			Pts = NEW_lint(Wedge->Surf->nb_pts_on_surface_with_27_lines);

			vector<long int> Points;
			int h;

			int *equation;

			equation = Eqn + orbit_index * 20;

			if (f_v) {
				cout << "equation:" << endl;
				Int_vec_print(cout, equation, 20);
				cout << endl;
			}
			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"before Wedge->Surf->enumerate_points" << endl;
			}
			Wedge->Surf->enumerate_points(
					equation, Points,
					0 /* verbose_level */);
			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"after Wedge->Surf->enumerate_points" << endl;
			}

			nb_pts = Points.size();
			Pts = NEW_lint(nb_pts);
			for (h = 0; h < nb_pts; h++) {
				Pts[h] = Points[h];
			}

			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"nb_pts = " << nb_pts << endl;
				cout << "surface_repository::generate_source_code "
						"Pts=";
				Lint_vec_print(cout, Pts, nb_pts);
				cout << endl;
			}



			if (nb_pts != Wedge->Surf->nb_pts_on_surface_with_27_lines) {
				cout << "surface_repository::generate_source_code "
						"nb_pts != Wedge->Surf->nb_pts_on_surface_with_27_lines" << endl;
				exit(1);
			}

			int *f_is_on_line;

			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"before Wedge->Surf->compute_points_on_lines" << endl;
			}
			Wedge->Surf->compute_points_on_lines(
					Pts, nb_pts,
				Lines + orbit_index * 27, 27 /*nb_lines*/,
				pts_on_lines,
				f_is_on_line,
				0/*verbose_level*/);
			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"after Wedge->Surf->compute_points_on_lines" << endl;
			}

			if (!pts_on_lines->has_constant_size_property()) {
				cout << "surface_repository::generate_source_code "
						"pts_on_lines:" << endl;
				pts_on_lines->print_table();
				cout << "equation:" << endl;
				Int_vec_print(cout, equation, 20);
				cout << endl;
				cout << "surface_repository::generate_source_code "
						"pts_on_lines does not have the constant size property. "
						"Something is wrong." << endl;
				exit(1);
			}
			if (pts_on_lines->get_constant_size() != Wedge->q + 1) {
				cout << "surface_repository::generate_source_code "
						"pts_on_lines:" << endl;
				pts_on_lines->print_table();
				cout << "equation:" << endl;
				Int_vec_print(cout, equation, 20);
				cout << endl;
				cout << "surface_repository::generate_source_code "
						"The lines do not have exactly q + 1 points. "
						"Something is wrong" << endl;
				exit(1);

			}

			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"pts_on_lines:" << endl;
				pts_on_lines->print_table();
			}

			FREE_int(f_is_on_line);

			nb_E = pts_on_lines->number_of_eckardt_points(verbose_level);
#endif

			int nb_E;

			nb_E = SOA[orbit_index]->SO->SOP->nb_Eckardt_points;

			f << nb_E;
			if (orbit_index < nb_surfaces - 1) {
				f << ", ";
			}
			if (((orbit_index + 1) % 10) == 0) {
				f << endl << "\t";
			}


			//FREE_OBJECT(pts_on_lines);
			//FREE_lint(Pts);
		}
		f << "};" << endl;



		if (f_v) {
			cout << "surface_repository::generate_source_code "
					"preparing Lines" << endl;
		}
		f << "// the lines in the order of the double six as "
				"a_i, b_i and 15 more lines c_ij:" << endl;
		f << "static long int " << fname_base << "_Lines[] = { " << endl;


		for (orbit_index = 0;
				orbit_index < nb_surfaces;
				orbit_index++) {

			f << "\t";
			for (j = 0; j < 27; j++) {
				f << Lines[orbit_index * 27 + j];
				f << ", ";
			}
			f << endl;

		}
		f << "};" << endl;

		f << "static int " << fname_base << "_make_element_size = "
				<< Wedge->A->make_element_size << ";" << endl;

		{
			int *stab_gens_first;
			int *stab_gens_len;
			int fst;

			stab_gens_first = NEW_int(nb_surfaces);
			stab_gens_len = NEW_int(nb_surfaces);
			fst = 0;
			for (orbit_index = 0;
					orbit_index < nb_surfaces;
					orbit_index++) {

				stab_gens_first[orbit_index] = fst;
				stab_gens_len[orbit_index] =
						SaS[orbit_index]->Strong_gens->gens->len;
				//stab_gens_len[orbit_index] =
				//The_surface[iso_type]->stab_gens->gens->len;
				fst += stab_gens_len[orbit_index];
			}


			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"preparing stab_gens_fst" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_fst[] = { ";
			for (orbit_index = 0;
					orbit_index < nb_surfaces;
					orbit_index++) {

				f << stab_gens_first[orbit_index];
				if (orbit_index < nb_surfaces - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;

			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"preparing stab_gens_len" << endl;
			}
			f << "static int " << fname_base << "_stab_gens_len[] = { ";
			for (orbit_index = 0;
					orbit_index < nb_surfaces;
					orbit_index++) {
				f << stab_gens_len[orbit_index];
				if (orbit_index < nb_surfaces - 1) {
					f << ", ";
				}
				if (((orbit_index + 1) % 10) == 0) {
					f << endl << "\t";
				}
			}
			f << "};" << endl;


			if (f_v) {
				cout << "surface_repository::generate_source_code "
						"preparing stab_gens" << endl;
			}
			f << "static int " << fname_base << "_stab_gens[] = {" << endl;
			for (orbit_index = 0;
					orbit_index < nb_surfaces;
					orbit_index++) {
				int j;

				for (j = 0; j < stab_gens_len[orbit_index]; j++) {
					if (f_vv) {
						cout << "surface_repository::generate_source_code "
								"before extract_strong_generators_in_order "
								"generator " << j << " / "
								<< stab_gens_len[orbit_index] << endl;
					}
					f << "\t";
					Wedge->A->Group_element->element_print_for_make_element(
							SaS[orbit_index]->Strong_gens->gens->ith(j), f);
					//A->element_print_for_make_element(
					//The_surface[iso_type]->stab_gens->gens->ith(j), f);
					f << endl;
				}
			}
			f << "};" << endl;


			FREE_int(stab_gens_first);
			FREE_int(stab_gens_len);
		}
	}

	other::orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "surface_repository::generate_source_code done" << endl;
	}
}

void surface_repository::report_surface(
		std::ostream &ost,
		int orbit_index,
		int f_print_orbits, std::string &fname_mask,
		other::graphics::layered_graph_draw_options *draw_options,
		int max_nb_elements_printed,
		//poset_classification::poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_repository::report_surface "
				"orbit_index = " << orbit_index << endl;
	}

	ost << endl; // << "\\clearpage" << endl << endl;
	ost << "\\section*{Surface $" << Wedge->q << "\\#"
			<< orbit_index << "$}" << endl;



	//Surf->print_equation_wrapped(ost, equation);
#if 0
	geometry::algebraic_geometry::surface_object *SO;
	int *equation;

	equation = Eqn + orbit_index * 20;

	string label_txt;
	string label_tex;

	label_txt = "surface_q" + std::to_string(Wedge->q) + "_iso" + std::to_string(orbit_index);
	label_tex = "surface\\_q" + std::to_string(Wedge->q) + "\\_iso" + std::to_string(orbit_index);

	SO = NEW_OBJECT(geometry::algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->init_with_27_lines "
				"orbit_index = " << orbit_index << endl;
	}
	SO->init_with_27_lines(
			Wedge->Surf,
			Lines + orbit_index * 27,
			equation,
			label_txt, label_tex,
			true /*f_find_double_six_and_rearrange_lines*/,
			verbose_level - 2);
	if (f_v) {
		cout << "surface_repository::report_surface "
				"after SO->init_with_27_lines "
				"orbit_index = " << orbit_index << endl;
	}



#if 0
	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->enumerate_points" << endl;
	}
	SO->enumerate_points(verbose_level);
#endif

	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->compute_properties "
				"orbit_index = " << orbit_index << endl;
	}
	SO->compute_properties(verbose_level - 2);
	if (f_v) {
		cout << "surface_repository::report_surface "
				"after SO->compute_properties "
				"orbit_index = " << orbit_index << endl;
	}


	SO->SOP->print_equation(ost);


	cubic_surfaces_in_general::surface_object_with_group *SOA;

	SOA = NEW_OBJECT(cubic_surfaces_in_general::surface_object_with_group);

	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SOA->init_surface_object "
				"orbit_index = " << orbit_index << endl;
	}
	SOA->init_surface_object(
			Wedge->Surf_A, SO,
		SaS[orbit_index]->Strong_gens,
		verbose_level - 2);
	if (f_v) {
		cout << "surface_repository::report_surface "
				"after SOA->init_surface_object "
				"orbit_index = " << orbit_index << endl;
	}
#endif



	SOA[orbit_index]->cheat_sheet(
			ost,
			f_print_orbits, fname_mask,
			draw_options,
			max_nb_elements_printed,
			verbose_level);


#if 0
	geometry::algebraic_geometry::surface_object *SO;

	SO = SOA[orbit_index]->SO;


	algebra::ring_theory::longinteger_object ago;
	SaS[orbit_index]->Strong_gens->group_order(ago);
	ost << "The automorphism group of the surface "
			"has order " << ago << "\\\\" << endl;
	ost << "The automorphism group is the following group\\\\" << endl;

	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SaS->Strong_gens->print_generators_tex "
				"orbit_index = " << orbit_index << endl;
	}
	SaS[orbit_index]->Strong_gens->print_generators_tex(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->print_summary "
				"orbit_index = " << orbit_index << endl;
	}
	SO->SOP->print_summary(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;



	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->print_lines "
				"orbit_index = " << orbit_index << endl;
	}
	SO->SOP->print_lines(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;



	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->print_points "
				"orbit_index = " << orbit_index << endl;
	}
	SO->SOP->print_points(ost);


	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;

	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->print_Hesse_planes "
				"orbit_index = " << orbit_index << endl;
	}
	SO->SOP->print_Hesse_planes(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->print_tritangent_planes "
				"orbit_index = " << orbit_index << endl;
	}
	SO->SOP->SmoothProperties->print_tritangent_planes(ost);

	ost << endl;
	ost << "\\clearpage" << endl;
	ost << endl;


	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->print_axes "
				"orbit_index = " << orbit_index << endl;
	}
	SO->SOP->print_axes(ost);
	if (f_v) {
		cout << "surface_repository::report_surface "
				"after SO->print_axes "
				"orbit_index = " << orbit_index << endl;
	}


	//New_clebsch->SO->print_planes_in_trihedral_pairs(fp);

#if 0
	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SO->print_generalized_quadrangle "
				"orbit_index = " << orbit_index << endl;
	}
	SO->print_generalized_quadrangle(ost);

	if (f_v) {
		cout << "surface_repository::report_surface "
				"before SOA->quartic" << endl;
	}
	SOA->quartic(ost,  verbose_level);
#endif


#if 0
	if (f_v) {
		cout << "surface_repository::report_surface "
				"before FREE_OBJECT(SOA) "
				"orbit_index = " << orbit_index << endl;
	}
	FREE_OBJECT(SOA);
	if (f_v) {
		cout << "surface_repository::report_surface "
				"after FREE_OBJECT(SOA) "
				"orbit_index = " << orbit_index << endl;
	}
	if (f_v) {
		cout << "surface_repository::report_surface "
				"before FREE_OBJECT(SO) "
				"orbit_index = " << orbit_index << endl;
	}
	FREE_OBJECT(SO);
	if (f_v) {
		cout << "surface_repository::report_surface "
				"after FREE_OBJECT(SO) "
				"orbit_index = " << orbit_index << endl;
	}
#endif
#endif

	if (f_v) {
		cout << "surface_repository::report_surface "
				"orbit_index = " << orbit_index << " done" << endl;
	}
}


}}}}


