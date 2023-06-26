/*
 * spread_activity.cpp
 *
 *  Created on: Sep 19, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


spread_activity::spread_activity()
{
	Descr = NULL;
	Spread_create = NULL;
	SD = NULL;
	A = NULL;
	A2 = NULL;
	AG = NULL;
	AGr = NULL;
}

spread_activity::~spread_activity()
{
}

void spread_activity::init(
		spread_activity_description *Descr,
		spread_create *Spread_create,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::init" << endl;
	}

	spread_activity::Descr = Descr;
	spread_activity::Spread_create = Spread_create;

	SD = NEW_OBJECT(geometry::spread_domain);


	if (f_v) {
		cout << "spread_activity::init "
				"before SD->init_spread_domain" << endl;
	}

	SD->init_spread_domain(
			Spread_create->F,
			2 * Spread_create->k, Spread_create->k,
			verbose_level - 1);

	if (f_v) {
		cout << "spread_activity::init "
				"after SD->init_spread_domain" << endl;
	}



	A = Spread_create->A;

	A2 = NEW_OBJECT(actions::action);
	AG = NEW_OBJECT(induced_actions::action_on_grassmannian);

#if 0
	longinteger_object go;
	A->Sims->group_order(go);
	if (f_v) {
		cout << "spread_activity::init go = " << go <<  endl;
	}
#endif


	if (f_v) {
		cout << "action A: ";
		A->print_info();
	}





	if (f_v) {
		cout << "spread_activity::init "
				"before AG->init" <<  endl;
	}

	AG->init(*A, SD->Grass, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "spread_activity::init "
				"after AG->init" <<  endl;
	}

	if (f_v) {
		cout << "spread_activity::init before "
				"induced_action_on_grassmannian_preloaded" <<  endl;
	}

	A2 = A->Induced_action->induced_action_on_grassmannian_preloaded(AG,
		false /*f_induce_action*/, NULL /*sims *old_G */,
		0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "spread_activity::init after "
				"induced_action_on_grassmannian_preloaded" <<  endl;
	}

	if (f_v) {
		cout << "action A2 created: ";
		A2->print_info();
	}

	std::string label_of_set;


	label_of_set.assign("on_grassmannian");

	AGr = A2->Induced_action->restricted_action(
			Spread_create->set, Spread_create->sz, label_of_set,
			verbose_level);

	if (f_v) {
		cout << "action AGr created: ";
		AGr->print_info();
	}


	if (f_v) {
		cout << "spread_activity::init done" << endl;
	}
}


void spread_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::perform_activity" << endl;
	}


	if (Descr->f_report) {

		if (f_v) {
			cout << "spread_activity::perform_activity f_report" << endl;
		}


		if (f_v) {
			cout << "spread_classify_activity::perform_activity "
					"before Spread_classify->classify_partial_spreads" << endl;
		}
		report(verbose_level);
		if (f_v) {
			cout << "spread_classify_activity::perform_activity "
					"after Spread_classify->classify_partial_spreads" << endl;
		}

		if (f_v) {
			cout << "spread_classify_activity::perform_activity "
					"f_report done" << endl;
		}

	}


	if (f_v) {
		cout << "spread_activity::perform_activity done" << endl;
	}

}

void spread_activity::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::report" << endl;
	}

	string fname;
	string title, author, extra_praeamble;

	fname = Spread_create->label_txt + "_report.tex";

	title = "Translation plane " + Spread_create->label_tex;


	{
		ofstream ost(fname);

		l1_interfaces::latex_interface L;

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


		if (f_v) {
			cout << "spread_activity::report before report2" << endl;
		}
		report2(ost, verbose_level);
		if (f_v) {
			cout << "spread_activity::report after report2" << endl;
		}


		L.foot(ost);
	}


	if (f_v) {
		cout << "spread_activity::report done" << endl;
	}
}

void spread_activity::report2(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_activity::report2" << endl;
		cout << "spread_activity::report2 spread_size=" << SD->spread_size << endl;
	}


	ost << "The spread: \\\\" << endl;

	if (f_v) {
		cout << "spread_activity::report2 before Grass->print_set_tex" << endl;
	}
	SD->Grass->print_set_tex(
			ost, Spread_create->set, Spread_create->sz, verbose_level);
	if (f_v) {
		cout << "spread_activity::report2 after Grass->print_set_tex" << endl;
	}


	int *Spread_set;
	int sz, i, k, k2;
	k = Spread_create->k;
	k2 = k * k;
	l1_interfaces::latex_interface Li;

	if (f_v) {
		cout << "spread_activity::report2 "
				"before SD->Grass->make_spread_set_from_spread" << endl;
	}

	SD->Grass->make_spread_set_from_spread(
			Spread_create->set, Spread_create->sz,
			Spread_set, sz,
			verbose_level);

	if (f_v) {
		cout << "spread_activity::report2 "
				"after SD->Grass->make_spread_set_from_spread" << endl;
	}

	ost << "The spread set: \\\\" << endl;
	for (i = 0; i < sz; i++) {
		ost << "$";
		ost << "\\left[";
		Li.print_integer_matrix_tex(ost, Spread_set + i * k2, k, k);
		ost << "\\right]";
		ost << "$";
		if (i < sz - 1) {
			ost << ", ";
		}
	}
	ost << "\\\\" << endl;


	if (Spread_create->f_has_group) {

		ring_theory::longinteger_object go;

		Spread_create->Sg->group_order(go);

		ost << "The spread stabilizer has order: " << go << " \\\\" << endl;

		Spread_create->Sg->print_generators_tex(ost);


		groups::orbits_on_something *O1;
		groups::orbits_on_something *O2;
		string prefix1, prefix2;

		prefix1.assign(Spread_create->label_txt);
		prefix1.append("_on_gr");
		prefix2.assign(Spread_create->label_txt);
		prefix2.append("_on_spread");

		O1 = NEW_OBJECT(groups::orbits_on_something);

		O1->init(A2,
				Spread_create->Sg,
				false /* f_load_save */,
				prefix1,
				verbose_level);

		ost << "Orbits on grassmannian: ";
		O1->Sch->print_orbit_lengths_tex(ost);
		ost << "\\\\" << endl;

		O2 = NEW_OBJECT(groups::orbits_on_something);

		O2->init(AGr,
				Spread_create->Sg,
				false /* f_load_save */,
				prefix2,
				verbose_level);


		ost << "Orbits on spread: ";
		O2->Sch->print_orbit_lengths_tex(ost);
		ost << "\\\\" << endl;

		O2->Classify_orbits_by_length->Set_partition->print_table_latex_simple(ost);


		ost << "Orbits: \\\\";
		O2->Sch->list_all_orbits_tex(ost);
		ost << "\\\\" << endl;

		int f, l, j;
		long int a;
		long int *Orbit_elements;
		long int *Orbit_elements_lines;

		for (i = 0; i < O2->Sch->nb_orbits; i++) {
			f = O2->Sch->orbit_first[i];
			l = O2->Sch->orbit_len[i];
			Orbit_elements = NEW_lint(l);
			Orbit_elements_lines = NEW_lint(l);
			for (j = 0; j < l; j++) {
				a = O2->Sch->orbit[f + j];
				Orbit_elements[j] = a;
			}

			for (j = 0; j < l; j++) {
				a = Orbit_elements[j];
				Orbit_elements_lines[j] = Spread_create->set[a];
			}



			ost << "Orbit of length " << l << " consists of these subspaces:\\\\" << endl;
			Lint_vec_print(ost, Orbit_elements_lines, l);
			ost << "\\\\" << endl;
			SD->Grass->print_set_tex(ost, Orbit_elements_lines, l, verbose_level);

			FREE_lint(Orbit_elements);
			FREE_lint(Orbit_elements_lines);
		}

		if (Spread_create->k == 2) {


			geometry::projective_space *P5;
			orthogonal_geometry::orthogonal *O;
			geometry::klein_correspondence *Klein;

			P5 = NEW_OBJECT(geometry::projective_space);
			if (f_v) {
				cout << "spread_activity::report2 "
						"before P5->projective_space_init" << endl;
			}
			P5->projective_space_init(5, Spread_create->F,
				false /*f_init_incidence_structure */,
				verbose_level - 2);
			if (f_v) {
				cout << "spread_activity::report2 "
						"after P5->projective_space_init" << endl;
			}



			if (f_v) {
				cout << "spread_activity::report2 "
						"initializing orthogonal" << endl;
			}
			O = NEW_OBJECT(orthogonal_geometry::orthogonal);
			O->init(
					1 /* epsilon */,
					6 /* n */,
					Spread_create->F,
					verbose_level - 2);
			if (f_v) {
				cout << "spread_activity::report2 "
						"initializing orthogonal done" << endl;
			}

			Klein = NEW_OBJECT(geometry::klein_correspondence);

			if (f_v) {
				cout << "spread_activity::report2 before Klein->init" << endl;
			}
			Klein->init(Spread_create->F, O, verbose_level - 2);
			if (f_v) {
				cout << "spread_activity::report2 after Klein->init" << endl;
			}


			long int *Pts_on_Klein;
			long int *Pts_in_PG5;
			long int line_rk;


			for (i = 0; i < O2->Sch->nb_orbits; i++) {
				f = O2->Sch->orbit_first[i];
				l = O2->Sch->orbit_len[i];

				if (f_v) {
					cout << "spread_activity::report2 orbit " << i << " of length " << l << endl;
				}

				Orbit_elements = NEW_lint(l);
				Orbit_elements_lines = NEW_lint(l);
				Pts_on_Klein = NEW_lint(l);
				Pts_in_PG5 = NEW_lint(l);
				for (j = 0; j < l; j++) {
					a = O2->Sch->orbit[f + j];
					Orbit_elements[j] = a;
				}

				for (j = 0; j < l; j++) {
					a = Orbit_elements[j];

					line_rk = Spread_create->set[a];
					Orbit_elements_lines[j] = line_rk;

					Pts_on_Klein[j] = Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level */);
					Pts_in_PG5[j] = Klein->point_on_quadric_embedded_in_P5(Pts_on_Klein[j]);
				}
				int *v;
				int *w;

				v = NEW_int(l * 6);
				w = NEW_int(l * 6);
				for (j = 0; j < l; j++) {
					P5->unrank_point(v + j * 6, Pts_in_PG5[j]);
				}

				ost << "List of points:\\\\" << endl;
				for (j = 0; j < l; j++) {
					Int_vec_print(ost, v + j * 6, 6);
					ost << "\\\\" << endl;
				}
				ost << "j : pt on Klein : pt in  P5:\\\\" << endl;
				for (j = 0; j < l; j++) {
					ost << j << " : ";
					ost << Pts_on_Klein[j] << " : ";
					ost << Pts_in_PG5[j];
					ost << "\\\\" << endl;
				}
				ost << "Points on Klein quadric:" << endl;
				Lint_vec_print(ost, Pts_on_Klein, l);
				ost << "\\\\" << endl;

				ost << "Points in P5:" << endl;
				Lint_vec_print(ost, Pts_in_PG5, l);
				ost << "\\\\" << endl;


				int rk;
				int base_cols[6];
				int f_complete = true;

				rk = Spread_create->F->Linear_algebra->rank_of_rectangular_matrix_memory_given(v,
						l, 6, w, base_cols, f_complete, 0 /* verbose_level*/);
				ost << "rk=" << rk << "\\\\" << endl;
				ost << "RREF:\\\\" << endl;
				for (j = 0; j < rk; j++) {
					Int_vec_print(ost, w + j * 6, 6);
					ost << "\\\\" << endl;
				}

				int *type;

				long int nb_planes;

				nb_planes = P5->Subspaces->nb_rk_k_subspaces_as_lint(3);
				if (f_v) {
					cout << "spread_activity::report2 nb_planes = " << nb_planes << endl;
				}


				type = NEW_int(nb_planes);

				if (f_v) {
					cout << "spread_activity::report2 "
							"before P5->plane_intersection_type_basic" << endl;
				}
				P5->Subspaces->plane_intersection_type_basic(Pts_in_PG5, l,
						type, 0 /* verbose_level*/);
						// type[N_planes]

				data_structures::tally T;

				T.init(type, nb_planes, false, 0);



				ost << "Orbit of length " << l << " consists of these subspaces:\\\\" << endl;
				Lint_vec_print(ost, Orbit_elements_lines, l);
				ost << "\\\\" << endl;
				SD->Grass->print_set_tex(ost, Orbit_elements_lines, l, verbose_level);
				ost << "Orbit of length " << l << " consists of these points on the Klein quadric:\\\\" << endl;
				Lint_vec_print(ost, Pts_on_Klein, l);
				ost << "\\\\" << endl;

				ost << "Plane type: ";
				ost << "$";
				T.print_naked_tex(ost, true /* f_backwards */);
				ost << "$";
				ost << "\\\\" << endl;


				if (f_v) {
					cout << "spread_activity::report2 "
							"before FREE_int(type);" << endl;
				}
				FREE_int(type);
				if (f_v) {
					cout << "spread_activity::report2 "
							"before FREE_lint(Orbit_elements);" << endl;
				}
				FREE_lint(Orbit_elements);
				FREE_lint(Orbit_elements_lines);
				if (f_v) {
					cout << "spread_activity::report2 "
							"before FREE_lint(Pts_on_Klein);" << endl;
				}
				FREE_lint(Pts_on_Klein);
				FREE_lint(Pts_in_PG5);
				FREE_int(v);
				FREE_int(w);
			}

			if (f_v) {
				cout << "spread_activity::report2 "
						"before FREE_OBJECT(P5);" << endl;
			}
			FREE_OBJECT(P5);
			if (f_v) {
				cout << "spread_activity::report2 "
						"before FREE_OBJECT(O);" << endl;
			}
			FREE_OBJECT(O);
			if (f_v) {
				cout << "spread_activity::report2 "
						"before FREE_OBJECT(Klein);" << endl;
			}
			FREE_OBJECT(Klein);
			if (f_v) {
				cout << "spread_activity::report2 "
						"after FREE_OBJECT(Klein);" << endl;
			}

		} // if k == 2


	}

	if (f_v) {
		cout << "spread_activity::report2 done" << endl;
	}
}




}}}




