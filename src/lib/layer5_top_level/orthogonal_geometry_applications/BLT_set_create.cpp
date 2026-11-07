// BLT_set_create.cpp
// 
// Anton Betten
//
// March 17, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


BLT_set_create::BLT_set_create()
{
	Record_birth();
	Descr = NULL;

	//std::string prefix;

	//std::string label_txt;
	//std::string label_tex;

	OA = NULL;

	Blt_set = NULL;

	//set = NULL;
	//ABC = NULL;


	f_has_group = false;
	Sg = NULL;
	Blt_set_domain_with_action = NULL;
	BA = NULL;

	FQ = NULL;
	q = 0;
	Q = 0;

	U = NULL;
}

BLT_set_create::~BLT_set_create()
{
	Record_death();

	if (Blt_set) {
		FREE_OBJECT(Blt_set);
	}
	if (Sg) {
		FREE_OBJECT(Sg);
	}
	if (BA) {
		FREE_OBJECT(BA);
	}
	if (FQ) {
		FREE_OBJECT(FQ);
	}
	if (U) {
		FREE_OBJECT(U);
	}
}

void BLT_set_create::init(
		BLT_set_create_description *Descr,
		orthogonal_space_with_action *OA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "BLT_set_create::init" << endl;
	}
	BLT_set_create::Descr = Descr;
	BLT_set_create::OA = OA;
	BLT_set_create::Blt_set_domain_with_action = OA->Blt_set_domain_with_action;

	if (OA->Descr->n != 5) {
		cout << "BLT_set_create::init OA->Descr->n != 5" << endl;
		exit(1);
	}


	Blt_set = NEW_OBJECT(geometry::orthogonal_geometry::blt_set);

	
	if (f_v) {
		cout << "BLT_set_create::init "
				"before Blt_set->init" << endl;
	}
	Blt_set->init(
			Blt_set_domain_with_action->Blt_set_domain,
			verbose_level - 2);
	if (f_v) {
		cout << "BLT_set_create::init "
				"after Blt_set->init" << endl;
	}


	if (Descr->f_family) {

		if (f_v) {
			cout << "BLT_set_create::init "
					"before create_family" << endl;
		}
		create_family(verbose_level - 2);
		if (f_v) {
			cout << "BLT_set_create::init "
					"after create_family" << endl;
		}

	}
	else if (Descr->f_flock) {
		if (f_v) {
			cout << "BLT_set_create::init f_flock" << endl;
		}

		f_has_group = false;
		geometry::orthogonal_geometry::orthogonal_global OG;

		int *ABC_matrix;
		int m, n;

		Get_matrix(Descr->flock_label, ABC_matrix, m, n);
		if (m != OA->Descr->F->q) {
			cout << "BLT_set_create::init m != OA->Descr->F->q" << endl;
			exit(1);
		}
		if (n != 3) {
			cout << "BLT_set_create::init n != 3" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "BLT_set_create::init flock:" << endl;
			Int_matrix_print(ABC_matrix, OA->Descr->F->q, 3);
		}

		Blt_set = NEW_OBJECT(geometry::orthogonal_geometry::blt_set);

		//set = NEW_lint(OA->Descr->F->q + 1);

		Int_vec_copy(ABC_matrix, Blt_set->ABC, m * n);

		FREE_int(ABC_matrix);


		// creates the points (c/2,-c^2/4-ab,1,b,a) : (a,b,c) \in FLock.


		if (f_v) {
			cout << "BLT_set_create::init "
					"before create_BLT_set_from_flock" << endl;
		}
		OG.create_BLT_set_from_flock(
				OA->O,
				Blt_set->set, Blt_set->ABC, verbose_level - 2);
		if (f_v) {
			cout << "BLT_set_create::init "
					"after create_BLT_set_from_flock" << endl;
		}


		string str_q;

		str_q = "q" + std::to_string(OA->Descr->F->q);
		prefix = Descr->flock_label;

		Blt_set->label_txt = prefix + "_" + str_q;
		Blt_set->label_tex = prefix + "\\_" + str_q;

	}

	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "BLT_set_create::init BLT set from catalogue" << endl;
		}

		if (f_v) {
			cout << "BLT_set_create::init before make_from_catalogue" << endl;
		}
		make_from_catalogue(
				verbose_level - 2);
		if (f_v) {
			cout << "BLT_set_create::init after make_from_catalogue" << endl;
		}

	}
	else {
		cout << "BLT_set_create::init we do not recognize "
				"the type of BLT-set" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "BLT_set_create::init set = ";
		Lint_vec_print(cout, Blt_set->set, OA->Descr->F->q + 1);
		cout << endl;
	}


	if (f_v) {
		cout << "BLT_set_create::init "
				"before Blt_set->compute_properties" << endl;
	}
	Blt_set->compute_properties(
			Descr->f_invariants,
			verbose_level - 2);
	if (f_v) {
		cout << "BLT_set_create::init "
				"after Blt_set->compute_properties" << endl;
	}



	if (f_has_group) {
		cout << "BLT_set_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
	}



	BA = NEW_OBJECT(blt_set_with_action);

	if (f_v) {
		cout << "BLT_set_create::init before BA->init_set" << endl;
	}
	BA->init_set(
			OA->A,
			Blt_set_domain_with_action,
			Blt_set,
			Sg,
			//Descr->f_invariants,
			verbose_level - 2);
	if (f_v) {
		cout << "BLT_set_create::init after BA->init_set" << endl;
	}




	if (f_v) {
		cout << "BLT_set_create::init done" << endl;
	}
}

void BLT_set_create::create_family(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::create_family" << endl;
	}

	if (f_v) {
		cout << "BLT_set_create::create_family "
				"family_name=" << Descr->family_name << endl;
	}

	string str;
	string str_q;
	f_has_group = false;

	other::data_structures::string_tools ST;


	if (ST.stringcmp(Descr->family_name, "Linear") == 0) {

		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object of family Linear" << endl;
		}


		if (f_v) {
			cout << "BLT_set_create::create_family "
					"before create_quadratic_field_extension" << endl;
		}
		create_quadratic_field_extension(verbose_level - 2);
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"after create_quadratic_field_extension" << endl;
		}


		U = NEW_OBJECT(geometry::orthogonal_geometry::unusual_model);


		U->setup(FQ, OA->Descr->F, verbose_level);

		U->create_Linear_BLT_set(Blt_set->set, Blt_set->ABC, verbose_level);



		str = "Linear";
		str_q = "q" + std::to_string(OA->Descr->F->q);

	}

	else if (ST.stringcmp(Descr->family_name, "Fisher") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object of family Fisher" << endl;
		}

		if (f_v) {
			cout << "BLT_set_create::create_family "
					"before create_quadratic_field_extension" << endl;
		}
		create_quadratic_field_extension(verbose_level - 2);
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"after create_quadratic_field_extension" << endl;
		}


		U = NEW_OBJECT(geometry::orthogonal_geometry::unusual_model);

		U->setup(FQ, OA->Descr->F, verbose_level);

		U->create_Fisher_BLT_set(Blt_set->set, Blt_set->ABC, verbose_level);


		str = "Fisher";
		str_q = "q" + std::to_string(OA->Descr->F->q);

	}

	else if (ST.stringcmp(Descr->family_name, "Mondello") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object of family Mondello" << endl;
		}


		if (f_v) {
			cout << "BLT_set_create::create_family "
					"before create_quadratic_field_extension" << endl;
		}
		create_quadratic_field_extension(verbose_level - 2);
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"after create_quadratic_field_extension" << endl;
		}


		U = NEW_OBJECT(geometry::orthogonal_geometry::unusual_model);

		U->setup(FQ, OA->Descr->F, verbose_level);

		U->create_Mondello_BLT_set(Blt_set->set, Blt_set->ABC, verbose_level);


		str = "Mondello";
		str_q = "q" + std::to_string(OA->Descr->F->q);

	}


	else if (ST.stringcmp(Descr->family_name, "FTWKB") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object of family FTWKB" << endl;
		}

		geometry::orthogonal_geometry::orthogonal_global OG;

		OG.create_FTWKB_flock_and_BLT_set(
				OA->O,
				Blt_set->set, Blt_set->ABC,
				verbose_level);
		// for q congruent 2 mod 3
		// a(t)= t, b(t) = 3*t^2, c(t) = 3*t^3, all t \in GF(q)
		// together with the point (0, 0, 0, 1, 0)

		str = "FTWKB";
		str_q = "q" + std::to_string(OA->Descr->F->q);

	}
	else if (ST.stringcmp(Descr->family_name, "Kantor1") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object of family Kantor1" << endl;
		}

		geometry::orthogonal_geometry::orthogonal_global OG;

		OG.create_K1_flock_and_BLT_set(
				OA->O,
				Blt_set->set, Blt_set->ABC,
				verbose_level);
		// for a non-square m, and q=p^e
		// a(t)= t, b(t) = 0, c(t) = -m*t^p, all t \in GF(q)
		// together with the point (0, 0, 0, 1, 0)

		str = "Kantor1";
		str_q = "q" + std::to_string(OA->Descr->F->q);
	}
	else if (ST.stringcmp(Descr->family_name, "Kantor2") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object of family Kantor2" << endl;
		}


		geometry::orthogonal_geometry::orthogonal_global OG;

		OG.create_K2_flock_and_BLT_set(
				OA->O,
				Blt_set->set, Blt_set->ABC,
				verbose_level);
		// for q congruent 2 or 3 mod 5
		// a(t)= t, b(t) = 5*t^3, c(t) = 5*t^5, all t \in GF(q)
		// together with the point (0, 0, 0, 1, 0)

		str = "Kantor2";
		str_q = "q" + std::to_string(OA->Descr->F->q);
	}
	else if (ST.stringcmp(Descr->family_name, "LP_37_72") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object LP_37_72" << endl;
		}

		geometry::orthogonal_geometry::orthogonal_global OG;

		OG.create_LP_37_72_BLT_set(
				OA->O,
				Blt_set->set,
				verbose_level);

		str = "LP_ago72";
		str_q = "q" + std::to_string(OA->Descr->F->q);
	}
	else if (ST.stringcmp(Descr->family_name, "LP_37_4a") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object LP_37_4a" << endl;
		}

		geometry::orthogonal_geometry::orthogonal_global OG;

		OG.create_LP_37_4a_BLT_set(
				OA->O,
				Blt_set->set,
				verbose_level);

		str = "LP_ago4a";
		str_q = "q" + std::to_string(OA->Descr->F->q);
	}
	else if (ST.stringcmp(Descr->family_name, "LP_37_4b") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object LP_37_4b" << endl;
		}

		geometry::orthogonal_geometry::orthogonal_global OG;

		OG.create_LP_37_4b_BLT_set(
				OA->O,
				Blt_set->set,
				verbose_level);

		str = "LP_ago4b";
		str_q = "q" + std::to_string(OA->Descr->F->q);
	}
	else if (ST.stringcmp(Descr->family_name, "LP_71") == 0) {
		if (f_v) {
			cout << "BLT_set_create::create_family "
					"creating object LP_71" << endl;
		}

		geometry::orthogonal_geometry::orthogonal_global OG;

		OG.create_Law_71_BLT_set(
				OA->O,
				Blt_set->set,
				verbose_level);

		str = "LP";
		str_q = "q" + std::to_string(OA->Descr->F->q);
	}
	else {
		cout << "BLT_set_create::create_family family name is not recognized" << endl;
		exit(1);
	}

	prefix.assign(str);

	Blt_set->label_txt = prefix + "_" + str_q;
	Blt_set->label_tex = prefix + "\\_" + str_q;

	if (f_v) {
		cout << "BLT_set_create::create_family done" << endl;
	}
}

void BLT_set_create::make_from_catalogue(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue" << endl;
	}
	int nb_iso;
	combinatorics::knowledge_base::knowledge_base K;

	nb_iso = K.BLT_nb_reps(OA->Descr->F->q);

	if (Descr->iso >= nb_iso) {
		cout << "BLT_set_create::make_from_catalogue "
				"iso >= nb_iso, index is out of range" << endl;
		exit(1);
	}


	//set = NEW_lint(OA->Descr->F->q + 1);
	Lint_vec_copy(
			K.BLT_representative(OA->Descr->F->q, Descr->iso),
			Blt_set->set,
			OA->Descr->F->q + 1);



	data_structures_groups::vector_ge *gens;
	std::string target_go_text;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue "
				"gens->stab_BLT_set_from_catalogue" << endl;
	}
	gens->stab_BLT_set_from_catalogue(
			OA->A,
			OA->Descr->F, Descr->iso,
			target_go_text,
			verbose_level - 2);
	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue after "
				"gens->stab_BLT_set_from_catalogue" << endl;
	}

	int c;

	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue "
				"before gens->test_if_in_set_stabilizer" << endl;
	}
	c = gens->test_if_in_set_stabilizer(
			OA->A,
			Blt_set->set, OA->Descr->F->q + 1, verbose_level);
	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue "
				"after gens->test_if_in_set_stabilizer" << endl;
	}

	if (c) {
		if (f_v) {
			cout << "BLT_set_create::make_from_catalogue the generators "
				"stabilize the given BLT set, good" << endl;
		}
	}
	else {
		cout << "BLT_set_create::make_from_catalogue the generators "
				"do not stabilize the given BLT set, bad!" << endl;

		int i;

		for (i = 0; i < gens->len; i++) {
			cout << "BLT_set_create::make_from_catalogue "
					"checking generator " << i << " / " << gens->len << endl;
			OA->A->Group_element->check_if_in_set_stabilizer_debug(
					gens->ith(i),
					OA->Descr->F->q + 1, Blt_set->set, verbose_level);
		}
		exit(1);
	}



	algebra::ring_theory::longinteger_object target_go;

	target_go.create_from_base_10_string(target_go_text);
	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue "
				"target_go = " << target_go << endl;
	}


	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue "
				"before generators_to_strong_generators" << endl;
	}
	OA->A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Sg,
		verbose_level - 3);

	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue "
				"after generators_to_strong_generators" << endl;
	}


	f_has_group = true;

	string str_q;

	str_q = std::to_string(OA->Descr->F->q);

	string str_iso;

	str_iso = std::to_string(Descr->iso);


	prefix = "catalogue_q" + str_q + "_iso" + str_iso;

	Blt_set->label_txt = "catalogue_q" + str_q + "_iso" + str_iso;
	Blt_set->label_tex = "catalogue\\_q" + str_q + "\\_iso" + str_iso;

	if (f_v) {
		cout << "BLT_set_create::make_from_catalogue done" << endl;
	}

}

void BLT_set_create::create_quadratic_field_extension(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::create_quadratic_field_extension done" << endl;
	}

	FQ = NEW_OBJECT(algebra::field_theory::finite_field);

	q = OA->Descr->F->q;
	Q = q * q;

	if (f_v) {
		cout << "BLT_set_create::create_quadratic_field_extension "
				"before FQ->finite_field_init_small_order Q=" << Q << endl;
	}
	FQ->finite_field_init_small_order(
			Q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			verbose_level);
	if (f_v) {
		cout << "BLT_set_create::create_quadratic_field_extension "
				"after FQ->finite_field_init_small_order Q=" << Q << endl;
	}

	if (f_v) {
		cout << "BLT_set_create::create_quadratic_field_extension done" << endl;
	}
}

void BLT_set_create::apply_transformations(
		std::vector<std::string> transform_coeffs,
		std::vector<int> f_inverse_transform, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::apply_transformations done" << endl;
	}
}

void BLT_set_create::report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::report" << endl;
	}

	string fname;

	fname = "BLT_" + Blt_set->label_txt + ".tex";
	ofstream ost(fname);

	report2(ost, verbose_level);


	if (f_v) {
		cout << "BLT_set_create::report done" << endl;
	}
}

void BLT_set_create::export_gap(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::export_gap" << endl;
	}

	string fname;

	fname = "BLT_" + Blt_set->label_txt + ".gap";

	ofstream ost(fname);

	//report2(ost, verbose_level);

	{
		ofstream ost(fname);

		other::orbiter_kernel_system::os_interface Os;
		string str;

		Os.get_date(str);


		ost << "# file " << fname << endl;
		ost << "# created by Orbiter" << endl;
		ost << "# date " << str << endl;
		ost << "#" << endl;

		ost << "LoadPackage(\"fining\");" << endl;


		//ost << "# BLT-set " << label_txt << endl;


		geometry::orthogonal_geometry::quadratic_form *Quadratic_form;

		Quadratic_form = Blt_set_domain_with_action->Blt_set_domain->O->Quadratic_form;


		ost << "# Quadratic form: ";

		Quadratic_form->Poly->print_equation_tex(
				ost, Quadratic_form->the_quadratic_form);
		ost << endl;


		interfaces::l3_interface_gap GAP;


		if (f_v) {
			cout << "BLT_set_create::export_gap "
					"before GAP.export_BLT_set" << endl;
		}
		GAP.export_BLT_set(
				ost,
				Blt_set->label_txt,
				f_has_group,
				Sg,
				OA->A,
				Blt_set_domain_with_action->Blt_set_domain,
				Blt_set->set, verbose_level);
		if (f_v) {
			cout << "BLT_set_create::export_gap "
					"before GAP.export_BLT_set" << endl;
		}


	}


	if (f_v) {
		cout << "BLT_set_create::export_gap done" << endl;
	}
}



void BLT_set_create::create_flock(
		int point_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::create_flock point_idx=" << point_idx << endl;
	}


	geometry::orthogonal_geometry::flock_from_blt_set *Flock;

	Flock = NEW_OBJECT(geometry::orthogonal_geometry::flock_from_blt_set);

	if (f_v) {
		cout << "BLT_set_create::create_flock "
				"before Flock->init" << endl;
	}
	Flock->init(
			BA->Blt_set,
			point_idx,
			verbose_level);
	if (f_v) {
		cout << "BLT_set_create::create_flock "
				"after Flock->init" << endl;
	}

	string str;
	string fname_csv;
	other::orbiter_kernel_system::file_io Fio;


	str = "_flock_pt" + std::to_string(point_idx) + ".csv";
	fname_csv = BA->Blt_set->label_txt + str;

	Fio.Csv_file_support->int_matrix_write_csv(
			fname_csv, Flock->Table_of_ABC->M, Flock->q, 3);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;



	if (f_v) {
		cout << "BLT_set_create::create_flock "
				"before Blt_set_domain_with_action->Blt_set_domain->quadratic_lift" << endl;
	}
	Blt_set_domain_with_action->Blt_set_domain->quadratic_lift(
			Flock->coeff_f, Flock->coeff_g, Flock->nb_coeff, verbose_level);
	if (f_v) {
		cout << "BLT_set_create::create_flock "
				"after Blt_set_domain_with_action->Blt_set_domain->quadratic_lift" << endl;
	}


	if (f_v) {
		cout << "flock::init "
				"before Blt_set_domain_with_action->Blt_set_domain->cubic_lift" << endl;
	}
	Blt_set_domain_with_action->Blt_set_domain->cubic_lift(
			Flock->coeff_f, Flock->coeff_g, Flock->nb_coeff, verbose_level);
	if (f_v) {
		cout << "flock::init "
				"after Blt_set_domain_with_action->Blt_set_domain->cubic_lift" << endl;
	}


	if (f_v) {
		cout << "BLT_set_create::create_flock done" << endl;
	}
}

void BLT_set_create::BLT_test(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::BLT_test" << endl;
	}
	int ret;

	if (f_v) {
		cout << "BLT_set_create::BLT_test "
				"before Blt_set_domain->check_conditions" << endl;
	}
	ret = Blt_set_domain_with_action->Blt_set_domain->check_conditions(
			Blt_set_domain_with_action->Blt_set_domain->target_size,
			Blt_set->set,
			verbose_level);
	if (f_v) {
		cout << "BLT_set_create::BLT_test "
				"after Blt_set_domain->check_conditions" << endl;
		if (ret) {
			cout << "The set passes the BLT-test" << endl;
		}
		else {
			cout << "The set fails the BLT-test" << endl;
		}
	}


	if (f_v) {
		cout << "BLT_set_create::BLT_test done" << endl;
	}
}


void BLT_set_create::plane_invariant(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::plane_invariant" << endl;
	}

	int sz;
	geometry::orthogonal_geometry::orthogonal_plane_invariant *PI;

	sz = Blt_set_domain_with_action->Blt_set_domain->target_size;

	//F = Blt_set_domain_with_action->Blt_set_domain->O->F;


	PI = NEW_OBJECT(geometry::orthogonal_geometry::orthogonal_plane_invariant);

	PI->init(
			Blt_set_domain_with_action->Blt_set_domain->O,
		sz, Blt_set->set,
		verbose_level - 2);

	if (f_v) {
		cout << "BLT_set_create::plane_invariant" << endl;
	}
}


void BLT_set_create::report2(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::report2" << endl;
	}

	int f_book = false;
	int f_title = true;
	string str;

	string title, author, extra_praeamble;

	int f_toc = false;
	int f_landscape = false;
	int f_12pt = false;
	int f_enlarged_page = true;
	int f_pagenumbers = true;
	other::l1_interfaces::latex_interface L;

	str = "BLT-set " + Blt_set->label_tex;
	title = str;
	author.assign("Orbiter");

	L.head(ost, f_book, f_title,
		title,
		author,
		f_toc,
		f_landscape,
		f_12pt,
		f_enlarged_page,
		f_pagenumbers,
		extra_praeamble /* extra_praeamble */);


	OA->O->Quadratic_form->report_quadratic_form(ost, verbose_level - 1);

	if (Blt_set->ABC) {
		print_set_of_points_with_ABC(ost, Blt_set->set, OA->O->F->q + 1);
	}
	else {
		print_set_of_points(ost, Blt_set->set, OA->O->F->q + 1);
	}

	BA->report(ost, verbose_level);

	L.foot(ost);

	if (f_v) {
		cout << "BLT_set_create::report2 done" << endl;
	}
}

void BLT_set_create::print_set_of_points(
		std::ostream &ost, long int *Pts, int nb_pts)
{
	int h, I;
	int *v;
	int n = 4;

	v = NEW_int(n + 1);

	ost << "The BLT-set is:\\\\" << endl;
	for (I = 0; I < (nb_pts + 39) / 40; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & \\mbox{Rank} & \\mbox{Point} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 40; h++) {
			if (I * 40 + h < nb_pts) {

				OA->O->Hyperbolic_pair->unrank_point(v, 1, Pts[I * 40 + h], 0 /* verbose_level */);

				ost << I * 40 + h << " & " << Pts[I * 40 + h] << " & ";
				Int_vec_print(ost, v, n + 1);
				ost << "\\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
	FREE_int(v);
}

void BLT_set_create::print_set_of_points_with_ABC(
		std::ostream &ost, long int *Pts, int nb_pts)
{
	int h, I;
	int *v;
	int n = 4;
	int a, b, c;

	v = NEW_int(n + 1);

	ost << "The BLT-set is:\\\\" << endl;
	for (I = 0; I < (nb_pts + 39) / 40; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & \\mbox{Rank} & \\mbox{Point} & (a,b,c) \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 40; h++) {
			if (I * 40 + h < nb_pts) {

				OA->O->Hyperbolic_pair->unrank_point(v, 1, Pts[I * 40 + h], 0 /* verbose_level */);

				a = Blt_set->ABC[3 * (I * 40 + h) + 0];
				b = Blt_set->ABC[3 * (I * 40 + h) + 1];
				c = Blt_set->ABC[3 * (I * 40 + h) + 2];

				ost << I * 40 + h << " & " << Pts[I * 40 + h] << " & ";
				Int_vec_print(ost, v, n + 1);
				ost << " & ";
				ost << "(";
				ost << a;
				ost << ", ";
				ost << b;
				ost << ", ";
				ost << c;
				ost << ")";
				ost << "\\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
	FREE_int(v);
}



}}}

