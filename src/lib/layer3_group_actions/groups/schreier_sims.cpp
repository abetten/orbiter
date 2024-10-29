// schreier_sims.cpp
//
// Anton Betten
// July 27, 2010

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace groups {



schreier_sims::schreier_sims()
{
	GA = NULL;
	G = NULL;

	f_interested_in_kernel = false;
	KA = NULL;
	K = NULL;

	//longinteger_object G_order, K_order, KG_order;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;

	f_has_target_group_order = false;
	//longinteger_object tgo;

	f_from_generators = false;
	external_gens = NULL;

	f_from_random_process = false;
	callback_choose_random_generator = NULL;
	callback_choose_random_generator_data = NULL;

	f_from_old_G = false;
	old_G = NULL;

	f_has_base_of_choice = false;
	base_of_choice_len = 0;
	base_of_choice = NULL;

	f_override_choose_next_base_point_method = false;
	choose_next_base_point_method = NULL;

	iteration = 0;
}

schreier_sims::~schreier_sims()
{
	if (Elt1) {
		FREE_int(Elt1);
		Elt1 = NULL;
	}
	if (Elt2) {
		FREE_int(Elt2);
		Elt1 = NULL;
	}
	if (Elt3) {
		FREE_int(Elt3);
		Elt1 = NULL;
	}
	if (G) {
		FREE_OBJECT(G);
		G = NULL;
	}
	if (K) {
		FREE_OBJECT(K);
		K = NULL;
	}
	if (f_from_generators) {
		FREE_OBJECT(external_gens);
		external_gens = NULL;
	}
}

void schreier_sims::init(
		actions::action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::init action:" << endl;
		A->print_info();
	}
	schreier_sims::GA = A;
	Elt1 = NEW_int(GA->elt_size_in_int);
	Elt2 = NEW_int(GA->elt_size_in_int);
	Elt3 = NEW_int(GA->elt_size_in_int);

	G = NEW_OBJECT(sims);

	//cout << "schreier_sims::init sims object " << G
	// << " with action " << GA << "=" << GA->label << endl;

	if (f_v) {
		cout << "schreier_sims::init action A:" << endl;
		A->print_info();
		cout << "schreier_sims::init "
				"before G->init" << endl;
	}
	G->init(GA, verbose_level - 2);
	if (f_v) {
		cout << "schreier_sims::init "
				"after G->init" << endl;
	}
	if (f_v) {
		cout << "schreier_sims::init "
				"before G->init_trivial_group" << endl;
	}
	G->init_trivial_group(0 /*verbose_level*/);
	if (f_v) {
		cout << "schreier_sims::init "
				"after G->init_trivial_group" << endl;
	}
	if (f_v) {
		cout << "schreier_sims::init done" << endl;
	}
}

void schreier_sims::interested_in_kernel(
		actions::action *KA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::interested_in_kernel "
				"kernel action:" << endl;
		KA->print_info();
	}
	schreier_sims::KA = KA;
	K = NEW_OBJECT(sims);
	K->init(KA, verbose_level - 2);
	K->init_trivial_group(0);
	f_interested_in_kernel = true;
}


void schreier_sims::init_target_group_order(
		ring_theory::longinteger_object &tgo, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::init_target_group_order " << tgo << endl;
	}
	tgo.assign_to(schreier_sims::tgo);
	f_has_target_group_order = true;
}

void schreier_sims::init_generators(
		data_structures_groups::vector_ge *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "schreier_sims::init_generators " << endl;
	}
	if (f_v) {
		cout << "schreier_sims::init_generators "
				"copying generators in" << endl;
		cout << "schreier_sims::init_generators "
				"number of generators is " << gens->len << endl;
	}
	if (f_v) {
		cout << "schreier_sims::init_generators "
				"before gens->copy" << endl;
	}
	gens->copy(external_gens, verbose_level);
	if (f_v) {
		cout << "schreier_sims::init_generators "
				"after gens->copy" << endl;
	}
	//schreier_sims::gens = gens;
	if (f_v) {
		cout << "schreier_sims::init_generators "
				"generators are:" << endl;
		gens->print_for_make_element(cout);
	}
	if (f_vv) {
		cout << "schreier_sims::init_generators "
				"generators are:" << endl;
		gens->print(cout, false /* f_print_as_permutation */,
				true /* f_offset */, 1 /* offset */,
				true /* f_do_it_anyway_even_for_big_degree */,
				false /* f_print_cycles_of_length_one*/,
				verbose_level - 1);
	}
	f_from_generators = true;
	if (f_v) {
		cout << "schreier_sims::init_generators done" << endl;
	}
}

void schreier_sims::init_random_process(
	void (*callback_choose_random_generator)(int iteration,
			int *Elt, void *data, int verbose_level),
	void *callback_choose_random_generator_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::init_random_process" << endl;
	}
	schreier_sims::callback_choose_random_generator =
			callback_choose_random_generator;
	schreier_sims::callback_choose_random_generator_data =
			callback_choose_random_generator_data;
	f_from_random_process = true;
}

void schreier_sims::init_old_G(
		sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::init_old_G" << endl;
	}
	schreier_sims::old_G = old_G;
	f_from_old_G = true;
}

void schreier_sims::init_base_of_choice(
	int base_of_choice_len, int *base_of_choice,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::init_base_of_choice" << endl;
	}
	schreier_sims::base_of_choice_len = base_of_choice_len;
	schreier_sims::base_of_choice = base_of_choice;
	f_has_base_of_choice = true;
}

void schreier_sims::init_choose_next_base_point_method(
	int (*choose_next_base_point_method)(actions::action *A,
			int *Elt, int verbose_level),
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::init_choose_next_base_point_method" << endl;
	}
	schreier_sims::choose_next_base_point_method =
			choose_next_base_point_method;
	f_override_choose_next_base_point_method = true;
}

void schreier_sims::compute_group_orders()
{
	G->group_order(G_order);
	if (f_interested_in_kernel) {
		ring_theory::longinteger_domain D;
		K->group_order(K_order);
		D.mult(G_order, K_order, KG_order);
	}
	else {
		G_order.assign_to(KG_order);
	}
}

void schreier_sims::print_group_orders()
{
	cout << "current group order is " << G_order;
	if (f_has_target_group_order) {
		cout << " target group order is " << tgo;
	}
	cout << " : in log base 10: "
			<< G_order.log10() << " / " << tgo.log10() << endl;
}

void schreier_sims::get_generator_internal(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier_sims::get_generator_internal "
				"choosing random schreier generator" << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}
	G->random_schreier_generator(Elt, verbose_level - 3);
	//GA->element_move(G->schreier_gen, Elt, 0);
	if (f_vvv) {
		cout << "schreier_sims::get_generator_internal "
				"we picked the following random Schreier generator:" << endl;
		GA->Group_element->element_print_quick(Elt, cout);
		cout << endl;
	}
}

void schreier_sims::get_generator_external(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier_sims::get_generator_external" << endl;
	}
	if (f_from_generators) {
		if (f_v) {
			cout << "schreier_sims::get_generator_external "
					"before get_generator_external_from_generators" << endl;
		}
		get_generator_external_from_generators(Elt, verbose_level);
	}
	else if (f_from_random_process) {
		if (f_v) {
			cout << "schreier_sims::get_generator_external "
					"before get_generator_external_random_process" << endl;
		}
		get_generator_external_random_process(Elt, verbose_level);
	}
	else if (f_from_old_G) {
		if (f_v) {
			cout << "schreier_sims::get_generator_external "
					"before get_generator_external_old_G" << endl;
		}
		get_generator_external_old_G(Elt, verbose_level);
	}
	if (false /*f_vvv*/) {
		cout << "schreier_sims::get_generator_external "
				"we have chosen the following generator" << endl;
		//GA->element_print_quick(Elt, cout);
		cout << endl;
	}
	if (f_v) {
		cout << "schreier_sims::get_generator_external done" << endl;
	}
}

void schreier_sims::get_generator_external_from_generators(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int r;
	orbiter_kernel_system::os_interface Os;
	
	if (f_v) {
		cout << "schreier_sims::get_generator_external_from_generators" << endl;
	}
	if (external_gens->len) {
		r = Os.random_integer(external_gens->len);
		if (f_v) {
			cout << "schreier_sims::get_generator_external_from_generators "
					"choosing generator "
					<< r << " / " << external_gens->len << endl;
		}
		GA->Group_element->element_move(
				external_gens->ith(r), Elt, 0);
	}
	else {
		if (f_v) {
			cout << "schreier_sims::get_generator_external_from_generators "
					"gens->len == 0" << endl;
		}
		// no generators, we are creating the identity group:
		GA->Group_element->element_one(Elt, 0);
	}
	if (f_v) {
		cout << "schreier_sims::get_generator_external_from_generators done" << endl;
	}
}

void schreier_sims::get_generator_external_random_process(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier_sims::get_generator_external_random_process" << endl;
	}
	(*callback_choose_random_generator)((iteration >> 1),
			Elt, callback_choose_random_generator_data, verbose_level - 1);
}

void schreier_sims::get_generator_external_old_G(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (false) {
		cout << "schreier_sims::get_generator_external_old_G" << endl;
	}
	old_G->random_element(Elt, verbose_level - 1);
	if (f_v) {
		cout << "schreier_sims::get_generator_external_old_G "
				"random element chosen, path = ";
		Int_vec_print(cout, old_G->path, old_G->A->base_len());
		cout << endl;
	}
}

void schreier_sims::get_generator(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_sims::get_generator" << endl;
	}
	if ((iteration % 2) == 0) {
		if (f_v) {
			cout << "schreier_sims::get_generator "
					"before get_generator_internal" << endl;
		}
		get_generator_internal(Elt, verbose_level);
		if (f_v) {
			cout << "schreier_sims::get_generator "
					"after get_generator_internal" << endl;
		}
	}
	else if ((iteration % 2) == 1) {
		if (f_v) {
			cout << "schreier_sims::get_generator "
					"before get_generator_external" << endl;
		}
		get_generator_external(Elt, verbose_level);
		if (f_v) {
			cout << "schreier_sims::get_generator "
					"after get_generator_external" << endl;
		}
	}
	if (f_v) {
		cout << "schreier_sims::get_generator done" << endl;
	}
}

void schreier_sims::closure_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object quo, rem;
	int cnt = 0;

	if (f_v) {
		cout << "schreier_sims::closure_group" << endl;
	}
	D.integral_division(tgo, KG_order, quo, rem, 0);

	while (!quo.is_zero() && !rem.is_zero()) {
		if (f_vvv) {
			cout << "schreier_sims::closure_group iteration "
					<< iteration << " cnt " << cnt
					<< ": remainder is not zero, "
					"this is not a subgroup" << endl;
		}
		int nb_times = 30;

		if (f_v) {
			cout << "schreier_sims::closure_group "
					"calling G->closure_group" << endl;
		}
		G->closure_group(nb_times, verbose_level - 3);
		compute_group_orders();
		D.integral_division(tgo, KG_order, quo, rem, 0);
		if (f_vvv) {
			cout << "schreier_sims::closure_group iteration "
					<< iteration << " cnt " << cnt
					<< ": after closure_group: "
					"remaining factor: " << quo
					<< " remainder " << rem << endl;
		}
		cnt++;
		if (cnt == 10) {
			cout << "schreier_sims::closure_group cnt == 100, "
					"KG_order=" << KG_order
					<< ", we are breaking off" << endl;
			break;
		}
	}
	if (f_v) {
		cout << "schreier_sims::closure_group done ";
		print_group_orders();
	}
}

void schreier_sims::create_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	int f_vvv = (verbose_level >= 4);
	int f_vvvv = (verbose_level >= 5);
	
	if (f_v) {
		cout << "schreier_sims::create_group "
				"verbose_level = " << verbose_level << endl;
		if (f_has_target_group_order) {
			cout << "schreier_sims::create_group "
					"target group order is " << tgo << endl;
		}
		else {
			cout << "schreier_sims::create_group "
					"target group order unavailable" << endl;
		}
		cout << "schreier_sims::create_group action GA:" << endl;
		GA->print_info();
		if (f_interested_in_kernel) {
			cout << "schreier_sims::create_group action KA:" << endl;
			KA->print_info();
		}
		else {
			cout << "schreier_sims::create_group not interested in kernel" << endl;
		}
		cout << "schreier_sims::create_group action G->A:" << endl;
		G->A->print_info();

	}
	if (f_has_target_group_order && tgo.is_zero()) {
		cout << "schreier_sims::create_group "
				"target group order is 0" << endl;
		exit(1);
	}


	ring_theory::longinteger_domain D;
	int drop_out_level, image, b, c, f_added, old_base_len;

	int offset = 0;
	int f_do_it_anyway_even_for_big_degree = true;
	int f_print_cycles_of_length_one = true;



	compute_group_orders();
	if (f_v) {
		print_group_orders();
	}
	iteration = 0;
	while (true) {
	
		if (f_vv) {
			cout << "schreier_sims::create_group "
					"iteration " << iteration << endl;
			G->print_generator_depth_and_perm();
		}
		if (f_has_target_group_order && iteration > 50000) {
			cout << "schreier_sims::create_group iteration > 50000, "
					"something seems to be wrong" << endl;
			cout << "target group order = " << tgo << endl;
			cout << "KG_order = " << KG_order << endl;		
			//test_if_subgroup(old_G, 2);
			exit(1);
		}

		if (!f_has_target_group_order && iteration == 10000) {
			if (f_v) {
				cout << "schreier_sims::create_group "
						"iteration == 1000, we seem to be done" << endl;
			}
			break;
		}

		if (f_vv) {
			cout << "schreier_sims::create_group "
					"iteration " << iteration
					<< " before get_generator" << endl;
			//GA->element_print_quick(Elt1, cout);
		}
		get_generator(Elt1, verbose_level - 3);
		if (f_vv) {
			cout << "schreier_sims::create_group "
					"iteration " << iteration
					<< " after get_generator" << endl;
		}
		if (f_vv) {
			cout << "schreier_sims::create_group "
					"iteration " << iteration
					<< " generator: " << endl;
			GA->Group_element->element_print_quick(Elt1, cout);

			GA->Group_element->element_print_for_make_element(Elt1, cout);
			cout << endl;

			GA->Group_element->element_print_as_permutation_with_offset(
					Elt1, cout,
				offset, f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one,
				0/*verbose_level*/);
			cout << endl;

		}

		if (f_vv) {
			cout << "schreier_sims::create_group "
					"calling strip:" << endl;
		}
		if (G->strip(Elt1, Elt2, drop_out_level,
				image, verbose_level - 2)) {
			if (f_vv) {
				cout << "schreier_sims::create_group: "
						"element strips through" << endl;
				if (f_vvvv) {
					cout << "schreier_sims::create_group: "
							"residue = " << endl;
					GA->Group_element->element_print_quick(Elt2, cout);
					cout << endl;

					GA->Group_element->element_print_for_make_element(Elt2, cout);
					cout << endl;

					GA->Group_element->element_print_as_permutation_with_offset(
							Elt2, cout,
						offset, f_do_it_anyway_even_for_big_degree,
						f_print_cycles_of_length_one,
						0/*verbose_level*/);
					cout << endl;

				}
			}
			f_added = false;
			if (!GA->Group_element->element_is_one(Elt2, 0)) {
				if (f_vvv) {
					cout << "schreier_sims::create_group: "
							"the residue is not trivial, "
							"we need to choose another base point" << endl;
				}
				if (f_override_choose_next_base_point_method) {
					if (f_vv) {
						cout << "schreier_sims::create_group "
								"before (*choose_next_base_point_method)" << endl;
					}
					b = (*choose_next_base_point_method)(GA,
							Elt2, verbose_level - 5);
				}
				else {
					if (f_vv) {
						cout << "schreier_sims::create_group "
								"before GA->choose_next_base_point_default_method" << endl;
					}
					b = GA->choose_next_base_point_default_method(
							Elt2, verbose_level - 5);
				}

				if (f_vv) {
					cout << "schreier_sims::create_group "
							"next suggested base point is "
							<< b << endl;
				}
				if (b == -1) {
					if (f_vv) {
						cout << "schreier_sims::create_group "
								"cannot find next base point" << endl;
					}
					if (K->strip(
							Elt2, Elt3, drop_out_level,
							image, 0/*verbose_level - 3*/)) {
						if (f_vv) {
							cout << "schreier_sims::create_group: "
									"element strips through kernel" << endl;
							if (f_vvvv) {
								cout << "schreier_sims::create_group: "
										"residue = " << endl;
								//KA->element_print_quick(Elt3, cout);
								cout << endl;
								K->print(false);
								K->print_basic_orbits();
								cout << "schreier_sims::create_group "
										"residue" << endl;
								KA->Group_element->element_print_image_of_set(
										Elt3, KA->base_len(), KA->get_base());
								cout << "schreier_sims::create_group "
										"Elt2" << endl;
								KA->Group_element->element_print_image_of_set(
										Elt2, KA->base_len(), KA->get_base());
							}
						}
						if (!KA->Group_element->element_is_one(Elt3, 0)) {
							cout << "schreier_sims::create_group "
									"element strips through kernel, "
									"residue = " << endl;
							cout << "but the element is not the "
									"identity, something is wrong" << endl;
							//GA->element_print(Elt3, cout);
							cout << endl;
							compute_group_orders();
							print_group_orders();

							exit(1);
						}
					}
					if (f_vv) {
						cout << "schreier_sims::create_group "
								"before K->add_generator_at_level" << endl;
					}
					K->add_generator_at_level(Elt3,
							drop_out_level, 0 /*verbose_level - 3*/);
					if (f_vvv) {
						cout << "schreier_sims::create_group "
								"the residue has been added as "
								"kernel generator at level "
								<< drop_out_level << endl;
					}
					f_added = true;
				}
				else {
					if (f_vvv) {
						cout << "schreier_sims::create_group "
								"choosing another base point " << b << endl;
					}
					old_base_len = GA->base_len();
					GA->Stabilizer_chain->reallocate_base(b, verbose_level);
					if (f_vvv) {
						//cout << "after reallocate_base 1" << endl;
					}
					G->reallocate_base(old_base_len, verbose_level - 1);
					if (f_vvv) {
						//cout << "after reallocate_base 2" << endl;
					}
					if (f_vv) {
						cout << "schreier_sims::create_group "
								"A new base point " << b
							<< " chosen, new base has length "
							<< GA->base_len() << endl;
						cout << "schreier_sims::create_group: "
								"calling add_generator_at_level" << endl;
					}
					G->add_generator_at_level(Elt2,
							GA->base_len() - 1, 0 /*verbose_level - 3*/);
					if (f_vv) {
						cout << "schreier_sims::create_group: "
								"the residue has been added "
								"at level " << GA->base_len() - 1 << endl;
					}
				} // if b
			} // if ! element is one
			else {
				if (f_vv) {
					cout << "schreier_sims::create_group: "
							"the residue is trivial" << endl;
				}
			}
			//G->closure_group(10, verbose_level - 2);
		}
		else {
			f_added = true;
			if (f_vv) {
				cout << "schreier_sims::create_group: "
						"residue needs to be inserted at level = "
					<< drop_out_level << " with base point "
					<< GA->Stabilizer_chain->base_i(drop_out_level) << " mapped to "
					<< image << endl;

				GA->Group_element->element_print(Elt2, cout);
				cout  << endl;

				GA->Group_element->element_print_for_make_element(Elt2, cout);
				cout << endl;

				GA->Group_element->element_print_as_permutation_with_offset(
						Elt2, cout,
					offset, f_do_it_anyway_even_for_big_degree,
					f_print_cycles_of_length_one,
					0/*verbose_level*/);
				cout << endl;


			}
			if (f_vv) {
				cout << "schreier_sims::create_group: "
						"before G->add_generator_at_level" << endl;
			}
			G->add_generator_at_level(Elt2,
					drop_out_level, 0 /*verbose_level - 3*/);
			if (f_vv) {
				cout << "schreier_sims::create_group: "
						"after G->add_generator_at_level" << endl;
			}
		}
		
		compute_group_orders();


		if ((f_v && f_added) || f_vv) {
			cout << "schreier_sims::create_group: "
					"updated group order is ";
			print_group_orders();
		}
		iteration++;

		if (f_has_target_group_order) {
			c = D.compare(tgo, KG_order);
			if (c == 0) {
				if (f_v) {
					cout << "schreier_sims::create_group: "
							"reached the full group after "
							<< iteration << " iterations" << endl;
				}
				break;
			}
			if (c < 0) {
				if (true) {
					cout << "schreier_sims::create_group "
							"overshooting the expected group after "
							<< iteration << " iterations" << endl;
					print_group_orders();
					if (KG_order.as_int() < 100) {
						cout << "schreier_sims::create_group so far, "
								"the group elements are:" << endl;
						//G->print_all_group_elements();
					}
				}
				//break;
				exit(1);
			}
			else {
				if (f_vv) {
					cout << "schreier_sims::create_group: "
							"before closure_group" << endl;
				}
				closure_group(0 /*verbose_level - 2*/);
				if (f_vv) {
					cout << "schreier_sims::create_group: "
							"after closure_group" << endl;
				}
			}
		}
		else {
			if (f_vv) {
				cout << "schreier_sims::create_group: "
						"before closure_group" << endl;
			}
			closure_group(0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "schreier_sims::create_group: "
						"after closure_group" << endl;
			}
		}
	}
	if (f_v) {
		cout << "schreier_sims::create_group finished:";
		print_group_orders();

		cout << "the new action has base ";
		Lint_vec_print(cout, GA->get_base(), GA->base_len());
		cout << " of length " << GA->base_len()  << endl;
	}
}

}}}


