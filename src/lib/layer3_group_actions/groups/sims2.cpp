// sims2.cpp
//
// Anton Betten
// January 11, 2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


void sims::build_up_group_random_process(
		sims *K,
	sims *old_G,
	ring_theory::longinteger_object &target_go,
	int f_override_choose_next_base_point,
	int (*choose_next_base_point_method)(actions::action *A,
			int *Elt, int verbose_level),
	int verbose_level)
// called from action_global::induce
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 6);
	int f_v4 = (verbose_level >= 7);

	if (f_v) {
		cout << "sims::build_up_group_random_process "
				"verbose_level=" << verbose_level << endl;
	}
	if (f_v) {
		cout << "sims::build_up_group_random_process "
				"A->action =" << A->label << endl;
		cout << "sims::build_up_group_random_process "
				"kernel action =" << K->A->label << endl;
		cout << "sims::build_up_group_random_process "
				"old_G action =" << old_G->A->label << endl;
	}

	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go, G_order, K_order, KG_order, quo, rem;
	int drop_out_level, image, cnt, b, c, old_base_len;
	actions::action *GA;
	actions::action *KA;
	int *Elt;

	GA = A;
	KA = K->A;
	Elt = NEW_int(A->elt_size_in_int);

	group_order(G_order);
	K->group_order(K_order);
	D.mult(G_order, K_order, KG_order);

	if (f_v) {
		cout << "sims::build_up_group_random_process: "
				"current group order is " << G_order
				<< " target " << target_go << endl;
		cout << "sims::build_up_group_random_process "
				"the old_G action " << old_G->A->label
				<< " has base_length = " << old_G->A->base_len()
			<< " and degree " << old_G->A->degree << endl;
		cout << "sims::build_up_group_random_process "
				"the kernel action " << KA->label
				<< " has base_length = " << KA->base_len()
			<< " and degree " << KA->degree << endl;
		cout << "sims::build_up_group_random_process "
				"the image action has base_length = " << GA->base_len()
			<< " and degree " << GA->degree << endl;
		cout << "sims::build_up_group_random_process "
				"current action " << GA->label << endl;
		cout << "sims::build_up_group_random_process "
				"current group order = " << G_order << endl;
		cout << "sims::build_up_group_random_process "
				"current kernel order = " << K_order << endl;
		cout << "sims::build_up_group_random_process "
				"together = " << KG_order << endl;
		cout << "sims::build_up_group_random_process "
				"target_go = " << target_go << endl;
	}
	cnt = 0;
	while (true) {

		if (f_vv) {
			cout << "sims::build_up_group_random_process "
					"iteration " << cnt << endl;
		}
		if (cnt > 1000) {
			cout << "sims::build_up_group_random_process "
					"cnt > 1000, something seems to be wrong" << endl;
			test_if_subgroup(old_G, 2);
			exit(1);
		}
		if (f_v4) {
			old_G->A->print_base();
			old_G->print_orbit_len();
		}
		if ((cnt % 2) == 0) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"choosing random schreier generator" << endl;
			}
			random_schreier_generator(Elt, 0 /*verbose_level - 5*/);
			A->Group_element->element_move(Elt, GA->Group_element->Elt1, 0);
			if (f_v4) {
				cout << "sims::build_up_group_random_process "
						"random element chosen:" << endl;
				A->Group_element->element_print_quick(GA->Group_element->Elt1, cout);
				cout << endl;
			}
		}
		else if ((cnt % 2) == 1) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"choosing random element in the group by "
						"which we extend" << endl;
			}
			old_G->random_element(GA->Group_element->Elt1, 0 /*verbose_level - 5*/);
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"random element chosen, path = ";
				Int_vec_print(cout, old_G->path, old_G->A->base_len());
				cout << endl;
			}
			if (f_v4) {
				GA->Group_element->element_print_quick(GA->Group_element->Elt1, cout);
				cout << endl;
			}
		}

		int strip_result;

		if (f_v4) {
			cout << "sims::build_up_group_random_process "
					"before strip" << endl;
		}
		strip_result = strip(
				GA->Group_element->Elt1,
				GA->Group_element->Elt2,
				drop_out_level, image,
				0 /*verbose_level - 3*/);
		if (f_v4) {
			cout << "sims::build_up_group_random_process "
					"after strip, strip_result = " << strip_result << endl;
		}

		if (strip_result) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"element strips through Sims" << endl;
				cout << "sims::build_up_group_random_process my_base_len=" << my_base_len << endl;
				if (f_vv) {
					cout << "sims::build_up_group_random_process "
							"residue = " << endl;
					GA->Group_element->element_print_quick(
							GA->Group_element->Elt2, cout);
					cout << endl;
				}
			}
			//f_added = false;
			if (!GA->Group_element->element_is_one(
					GA->Group_element->Elt2, 0)) {
				if (f_vvv) {
					cout << "sims::build_up_group_random_process "
							"the residue is not trivial, we need to "
							"choose another base point" << endl;
				}
				if (f_override_choose_next_base_point) {
					b = (*choose_next_base_point_method)(
							GA,
							GA->Group_element->Elt2,
							verbose_level);
				}
				else {
					b = GA->choose_next_base_point_default_method(
							GA->Group_element->Elt2,
							verbose_level);
				}

				if (f_vv) {
					cout << "sims::build_up_group_random_process "
							"suggested next base point " << b << endl;
				}
				if (b == -1) {

					int K_strip_result;

					if (f_vv) {
						cout << "sims::build_up_group_random_process "
								"cannot find next base point" << endl;
					}
					if (f_vv) {
						cout << "sims::build_up_group_random_process "
								"before K->strip" << endl;
					}
					K_strip_result = K->strip(
							GA->Group_element->Elt2,
							GA->Group_element->Elt3,
							drop_out_level, image,
							verbose_level - 3);
					if (f_vv) {
						cout << "sims::build_up_group_random_process "
								"after K->strip, K_strip_result = " << K_strip_result << endl;
					}
					if (K_strip_result) {
						if (f_vv) {
							cout << "sims::build_up_group_random_process "
									"element strips through kernel" << endl;
							if (f_v4) {
								cout << "sims::build_up_group_random_process "
										"residue = " << endl;
								KA->Group_element->element_print_quick(
										GA->Group_element->Elt3, cout);
								cout << endl;
								K->print(false);
								K->print_basic_orbits();
								cout << "sims::build_up_group_random_process "
										"residue" << endl;
								KA->Group_element->element_print_image_of_set(
										GA->Group_element->Elt3, KA->base_len(), KA->get_base());
								cout << "sims::build_up_group_random_process "
										"Elt2" << endl;
								KA->Group_element->element_print_image_of_set(
										GA->Group_element->Elt2, KA->base_len(), KA->get_base());
							}
						}
						if (!KA->Group_element->element_is_one(
								GA->Group_element->Elt3,
								0)) {
							cout << "sims::build_up_group_random_process "
									"element strips through kernel, "
									"residue = " << endl;
							cout << "but the element is not the identity, "
									"something is wrong" << endl;
							GA->Group_element->element_print(GA->Group_element->Elt3, cout);
							cout << endl;

							cout << "sims::build_up_group_random_process "
									"current group order is " << G_order
									<< " target " << target_go << endl;
							cout << "sims::build_up_group_random_process "
									"the old_G action " << old_G->A->label
									<< " has base_length = "
									<< old_G->A->base_len()
								<< " and degree " << old_G->A->degree << endl;
							cout << "sims::build_up_group_random_process "
									"the kernel action " << KA->label
									<< " has base_length = " << KA->base_len()
								<< " and degree " << KA->degree << endl;
							cout << "sims::build_up_group_random_process "
									"the image action has base_length = "
								<< GA->base_len()
								<< " and degree " << GA->degree << endl;
							cout << "sims::build_up_group_random_process "
									"current action " << GA->label << endl;
							cout << "sims::build_up_group_random_process "
									"current group order = "
								<< G_order << endl;
							cout << "sims::build_up_group_random_process "
									"current kernel order = "
								<< K_order << endl;
							cout << "sims::build_up_group_random_process "
									"together = " << KG_order << endl;
							cout << "sims::build_up_group_random_process "
									"target_go = " << target_go << endl;

							exit(1);
						}
					}
					else {
						if (f_vv) {
							cout << "sims::build_up_group_random_process "
									" K_strip drops out at level drop_out_level=" << drop_out_level << endl;
						}
						if (f_vv) {
							cout << "sims::build_up_group_random_process "
									"before K->add_generator_at_level " << endl;
						}
						K->add_generator_at_level(
								GA->Group_element->Elt3,
								drop_out_level,
								0 /*verbose_level - 3*/);
						if (f_vv) {
							cout << "sims::build_up_group_random_process "
									"after K->add_generator_at_level " << endl;
						}
						if (f_vvv) {
							cout << "sims::build_up_group_random_process "
									"the residue has been added as kernel "
									"generator at level " << drop_out_level
									<< endl;
						}
					}
					//f_added = true;
				}
				else {
					if (f_vvv) {
						cout << "sims::build_up_group_random_process "
								"choosing additional base point " << b << endl;
					}
					old_base_len = GA->base_len();
					GA->Stabilizer_chain->reallocate_base(b, verbose_level);
					if (f_v) {
						cout << "sims::build_up_group_random_process "
								"before reallocate_base" << endl;
					}
					reallocate_base(old_base_len, verbose_level - 1);
					if (f_v) {
						cout << "sims::build_up_group_random_process "
								"after reallocate_base" << endl;
					}
					if (f_vv) {
						cout << "sims::build_up_group_random_process "
								"additional base point " << b
							<< " chosen, increased base has length "
							<< GA->base_len() << endl;
						cout << "sims::build_up_group_random_process "
								"calling add_generator_at_level" << endl;
					}
					if (f_v) {
						cout << "sims::build_up_group_random_process "
								"before add_generator_at_level" << endl;
					}
					add_generator_at_level(
							GA->Group_element->Elt2,
							GA->base_len() - 1,
							0 /*verbose_level - 10*/);
					if (f_v) {
						cout << "sims::build_up_group_random_process "
								"after add_generator_at_level" << endl;
					}
					if (f_vv) {
						cout << "sims::build_up_group_random_process "
								"the residue has been added at level "
								<< GA->base_len() - 1 << endl;
					}
				} // if b
			} // if ! element is one
			else {
				if (f_vv) {
					cout << "sims::build_up_group_random_process "
							"the residue is trivial" << endl;
				}
			}
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"before closure_group" << endl;
			}
			//closure_group(10, verbose_level);
			closure_group(10, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"after closure_group" << endl;
			}
		}
		else {
			//f_added = true;
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"strip drops out, we need to insert the element at level = "
					<< drop_out_level << " with image "
					<< image << endl;
				if (f_vvv) {
					GA->Group_element->element_print(
							GA->Group_element->Elt2, cout);
					cout  << endl;
				}
			}
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"before add_generator_at_level" << endl;
			}
			add_generator_at_level(GA->Group_element->Elt2, drop_out_level,
					0/*verbose_level - 3*/);
			if (f_vv) {
				cout << "sims::build_up_group_random_process "
						"after add_generator_at_level" << endl;
			}
		}

		if (f_vv) {
			cout << "sims::build_up_group_random_process: "
					"computing group order G" << endl;
		}
		group_order(G_order);
		if (f_vv) {
			cout << "sims::build_up_group_random_process "
					"G_order=" << G_order << endl;
		}
		K->group_order(K_order);
		if (f_vv) {
			cout << "sims::build_up_group_random_process "
					"K_order=" << K_order << endl;
		}
		//cout << "K tl: ";
		//int_vec_print(cout, K->orbit_len, K->A->base_len);
		//cout << endl;
		//cout << "K action " << K->A->label << endl;
		D.mult(G_order, K_order, KG_order);
		if (f_v /* (f_v && f_added) || f_vv */) {
			cout << "sims::build_up_group_random_process "
					"current group order is " << KG_order
				<< " = " << G_order << " * " << K_order << endl;
		}
		if (f_vv) {
			cout << "sims::build_up_group_random_process ";
			print_transversal_lengths();
		}
		if (false) {
			cout << "sims::build_up_group_random_process "
					"before D.compare" << endl;
		}
		c = D.compare(target_go, KG_order);
		if (false) {
			cout << "sims::build_up_group_random_process "
					"after D.compare c=" << c
					<< " cnt=" << cnt << endl;
		}
		cnt++;
		if (c == 0) {
			if (f_v) {
				cout << "sims::build_up_group_random_process "
						"reached the full group after "
						<< cnt << " iterations" << endl;
			}
			break;
		}
		if (c < 0) {
			if (true) {
				cout << "sims::build_up_group_random_process "
						"overshooting the expected group after "
						<< cnt << " iterations" << endl;
				cout << "sims::build_up_group_random_process "
						"current group order is " << KG_order
					<< " = |G| * |K| = " << G_order << " * "
					<< K_order << ", target_go=" << target_go << endl;
			}
			//break;
			exit(1);
		}
	} // while true
	FREE_int(Elt);
	if (f_vv) {
		cout << "sims::build_up_group_random_process "
				"finished: "
				"found a group of order " << KG_order
			<< " = " << G_order << " * " << K_order << endl;
		if (f_vvv) {
			cout << "sims::build_up_group_random_process "
					"the new action has base_length = "
				<< GA->base_len()
				<< " and degree " << GA->degree << endl;
			print_transversal_lengths();
			if (false) {
				print_transversals();
			}
			if (false) {
				print(false);
			}
		}
	}
	if (f_v) {
		cout << "sims::build_up_group_random_process done" << endl;
	}
}

#if 0
void sims::build_up_group_from_generators(
		sims *K,
		data_structures_groups::vector_ge *gens,
	int f_target_go,
	ring_theory::longinteger_object *target_go,
	int f_override_choose_next_base_point,
	int (*choose_next_base_point_method)(actions::action *A,
			int *Elt, int verbose_level),
	int verbose_level)
// called from known_groups::init_group_from_strong_generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "sims::build_up_group_from_generators "
				"verbose_level = " << verbose_level << endl;
	}

	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object G_order, K_order, KG_order;
	int drop_out_level, image, f_added, j;
	int level, base_point, b, old_base_len;
	actions::action *GA;
	actions::action *KA;
	data_structures_groups::vector_ge subset_of_gens;

	GA = A;
	KA = K->A;

	if (f_v) {
		cout << "sims::build_up_group_from_generators base: ";
		Lint_vec_print(cout, GA->get_base(), GA->base_len());
		cout << endl;

#if 0
		cout << "generators:" << endl;
		gens->print(cout);
		cout << endl;
#endif

		if (f_target_go) {
			cout << "sims::build_up_group_from_generators "
					"target group order: " << *target_go << endl;
		}
		else {
			cout << "sims::build_up_group_from_generators "
					"no target group order given" << endl;
		}
		cout << "sims::build_up_group_from_generators "
				"verbose_level=" << verbose_level << endl;
	}
	group_order(G_order);
	K->group_order(K_order);
	D.mult(G_order, K_order, KG_order);
	for (level = GA->base_len() - 1; level >= 0; level--) {
		base_point = GA->base_i(level);
		if (f_vv) {
			cout << "sims::build_up_group_from_generators "
					"level " << level << " base point "
					<< base_point << endl;
		}
		GA->find_strong_generators_at_level(
			GA->base_len(), GA->get_base(), level,
			*gens, subset_of_gens, verbose_level - 3);

		{
			schreier O;

			if (f_v) {
				cout << "sims::build_up_group_from_generators "
						"calling O.init" << endl;
			}

			O.init(GA, verbose_level - 2);

			if (f_v) {
				cout << "sims::build_up_group_from_generators "
						"calling O.init_generators" << endl;
			}
			O.init_generators(subset_of_gens, verbose_level - 2);

			if (f_vvv) {
				cout << "sims::build_up_group_from_generators "
						"generators in schreier" << endl;
				O.print_generators();
			}

			if (f_vv) {
				cout << "sims::build_up_group_from_generators "
						"computing orbit of point " << base_point << endl;
			}
			O.compute_point_orbit(base_point, 0);
			if (f_vv) {
				cout << "sims::build_up_group_from_generators "
						"point " << base_point << " lies in an orbit "
						"of size " << O.orbit_len[0] << endl;
				if (false) {
					O.print(cout);
					O.print_tables(cout, false);
				}
			}
			for (j = 0; j < O.orbit_len[0]; j++) {
				if (false) {
					cout << "sims::build_up_group_from_generators "
							"level " << level << " coset rep " << j << endl;
				}
				O.coset_rep(j, 0 /* verbose_level */);
				if (false) {
					GA->Group_element->element_print(O.cosetrep, cout);
					cout << endl;
				}
				if (strip(
						O.cosetrep,
						GA->Group_element->Elt2 /* residue */,
						drop_out_level, image,
						0 /*verbose_level - 1*/)) {
					if (f_vv) {
						cout << "sims::build_up_group_from_generators "
								"element strips through" << endl;
						if (false /*f_vvv */) {
							cout << "sims::build_up_group_from_generators "
									"residue=" << endl;
							GA->Group_element->element_print_quick(
									GA->Group_element->Elt2, cout);
							cout << endl;
						}
					}
					if (false) {
						cout << "sims::build_up_group_from_generators "
								"element strips through." << endl;
						cout << "if it is the identity element, that's OK,"
								<< endl;
						cout << "sims::build_up_group_from_generators "
								"otherwise please add another base point,"
								<< endl;
						cout << "sims::build_up_group_from_generators "
								"a point which is moved by the residue"
								<< endl;
						GA->Group_element->element_print(GA->Group_element->Elt2, cout);
					}
					if (!GA->Group_element->element_is_one(GA->Group_element->Elt2, false)) {
						if (f_vvv) {
							cout << "sims::build_up_group_from_generators "
									"the residue is not trivial, "
									"we need to chose another base point"
									<< endl;
						}
						if (f_override_choose_next_base_point) {
							b = (*choose_next_base_point_method)(
									GA, GA->Group_element->Elt2, verbose_level - 3);
						}
						else {
							b = GA->choose_next_base_point_default_method(
									GA->Group_element->Elt2, verbose_level - 3);
						}
						if (b == -1) {
							if (f_vv) {
								cout << "sims::build_up_group_from_generators: "
										"cannot find next base point" << endl;
							}
							if (K->strip(
									GA->Group_element->Elt2,
									GA->Group_element->Elt3,
									drop_out_level, image, verbose_level - 3)) {
								if (f_vv) {
									cout << "sims::build_up_group_from_generators "
											"element strips through kernel, "
											"residue = " << endl;
									if (f_vv) {
										KA->Group_element->element_print(
												GA->Group_element->Elt3, cout);
										cout << endl;
										}
									K->print(false);
									K->print_basic_orbits();
									cout << "sims::build_up_group_from_generators "
											"residue" << endl;
									KA->Group_element->element_print_image_of_set(
											GA->Group_element->Elt3, KA->base_len(), KA->get_base());
									cout << "sims::build_up_group_from_generators Elt2" << endl;
									KA->Group_element->element_print_image_of_set(
											GA->Group_element->Elt2, KA->base_len(), KA->get_base());
								}
								if (!KA->Group_element->element_is_one(GA->Group_element->Elt3, false)) {
									cout << "sims::build_up_group_from_generators "
											"but the element is not the identity, "
											"something is wrong" << endl;
									GA->Group_element->element_print(GA->Group_element->Elt3, cout);
									cout << endl;
									exit(1);
								}
							}
							K->add_generator_at_level(
									GA->Group_element->Elt3,
									drop_out_level, verbose_level - 3);
							if (f_vv) {
								cout << "sims::build_up_group_from_generators "
										"the residue has been added as "
										"kernel generator at level "
										<< drop_out_level << endl;
							}
							f_added = true;
						}
						else {
							if (f_vv) {
								cout << "sims::build_up_group_from_generators: "
										"choosing additional base point "
										<< b << endl;
							}
							old_base_len = GA->base_len();
							GA->Stabilizer_chain->reallocate_base(b, verbose_level - 3);
							if (f_vv) {
								//cout << "after reallocate_base 1" << endl;
							}
							reallocate_base(old_base_len, verbose_level - 3);
							if (f_vv) {
								//cout << "after reallocate_base 2" << endl;
							}
							if (f_v) {
								cout << "sims::build_up_group_from_generators "
										"additional base point " << b
									<< " has been chosen, the increased base has length "
									<< GA->base_len() << endl;
								cout << "sims::build_up_group_from_generators "
										"calling add_generator_at_level" << endl;
							}
							add_generator_at_level(
									GA->Group_element->Elt2,
									GA->base_len() - 1, verbose_level - 3);
							if (f_vv) {
								cout << "sims::build_up_group_from_generators "
										"the residue has been added at level "
										<< GA->base_len() - 1 << endl;
							}
						} // if b
					} // if ! element is one
					else {
						if (f_vv) {
							cout << "sims::build_up_group_from_generators "
									"the residue is trivial" << endl;
						}
					}

					f_added = false;
				}
				else {
					f_added = true;
					if (f_vv) {
						cout << "sims::build_up_group_from_generators "
								"before add_generator_at_level" << endl;
					}
					add_generator_at_level(
							GA->Group_element->Elt2,
							drop_out_level,
							0 /*verbose_level - 1*/);
					if (f_vv) {
						cout << "sims::build_up_group_from_generators "
								"after add_generator_at_level" << endl;
					}
				}

				group_order(G_order);
				K->group_order(K_order);
				D.mult(G_order, K_order, KG_order);


				if (f_v && f_added) {
					cout << "sims::build_up_group_from_generators "
							"level " << level << " coset " << j
						<< " group of order increased to " << KG_order
						<< " = " << G_order << " * " << K_order << endl;
				}
				if (f_vv) {
					cout << "sims::build_up_group_from_generators "
							"level " << level << " coset " << j
						<< " found a group of order " << KG_order
						<< " = " << G_order << " * " << K_order << endl;
				}
			}
		} // end of schreier

	} // next level


	if (f_target_go) {
		int c, cnt;

		cnt = 0;
		while (true) {
			group_order(G_order);
			K->group_order(K_order);
			D.mult(G_order, K_order, KG_order);

			c = D.compare(*target_go, KG_order);
			cnt++;
			if (c == 0) {
				if (f_v) {
					cout << "sims::build_up_group_from_generators "
							"reached the full group after "
							<< cnt << " iterations" << endl;
				}
				break;
			}
			if (c < 0) {
				if (true) {
					cout << "sims::build_up_group_from_generators "
							"overshooting the expected group after "
							<< cnt << " iterations" << endl;
					cout << "current group order is " << KG_order
						<< " = " << G_order << " * " << K_order << endl;
				}
				//break;
				exit(1);
			}
			if (cnt > 10000) {
				cout << "sims::build_up_group_from_generators after "
						<< cnt << " iterations, we seem to be having "
						"problems reaching the target group order" << endl;
				cout << "sims::build_up_group_from_generators "
						"group order = " << KG_order << endl;
				cout << "sims::build_up_group_from_generators "
						"target group order = " << *target_go << endl;
				exit(1);
			}

			if (f_vv) {
				cout << "sims::build_up_group_from_generators "
						"calling closure group" << endl;
			}
			closure_group(10, verbose_level - 2);

		}
	}

	if (f_v) {
		cout << "sims::build_up_group_from_generators finished: "
				"found a group of order " << KG_order
			<< " = " << G_order << " * " << K_order << endl;
		cout << "sims::build_up_group_from_generators "
				"the new action has base_length = " << GA->base_len()
			<< " and degree " << GA->degree << endl;
		print_transversal_lengths();

#if 0
		if (f_vv) {
			print_transversals();
		}
		if (f_vvv) {
			print(false);
		}
#endif
	}
	if (f_v) {
		cout << "sims::build_up_group_from_generators "
				"found a group of order " << G_order << endl;
	}
	if (f_v) {
		cout << "sims::build_up_group_from_generators done" << endl;
	}
}
#endif


void sims::build_up_subgroup_random_process(
		sims *G,
	void (*choose_random_generator_for_subgroup)(sims *G,
			int *Elt, int verbose_level),
	int verbose_level)
// called from wreath_product::orbits_restricted_compute
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	//int f_vvvv = (verbose_level >= 10);

	if (f_v) {
		cout << "sims::build_up_subgroup_random_process" << endl;
		cout << "sims::build_up_subgroup_random_process verbose_level = " << verbose_level << endl;
	}

	ring_theory::longinteger_object go, G_order, quo, rem;
	int drop_out_level, image, cnt, f_added;
	actions::action *GA;
	
	GA = A;
	

	G->group_order(G_order);
	group_order(go);
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process(): "
				"old group order is " << G_order << endl;
		cout << "the group is in action " << G->A->label
				<< " with base_length = " << G->A->base_len()
			<< " and degree " << G->A->degree << endl;
		cout << "the image action has base_length = " << GA->base_len()
			<< " and degree " << GA->degree << endl;
		cout << "current action " << GA->label << endl;
		cout << "current group order = " << go << endl;
	}
	cnt = 0;
	while (cnt < 200) {
	
		if (f_vv) {
			cout << "sims::build_up_subgroup_random_process iteration " << cnt << endl;
		}
#if 0
		if (cnt > 1000) {
			cout << "sims::build_up_subgroup_random_process "
					"cnt > 1000, something seems to be wrong" << endl;
			test_if_subgroup(G, 2);
			exit(1);
		}
#endif
		if (false) {
			G->A->print_base();
			G->print_orbit_len();
		}
		if ((cnt % 2) == 0) {
			if (f_vvv) {
				cout << "sims::build_up_subgroup_random_process "
						"choosing random Schreier generator" << endl;
			}
			random_schreier_generator(
					GA->Group_element->Elt1,
					0/*verbose_level - 3*/);
			//A->element_move(schreier_gen, GA->Elt1, 0);
			if (false) {
				cout << "sims::build_up_subgroup_random_process "
						"random element chosen:" << endl;
				A->Group_element->element_print(GA->Group_element->Elt1, cout);
				cout << endl;
			}
		}
		else if ((cnt % 2) == 1) {
			if (f_vvv) {
				cout << "sims::build_up_subgroup_random_process "
						"choosing random element in the "
						"group by which we extend" << endl;
			}
			(*choose_random_generator_for_subgroup)(
					G,
					GA->Group_element->Elt1, 0 /*verbose_level - 1*/);
			if (false) {
				cout << "sims::build_up_subgroup_random_process "
						"random element chosen" << endl;
			}
			if (false) {
				GA->Group_element->element_print(
						GA->Group_element->Elt1, cout);
				cout << endl;
			}
		}
		if (strip(GA->Group_element->Elt1, GA->Group_element->Elt2, drop_out_level,
				image, 0/*verbose_level*/)) {
			if (f_vvv) {
				cout << "sims::build_up_subgroup_random_process "
						"element strips through" << endl;
				if (false) {
					cout << "sims::build_up_subgroup_random_process "
							"residue = " << endl;
					GA->Group_element->element_print(
							GA->Group_element->Elt2, cout);
					cout << endl;
				}
			}
			f_added = false;
			closure_group(100, verbose_level - 2);
		}
		else {
			f_added = true;
			if (f_v) {
				cout << "sims::build_up_subgroup_random_process "
						"element needs to be inserted at level = "
					<< drop_out_level << " with image "
					<< image << endl;
				if (true) {
					GA->Group_element->element_print(
							GA->Group_element->Elt2, cout);
					cout  << endl;
				}
			}
			add_generator_at_level(
					GA->Group_element->Elt2,
					drop_out_level,
					0/*verbose_level - 3*/);
		}
		
		group_order(go);
		if ((f_v && f_added) || f_vv) {
			cout << "sims::build_up_subgroup_random_process "
					"new group order is " << go << " : ";
			print_transversal_lengths();
		}
		cnt++;
	}
	if (f_v) {
		cout << "sims::build_up_subgroup_random_process "
				"finished: found a group of order " << go << endl;
		print_transversal_lengths();
	}
}


// global function:

void choose_random_generator_derived_group(
		sims *G,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6;
	actions::action *A;

	if (f_v) {
		cout << "choose_random_generator_derived_group" << endl;
	}
	A = G->A;
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	Elt5 = NEW_int(A->elt_size_in_int);
	Elt6 = NEW_int(A->elt_size_in_int);

	G->random_element(Elt1, verbose_level - 1);
	G->random_element(Elt2, verbose_level - 1);
	A->Group_element->invert(Elt1, Elt3);
	A->Group_element->invert(Elt2, Elt4);
	A->Group_element->mult(Elt3, Elt4, Elt5);
	A->Group_element->mult(Elt1, Elt2, Elt6);
	A->Group_element->mult(Elt5, Elt6, Elt);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	FREE_int(Elt5);
	FREE_int(Elt6);
}



}}}

