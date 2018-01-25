// geometric_tests.C
// Anton Betten
//
// started: December 2006
// moved away from inc_gen: 8/27/07
// renamed from refine to geometric_tests: 1/3/10

#include "galois.h"
#include "incidence.h"

INT tdo_scheme::geometric_test_for_row_scheme(partitionstack &P, 
	INT *point_types, INT nb_point_types, INT point_type_len, 
	INT *distributions, INT nb_distributions, 
	INT f_omit1, INT omit1, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_vvvv = (verbose_level >= 4);
	INT f_v5 = (verbose_level >= 7);
	INT i, s, d, l2, L1, L2, cnt, new_nb_distributions; 
	INT f_ruled_out;
	INT *ruled_out_by;
	INT *non_zero_blocks, nb_non_zero_blocks;
	
	if (f_vvv) {
		cout << "tdo_refine::geometric_test_for_row_scheme nb_distributions=" << nb_distributions << endl;
		}
	l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit1, omit1, L1, L2, verbose_level - 3);
	if (L2 != point_type_len) {
		cout << "tdo_refine::geometric_test_for_row_scheme L2 != point_type_len" << endl;
		exit(1);
		}
	
	ruled_out_by = NEW_INT(nb_point_types + 1);
	non_zero_blocks = NEW_INT(nb_point_types);
	for (i = 0; i <= nb_point_types; i++) {
		ruled_out_by[i] = 0;
		}

	new_nb_distributions = 0;
	for (cnt = 0; cnt < nb_distributions; cnt++) {
		nb_non_zero_blocks = 0;
		for (i = 0; i < nb_point_types; i++) {
			d = distributions[cnt * nb_point_types + i];
			if (d == 0)
				continue;
			non_zero_blocks[nb_non_zero_blocks++] = i;
			}
		
		if (f_vvvv) {
			cout << "geometric_test_for_row_scheme: testing distribution " 
				<< cnt << " / " << nb_distributions << " : ";
			INT_vec_print(cout, distributions + cnt * nb_point_types, nb_point_types);
			cout << endl;
			if (f_v5) {
				cout << "that is" << endl;
				for (i = 0; i < nb_non_zero_blocks; i++) {
					d = distributions[cnt * nb_point_types + non_zero_blocks[i]];
					cout << setw(3) << i << " : " << setw(3) << d << " x ";
					INT_vec_print(cout, point_types + non_zero_blocks[i] * point_type_len, point_type_len);
					cout << endl;
					}
				}
			}
		f_ruled_out = FALSE;
		for (s = 1; s <= nb_non_zero_blocks; s++) {
			if (!geometric_test_for_row_scheme_level_s(P, s, 
				point_types, nb_point_types, point_type_len, 
				distributions + cnt * nb_point_types, 
				non_zero_blocks, nb_non_zero_blocks, 
				f_omit1, omit1, verbose_level - 4)) {
				f_ruled_out = TRUE;
				ruled_out_by[s]++;
				if (f_vv) {
					cout << "geometric_test_for_row_scheme: distribution " << cnt << " / " << nb_distributions << " eliminated by test of order " << s << endl;
					}
				if (f_vvv) {
					cout << "the eliminated scheme is:" << endl;
					for (i = 0; i < nb_non_zero_blocks; i++) {
						d = distributions[cnt * nb_point_types + non_zero_blocks[i]];
						cout << setw(3) << i << " : " << setw(3) << d << " x ";
						INT_vec_print(cout, point_types + non_zero_blocks[i] * point_type_len, point_type_len);
						cout << endl;						
						}
					cout << "we repeat the test with more printout:" << endl;
					geometric_test_for_row_scheme_level_s(P, s, 
						point_types, nb_point_types, point_type_len, 
						distributions + cnt * nb_point_types, 
						non_zero_blocks, nb_non_zero_blocks, 
						f_omit1, omit1, verbose_level - 3);
					}
				break;
				}
			}
		


		if (!f_ruled_out) {
			for (i = 0; i < nb_point_types; i++) {
				distributions[new_nb_distributions * nb_point_types + i] = 
					distributions[cnt * nb_point_types + i];
				}
			new_nb_distributions++;
			}
		} // next cnt
	if (f_v) {
		cout << "geometric_test_for_row_scheme: number of distributions reduced from " << nb_distributions << " to " 
			<< new_nb_distributions << ", i.e. Eliminated " 
			<< nb_distributions - new_nb_distributions << " cases" << endl;
		cout << "# of ruled out by test of order ";
		INT_vec_print(cout, ruled_out_by, nb_point_types + 1);
		cout << endl;
		//cout << "nb ruled out by first order test  = " << nb_ruled_out_by_order1 << endl;
		//cout << "nb ruled out by second order test = " << nb_ruled_out_by_order2 << endl;
		for (i = nb_point_types; i >= 1; i--) {
			if (ruled_out_by[i])
				break;
			}
		if (i) {
			cout << "highest order test that was successfully applied is order " << i << endl;
			}
		}
	FREE_INT(ruled_out_by);
	FREE_INT(non_zero_blocks);
	return new_nb_distributions;
}
#if 0
INT tdo_scheme::test_row_distribution(INT *point_types, INT nb_point_types, INT point_type_len, 
	INT *distributions, INT nb_distributions, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT l2, cnt, J, len, k, i, d, c, new_nb_distributions, bound;
	INT f_ruled_out, f_ruled_out_by_braun, f_ruled_out_by_packing;
	INT nb_ruled_out_by_braun = 0, nb_ruled_out_by_packing = 0, nb_ruled_out_by_both = 0;
	
	if (f_v) {
		cout << "tdo_refine::test_row_distribution nb_distributions=" << nb_distributions << endl;
		}
	l2 = nb_col_classes[COL];
	if (l2 != point_type_len) {
		cout << "tdo_refine::test_row_distribution l2 != point_type_len" << endl;
		exit(1);
		}
	
	new_nb_distributions = 0;
	
	for (cnt = 0; cnt < nb_distributions; cnt++) {
		if (f_vv) {
			cout << "testing distribution " << cnt << " : ";
			INT_vec_print(cout, distributions + cnt * nb_point_types, nb_point_types);
			cout << endl;
			if (f_vvv) {
				cout << "that is" << endl;
				for (i = 0; i < nb_point_types; i++) {
					d = distributions[cnt * nb_point_types + i];
					if (d == 0)
						continue;
					cout << setw(3) << d << " x ";
					INT_vec_print(cout, point_types + i * point_type_len, point_type_len);
					cout << endl;
					}
				}
			}
		f_ruled_out = FALSE;
		f_ruled_out_by_braun = FALSE;
		f_ruled_out_by_packing = FALSE;
		
		for (J = 0; J < l2; J++) {
			len = col_classes_len[COL][J];
			INT *type;
			
			if (f_vvv) {
				cout << "testing distribution " << cnt << " in block " << J << " len=" << len << endl;
				}
			type = NEW_INT(len + 1);
			for (k = 0; k <= len; k++)
				type[k] = 0;
			for (i = 0; i < nb_point_types; i++) {
				d = distributions[cnt * nb_point_types + i];
				c = point_types[i * point_type_len + J];
				type[c] += d;
				}
			if (f_vvv) {
				cout << "line type: ";
				INT_vec_print(cout, type + 1, len);
				cout << endl;
				}
			if (!braun_test_on_line_type(len, type)) {
				if (f_vv) {
					cout << "distribution " << cnt << " is eliminated in block " << J << " using Braun test" << endl;
					}
				f_ruled_out = TRUE;
				f_ruled_out_by_braun = TRUE;
				FREE_INT(type);
				break;
				}
			FREE_INT(type);
			} // next J
		for (J = 0; J < l2; J++) {
			len = col_classes_len[COL][J];
			if (len == 1) 
				continue;
			for (i = 0; i < nb_point_types; i++) {
				d = distributions[cnt * nb_point_types + i];
				if (d == 0)
					continue;
				c = point_types[i * point_type_len + J];
				// now we want d lines of size c on len points
				if (c > 1) {
					if (c > len) {
						cout << "c > len" << endl;
						cout << "J=" << J << " i=" << i << " d=" << d << " c=" << c << endl;
						exit(1);
						}
					bound = TDO_upper_bound(len, c);
					if (d > bound) {
						if (f_vv) {
							cout << "distribution " << cnt << " is eliminated in block " << J << " row-block " << i 
								<< " using packing numbers" << endl;
							cout << "len=" << len << endl;
							cout << "d=" << d << endl;
							cout << "c=" << c << endl;
							cout << "bound=" << bound << endl;
							}
						f_ruled_out = TRUE;
						f_ruled_out_by_packing = TRUE;
						break;
						}
					}
				}
			if (f_ruled_out)
				break;
			}
		if (f_ruled_out) {
			if (f_ruled_out_by_braun) 
				nb_ruled_out_by_braun++;
			if (f_ruled_out_by_packing)
				nb_ruled_out_by_packing++;
			if (f_ruled_out_by_braun && f_ruled_out_by_packing)
				nb_ruled_out_by_both++;
			}
		else {
			for (i = 0; i < nb_point_types; i++) {
				distributions[new_nb_distributions * nb_point_types + i] = 
					distributions[cnt * nb_point_types + i];
				}
			new_nb_distributions++;
			}
		} // next cnt
	if (f_v) {
		cout << "number of distributions reduced from " << nb_distributions << " to " 
			<< new_nb_distributions << ", i.e. Eliminated " 
			<< nb_distributions - new_nb_distributions << " cases" << endl;
		cout << "nb_ruled_out_by_braun = " << nb_ruled_out_by_braun << endl;
		cout << "nb_ruled_out_by_packing = " << nb_ruled_out_by_packing << endl;
		cout << "nb_ruled_out_by_both = " << nb_ruled_out_by_both << endl;
		}
	return new_nb_distributions;
}
#endif

INT tdo_scheme::geometric_test_for_row_scheme_level_s(partitionstack &P, INT s, 
	INT *point_types, INT nb_point_types, INT point_type_len, 
	INT *distribution, 
	INT *non_zero_blocks, INT nb_non_zero_blocks, 
	INT f_omit1, INT omit1, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vvv = (verbose_level >= 3);
	INT set[1000];
	INT J, L1, L2, len, max, cur, u, D, d, c, nb_inc, e, f, nb_ordererd_pairs;
	
	if (f_vvv) {
		cout << "geometric_test_for_row_scheme_level_s s=" << s << endl;
		}
	if (s >= 1000) {
		cout << "level too deep" << endl;
		exit(1);
		}
	row_refinement_L1_L2(P, f_omit1, omit1, L1, L2, verbose_level - 3);
	first_k_subset(set, nb_non_zero_blocks, s);
	while (TRUE) {
		D = 0;
		for (u = 0; u < s; u++) {
			d = distribution[non_zero_blocks[set[u]]];
			D += d;
			}
		max = D * (D - 1);
		cur = 0;
		for (J = 0; J < L2; J++) {
			len = col_classes_len[COL][J];
			nb_inc = 0;
			for (u = 0; u < s; u++) {
				c = point_types[non_zero_blocks[set[u]] * point_type_len + J];
				d = distribution[non_zero_blocks[set[u]]];
				// we have d rows with c incidences in len columns
				nb_inc += d * c;
				}
			
			e = nb_inc % len; // the number of incidences in the extra row
			f = nb_inc / len; // the number of full rows

			nb_ordererd_pairs = 0;
			if (n) {
				nb_ordererd_pairs = e * (f + 1) * f + (len - e) * f * (f - 1);
				}
			cur += nb_ordererd_pairs;
			if (cur > max) {
				if (f_v) {
					cout << "tdo_scheme::geometric_test_for_row_scheme_level_s s=" << s << " failure in point type ";
					INT_vec_print(cout, set, s);
					cout << endl;
					cout << "max=" << max << endl;
					cout << "J=" << J << endl;
					cout << "nb_inc=" << nb_inc << endl;
					cout << "nb_ordererd_pairs=" << nb_ordererd_pairs << endl;
					cout << "cur=" << cur << endl;
					}
				return FALSE;
				}
			} // next J
		if (!next_k_subset(set, nb_non_zero_blocks, s))
			break;
		}
	return TRUE;
}


