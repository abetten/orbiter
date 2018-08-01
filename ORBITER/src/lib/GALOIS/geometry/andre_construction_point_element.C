// andre_construction_point_element.C
// 
// Anton Betten
// May 31, 2013
//
//
// 
//
//

#include "galois.h"




andre_construction_point_element::andre_construction_point_element()
{
	null();
}

andre_construction_point_element::~andre_construction_point_element()
{
	freeself();
}

void andre_construction_point_element::null()
{
	coordinates = NULL;
}

void andre_construction_point_element::freeself()
{
	if (coordinates) {
		FREE_INT(coordinates);
		}
	null();
}

void andre_construction_point_element::init(andre_construction *Andre, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "andre_construction_point_element::init" << endl;
		}
	andre_construction_point_element::Andre = Andre;
	andre_construction_point_element::k = Andre->k;
	andre_construction_point_element::n = Andre->n;
	andre_construction_point_element::q = Andre->q;
	andre_construction_point_element::spread_size = Andre->spread_size;
	andre_construction_point_element::F = Andre->F;
	coordinates = NEW_INT(n);
	if (f_v) {
		cout << "andre_construction_point_element::init done" << endl;
		}
}

void andre_construction_point_element::unrank(INT point_rank, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "andre_construction_point_element::unrank point_rank=" << point_rank << endl;
		}
	andre_construction_point_element::point_rank = point_rank;
	if (point_rank < spread_size) {
		f_is_at_infinity = TRUE;
		at_infinity_idx = point_rank;
		}
	else {
		f_is_at_infinity = FALSE;
		point_rank -= spread_size;
		AG_element_unrank(q, coordinates, 1, n, point_rank);
		}
	if (f_v) {
		cout << "andre_construction_point_element::unrank done" << endl;
		}
}

INT andre_construction_point_element::rank(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a;

	if (f_v) {
		cout << "andre_construction_point_element::rank" << endl;
		}
	point_rank = 0;
	if (f_is_at_infinity) {
		point_rank = at_infinity_idx;
		}
	else {
		point_rank = spread_size;
		AG_element_rank(q, coordinates, 1, n, a);
		point_rank += a;
		}
	if (f_v) {
		cout << "andre_construction_point_element::rank done" << endl;
		}
	return point_rank;
}



