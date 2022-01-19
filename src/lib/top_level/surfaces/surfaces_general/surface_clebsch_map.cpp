/*
 * surface_clebsch_map.cpp
 *
 *  Created on: Jul 17, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


surface_clebsch_map::surface_clebsch_map()
{
	SOA = NULL;

	orbit_idx = 0;
	f = l = hds = 0;

	Clebsch_map = NULL;

}


surface_clebsch_map::~surface_clebsch_map()
{
	if (Clebsch_map) {
		FREE_OBJECT(Clebsch_map);
	}
}

void surface_clebsch_map::report(std::ostream &ost, int verbose_level)
{

	ost << "\\subsection*{Orbit on single sixes " << orbit_idx << " / "
		<< SOA->Orbits_on_single_sixes->nb_orbits << "}" << endl;


	Clebsch_map->report(ost, verbose_level);

}

void surface_clebsch_map::init(surface_object_with_action *SOA, int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_clebsch_map::init orbit "
			"on single sixes " << orbit_idx << " / "
			<< SOA->Orbits_on_single_sixes->nb_orbits << ":" << endl;
	}

	surface_clebsch_map::SOA = SOA;
	surface_clebsch_map::orbit_idx = orbit_idx;


	f = SOA->Orbits_on_single_sixes->orbit_first[orbit_idx];
	l = SOA->Orbits_on_single_sixes->orbit_len[orbit_idx];
	if (f_v) {
		cout << "orbit " << orbit_idx << " has f=" << f <<  " l=" << l << endl;
	}
	hds = SOA->Orbits_on_single_sixes->orbit[f];

	if (f_v) {
		cout << "The half double six is no " << hds << " : ";
		Orbiter->Lint_vec->print(cout, SOA->Surf->Schlaefli->Half_double_sixes + hds * 6, 6);
		cout << endl;
	}



	Clebsch_map = NEW_OBJECT(algebraic_geometry::clebsch_map);


	if (f_v) {
		cout << "surface_clebsch_map::init orbit "
			"on single sixes " << orbit_idx << " / "
			<< SOA->Orbits_on_single_sixes->nb_orbits << " before Clebsch_map->init_half_double_six" << endl;
	}
	Clebsch_map->init_half_double_six(SOA->SO, hds, verbose_level);
	if (f_v) {
		cout << "surface_clebsch_map::init orbit "
			"on single sixes " << orbit_idx << " / "
			<< SOA->Orbits_on_single_sixes->nb_orbits << " after Clebsch_map->init_half_double_six" << endl;
	}



	if (f_v) {
		cout << "surface_clebsch_map::init orbit "
			"on single sixes " << orbit_idx << " / "
			<< SOA->Orbits_on_single_sixes->nb_orbits << " before Clebsch_map->compute_Clebsch_map_down" << endl;
	}
	Clebsch_map->compute_Clebsch_map_down(verbose_level);
	if (f_v) {
		cout << "surface_clebsch_map::init orbit "
			"on single sixes " << orbit_idx << " / "
			<< SOA->Orbits_on_single_sixes->nb_orbits << " after Clebsch_map->compute_Clebsch_map_down" << endl;
	}



	if (f_v) {
		cout << "clebsch map for lines " << Clebsch_map->line_idx[0]
			<< " = " << SOA->Surf->Schlaefli->Labels->Line_label_tex[Clebsch_map->line_idx[0]] << ", "
			<< Clebsch_map->line_idx[1] << " = " << SOA->Surf->Schlaefli->Labels->Line_label_tex[Clebsch_map->line_idx[1]]
			<< " before clebsch_map_print_fibers:" << endl;
	}
	Clebsch_map->clebsch_map_print_fibers();

	if (f_v) {
		cout << "clebsch map for lines " << Clebsch_map->line_idx[0]
			<< " = " << SOA->Surf->Schlaefli->Labels->Line_label_tex[Clebsch_map->line_idx[0]] << ", "
			<< Clebsch_map->line_idx[1] << " = " << SOA->Surf->Schlaefli->Labels->Line_label_tex[Clebsch_map->line_idx[1]]
			<< "  before Clebsch_map->clebsch_map_find_arc_and_lines:" << endl;
	}

	Clebsch_map->clebsch_map_find_arc_and_lines(verbose_level - 1);



	if (f_v) {
		cout << "surface_clebsch_map::init "
				"after Clebsch_map->clebsch_map_find_arc_and_lines" << endl;
	}


	if (f_v) {
		cout << "surface_clebsch_map::init "
				"Clebsch map for lines " << Clebsch_map->line_idx[0] << ", "
				<< Clebsch_map->line_idx[1] << " yields arc = ";
		Orbiter->Lint_vec->print(cout, Clebsch_map->Arc, 6);
		cout << " : blown up lines = ";
		Orbiter->Lint_vec->print(cout, Clebsch_map->Blown_up_lines, 6);
		cout << endl;
	}



}

}}
