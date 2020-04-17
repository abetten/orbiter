/*
 * parametric_curve.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


parametric_curve::parametric_curve()
{
	nb_dimensions = 0;
	desired_distance = 0;;
	t0 = 0;
	t1 = 0; // parameter interval
	compute_point_function = NULL;
	extra_data = 0;
	boundary = 0;

	nb_pts = 0;

}

parametric_curve::~parametric_curve()
{
}

void parametric_curve::init(int nb_dimensions,
		double desired_distance,
		double t0, double t1,
		int (*compute_point_function)(double t, double *pt, void *extra_data),
		void *extra_data,
		double boundary,
		int N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double t, d, dt, t_last, tf;
	double *coords0;
	double *coords;
	numerics Num;
	int f_is_valid;

	if (f_v) {
		cout << "parametric_curve::init" << endl;
	}
	parametric_curve::nb_dimensions = nb_dimensions;
	parametric_curve::desired_distance = desired_distance;
	parametric_curve::t0 = t0;
	parametric_curve::t1 = t1;
	parametric_curve::compute_point_function = compute_point_function;
	parametric_curve::extra_data = extra_data;
	parametric_curve::boundary = boundary;
	if (f_v) {
		cout << "parametric_curve::init nb_dimensions=" << nb_dimensions << endl;
		cout << "parametric_curve::init desired_distance=" << desired_distance << endl;
		cout << "parametric_curve::init boundary=" << boundary << endl;
		cout << "parametric_curve::init t0=" << t0 << endl;
		cout << "parametric_curve::init t1=" << t1 << endl;
		cout << "parametric_curve::init N=" << N << endl;
	}
	dt = (t1 - t0) / N;
	if (f_v) {
		cout << "parametric_curve::init dt=" << dt << endl;
	}
	if (nb_dimensions != 2) {
		cout << "parametric_curve::init nb_dimensions != 2" << endl;
		exit(1);
	}

	coords = new double[nb_dimensions];
	coords0 = new double[nb_dimensions];

	t = t0;

	f_is_valid = (*compute_point_function)(t, coords, extra_data);
	if (f_is_valid) {
		d = Num.distance_from_origin(coords, nb_dimensions);
		if (f_v) {
			cout << "created point t=" << t << " = (" << coords[0] << "," << coords[1] << ")" << endl;
		}
	}
	else {
		d = 0;
		if (f_v) {
			cout << "invalid starting point" << endl;
		}
	}

	if (!f_is_valid || d > boundary) {
		if (f_v) {
			cout << "parametric_curve::init d > boundary, performing a search for the starting value of t" << endl;
		}
		double tl, tr, tm;
		double epsilon = 0.0001;
		tl = t;
		tr = t;
		while (TRUE) {
			tr = tr + dt;
			f_is_valid = (*compute_point_function)(tr, coords, extra_data);

			if (f_is_valid) {
				d = Num.distance_from_origin(coords, nb_dimensions);
			}
			else {
				d = 0;
			}
			if (f_is_valid && d < boundary) {
				break;
			}
		}
		if (f_v) {
			cout << "performing a search for the starting value of t in the interval tl=" << tl << ", tr=" << tr << endl;
		}

		while (ABS(tr - tl) > epsilon) {
			tm = (tl + tr) * 0.5;
			f_is_valid = (*compute_point_function)(tm, coords, extra_data);

			if (f_is_valid) {
				d = Num.distance_from_origin(coords, nb_dimensions);
			}
			else {
				d = 0;
			}
			if (f_is_valid && d < boundary) {
				tr = tm;
			}
			else {
				tl = tm;
			}
		}
		t = tr;
		if (f_v) {
			cout << "created starting value t=" << t << endl;
		}
	}

	f_is_valid = (*compute_point_function)(t, coords, extra_data);

	if (f_is_valid) {
		d = Num.distance_from_origin(coords, nb_dimensions);
	}
	else {
		cout << "the initial point is invalid" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "created point t=" << t << " = (" << coords[0] << "," << coords[1] << ")" << endl;
	}


	parametric_curve_point Pt;

	vector<parametric_curve_point> Future;

	Pt.init2(t, f_is_valid, coords[0], coords[1]);
	Pts.push_back(Pt);

	//double epsilon = 0.01;
	int f_success;

	while (t < t1) {
		t_last = t;

		if (f_v) {
			cout << "t_last = " << t_last << " Future.size() = " << Future.size() << endl;
		}
		f_success = FALSE;
		while (Future.size()) {
			if (!Future[Future.size() - 1].f_is_valid) {
				break;
			}
			tf = Future[Future.size() - 1].t;
			Future.pop_back();
			if (tf > t_last) {
				t = (t_last + tf) * 0.5;
				f_success = TRUE;
				if (f_v) {
					cout << "t_last = " << t_last << " tf = " << tf << " t = " << t << " popped from Future" << endl;
				}
				break;
			}
		}
		if (t <= t_last) {
			f_success = FALSE;
		}
		if (!f_success) {
			t += dt;
			if (f_v) {
				cout << "t_last = " << t_last << " t = " << t << endl;
			}
		}
		if (f_v) {
			cout << "t = " << t << " t1 = " << t1 << endl;
			cout << "t1 - t = " << t1 - t << endl;
		}
		if (t > t1) {
			if (f_v) {
				cout << "t > t1, break" << endl;
			}
			break;
		}
		else {
			if (f_v) {
				cout << "t <= t1, keep going" << endl;
			}
		}

		f_is_valid = (*compute_point_function)(t, coords, extra_data);

		if (f_is_valid) {
			coords0[0] = Pts[Pts.size() - 1].coords[0];
			coords0[1] = Pts[Pts.size() - 1].coords[1];
			if (f_v) {
				cout << "t_last = " << t_last << " : (" << coords0[0] << "," << coords0[1] << ")" << endl;
			}
			d = Num.distance_euclidean(coords0, coords, nb_dimensions);
			if (f_v) {
				cout << "t = " << t << " d = " << d << endl;
			}
		}
		if (!f_is_valid || d > desired_distance) {
			cout << "t=" << t << " d > desired_distance, pushing to Future, moving back to t_last" << endl;

			Pt.init2(t, f_is_valid, coords[0], coords[1]);
			if (Future.size()) {
				cout << "top element of Future stack is t=" << Future[Future.size() - 1].t << " we are pushing " << t << endl;
			}
			Future.push_back(Pt);

			t = t_last;

			if (Future.size() > N) {
				if (f_v) {
					cout << "Future.size() > N, popping stack" << endl;
				}
				t = Future[0].t;
				while (Future.size() > 0) {
					Future.pop_back();
				}
				cout << "after popping stack, Future.size() = " << Future.size() << ", t = " << t << endl;
				if (t < t_last) {
					t = t_last;
				}
			}

		}
		else {
			cout << "t=" << t << " d < desired_distance, valid point" << endl;

			d = Num.distance_from_origin(coords, nb_dimensions);

			if (f_v) {
				cout << "created point t=" << t << " = (" << coords[0] << "," << coords[1] << ") with distance from origin " << d << endl;
			}

			if (d < boundary) {
				parametric_curve_point Pt;

				Pt.init2(t, f_is_valid, coords[0], coords[1]);
				Pts.push_back(Pt);

				coords0[0] = Pts[Pts.size() - 1].coords[0];
				coords0[1] = Pts[Pts.size() - 1].coords[1];
				cout << "created point " << coords0[0] << "," << coords0[1] << endl;
			}
			else {
				cout << "skipping" << endl;
			}
			if (Future.size()) {
				coords0[0] = Future[Future.size() - 1].coords[0];
				coords0[1] = Future[Future.size() - 1].coords[1];
				d = Num.distance_euclidean(coords0, coords, nb_dimensions);
				if (d < desired_distance) {
					Future.pop_back();
				}
			}
		}
	}


	delete [] coords;
	delete [] coords0;

	if (f_v) {
		cout << "parametric_curve::init done" << endl;
	}
}


}}

