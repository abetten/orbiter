/*
 * parametric_curve.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace graphics {


parametric_curve::parametric_curve()
{
	Record_birth();
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
	Record_death();
}

void parametric_curve::init(
		int nb_dimensions,
		double desired_distance,
		double t0, double t1,
		int (*compute_point_function)(
				double t, double *pt, void *extra_data, int verbose_level),
		void *extra_data,
		double boundary,
		int N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double t, d, dt, t_last, tf;
	double *coords0;
	double *coords;
	orbiter_kernel_system::numerics Num;
	int f_is_valid;
	double epsilon = 0.0001;

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
#if 0
	if (nb_dimensions != 2) {
		cout << "parametric_curve::init nb_dimensions != 2" << endl;
		exit(1);
	}
#endif

	coords = new double[nb_dimensions];
	coords0 = new double[nb_dimensions];

	t = t0;

	if (f_v) {
		cout << "parametric_curve::init computing value at t=" << setw(8) << t << endl;
	}
	f_is_valid = (*compute_point_function)(t, coords, extra_data, verbose_level);
	if (f_v) {
		cout << "parametric_curve::init computing value at t=" << setw(8) << t << " done" << endl;
	}
	if (f_is_valid) {
		d = Num.distance_from_origin(coords, nb_dimensions);
		if (f_v) {
			cout << "created point t=" << setw(8) << t << " = (" << coords[0] << "," << coords[1] << ")" << endl;
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
			cout << "d=" << d << endl;
			cout << "boundary=" << boundary << endl;
		}
		double tl, tr, tm;
		double epsilon = 0.0001;
		tl = t;
		tr = t;
		while (true) {
			tr = tr + dt;

			if (f_v) {
				cout << "parametric_curve::init computing value at tr=" << setw(8) << tr << endl;
			}
			f_is_valid = (*compute_point_function)(tr, coords, extra_data, 0 /*verbose_level*/);
			if (f_v) {
				cout << "parametric_curve::init computing value at tr=" << setw(8) << tr << " done" << endl;
			}

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


			if (f_v) {
				cout << "parametric_curve::init computing value at tm=" << setw(8) << tm << endl;
			}
			f_is_valid = (*compute_point_function)(tm, coords, extra_data, 0 /*verbose_level*/);
			if (f_v) {
				cout << "parametric_curve::init computing value at tm=" << setw(8) << tm << " done" << endl;
			}

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
			cout << "created starting value t=" << setw(8) << t << endl;
		}
	}

	if (f_v) {
		cout << "parametric_curve::init computing value at t=" << setw(8) << t << endl;
	}
	f_is_valid = (*compute_point_function)(t, coords, extra_data, 0 /*verbose_level*/);
	if (f_v) {
		cout << "parametric_curve::init computing value at t=" << setw(8) << t << " done" << endl;
	}

	if (f_is_valid) {
		d = Num.distance_from_origin(coords, nb_dimensions);
	}
	else {
		cout << "the initial point is invalid" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "created point t=" << setw(8) << t << " = (" << coords[0] << "," << coords[1] << ")" << endl;
	}



	vector<parametric_curve_point> Future;

	{
		parametric_curve_point Pt;
		if (f_v) {
			cout << "adding point " << Pts.size() << endl;
		}
		Pt.init(t, f_is_valid, coords, nb_dimensions, verbose_level);
		Pts.push_back(Pt);
	}

	int f_success;

	while (t < t1) {
		t_last = t;

		if (f_v) {
			cout << "t_last = " << setw(8) << t_last << " Future.size() = " << Future.size() << endl;
		}
		if (Future.size()) {
			f_success = false;
			while (Future.size()) {
				if (!Future[Future.size() - 1].f_is_valid) {
					break;
				}
				tf = Future[Future.size() - 1].t;
				Future.pop_back();
				if (tf > t_last) {
					t = (t_last + tf) * 0.5;
					if (f_v) {
						cout << "t_last = " << setw(8) << t_last << " t = " << setw(8) << t << " tf=" << setw(8) << tf << endl;
					}
					f_success = true;
					if (f_v) {
						cout << "t_last = " << setw(8) << t_last << " tf = " << setw(8) << tf << " t = " << setw(8) << t << " popped from Future" << endl;
					}
					break;
				}
			}
			if (!f_success) {
				if (f_v) {
					cout << "no success, moving on by dt, clearing Future" << endl;
				}
				t += dt;
				while (Future.size()) {
					Future.pop_back();
				}
				if (f_v) {
					cout << "t_last = " << setw(8) << t_last << " t = " << setw(8) << t << endl;
				}
			}
		}
		else {
			t += dt;
			f_success = true;
		}

		if (ABS(t_last - t) < epsilon) {
			cout << "t_last == t" << endl;
			cout << "t_last = " << setw(8) << t_last << endl;
			cout << "adding dt" << endl;
			t += dt;
			f_success = false;
		}
#if 0
		if (ABS(t - t_last) < epsilon) {
			if (f_v) {
				cout << "no success, ABS(t - t_last) < epsilon, ABS(t - t_last)=" << ABS(t - t_last) << endl;
			}
			f_success = false;
		}
#endif
		if (f_v) {
			cout << "t_last = " << setw(8) << t_last << " t = " << setw(8) << t << endl;
			cout << "t = " << setw(8) << t << " t1 = " << t1 << endl;
			cout << "t - t_last = " << setw(8) << t - t_last << endl;
		}
		if (t > t1) {
			if (f_v) {
				cout << "t > t1, break" << endl;
			}
			break;
		}

		if (f_v) {
			cout << "parametric_curve::init computing value at t=" << setw(8) << t << endl;
		}
		f_is_valid = (*compute_point_function)(t, coords, extra_data, 0 /*verbose_level*/);
		if (f_v) {
			cout << "parametric_curve::init computing value at t=" << setw(8) << t << " done" << endl;
		}

		if (f_is_valid) {
			int h;

			for (h = 0; h < nb_dimensions; h++) {
				coords0[h] = Pts[Pts.size() - 1].coords[h];
			}
			if (f_v) {
				cout << "t_last = " << setw(8) << t_last << " : (" << coords0[0] << "," << coords0[1] << ")" << endl;
			}
			d = Num.distance_euclidean(coords0, coords, nb_dimensions);
			if (f_v) {
				cout << "t = " << setw(8) << t << " d = " << d << endl;
			}
		}
		if (f_success && (!f_is_valid || d > desired_distance)) {


			if (f_v) {
				cout << "t=" << setw(8) << t << " d > desired_distance, pushing to Future, moving back to t_last" << setw(8) << t_last << endl;
			}

			{
				parametric_curve_point Pt;
				Pt.init(t, f_is_valid, coords, nb_dimensions, verbose_level);
				if (Future.size()) {
					cout << "top element of Future stack is t=" << setw(8) << Future[Future.size() - 1].t << " we are pushing " << t << endl;
				}
				Future.push_back(Pt);
			}

			t = t_last;

			if ((int) Future.size() > N) {
				if (f_v) {
					cout << "Future.size() > N, popping stack" << endl;
				}
				t = Future[0].t;
				while (Future.size() > 0) {
					Future.pop_back();
				}
				cout << "after popping stack, Future.size() = " << Future.size() << ", t = " << setw(8) << t << endl;
				if (t < t_last) {
					t = t_last;
				}
			}

		}
		else {
			if (f_v) {
				cout << "t=" << setw(8) << t << " d < desired_distance, valid point" << endl;
			}

			d = Num.distance_from_origin(coords, nb_dimensions);

			if (f_v) {
				cout << "created point t=" << setw(8) << t << " = (" << coords[0] << "," << coords[1] << ") with distance from origin " << d << endl;
			}

			if (d < boundary) {
				int h;

				{
					parametric_curve_point Pt;
					if (f_v) {
						cout << "adding point " << Pts.size() << endl;
					}
					Pt.init(t, f_is_valid, coords, nb_dimensions, verbose_level);
					Pts.push_back(Pt);
				}

				for (h = 0; h < nb_dimensions; h++) {
					coords0[h] = Pts[Pts.size() - 1].coords[h];
				}
				cout << "created point " << coords0[0] << "," << coords0[1] << endl;
			}
			else {
				cout << "skipping" << endl;
			}
			if (Future.size()) {
				int h;

				for (h = 0; h < nb_dimensions; h++) {
					coords0[h] = Future[Future.size() - 1].coords[h];
				}
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


}}}}



