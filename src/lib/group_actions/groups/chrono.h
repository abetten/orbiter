/*
 * File:   chrono.h
 * Author: Sajeeb
 *
 * Created on October 26, 2017, 5:02 PM
 */

/**
 * This file returns the time in milliseconds
 */

#ifndef CHRONO_H
#define CHRONO_H

#include <math.h>
#include <cmath>
#include <chrono>

class chrono_ {
public:
    chrono_() { start(); }
    ~chrono_() {
    }
    void start() {
    	auto now = std::chrono::system_clock::now();
		auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
		auto epoch = now_ns.time_since_epoch();
		auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
		timestamp = value.count();
    }
    long calculateDuration(const chrono_& c) {
        return (c.timestamp-timestamp);
    }
    void reset() { start(); }
    long timestamp;
private:
    static const bool debug =  false;
};

#endif /* CHRONO_H */
