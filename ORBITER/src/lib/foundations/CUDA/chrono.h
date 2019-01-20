/* 
 * File:   chrono.h
 * Author: Sajeeb
 *
 * Created on October 26, 2017, 5:02 PM
 */

#ifndef CHRONO_H
#define CHRONO_H

#include <math.h>
#include <cmath>
#include <chrono>

class chrono {
public:
    chrono() { start(); }
    ~chrono() {
    }
    void start() {
    	auto now = std::chrono::system_clock::now();
		auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
		auto epoch = now_ms.time_since_epoch();
		auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
		timestamp = value.count();
    }
    long calculateDuration(const chrono& c) {
        return (c.timestamp-timestamp);
    }
    void reset() { start(); }
    long timestamp;
private:
    static const bool debug =  false;
};

#endif /* CHRONO_H */

