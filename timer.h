#pragma once
#include <time.h>
#include <iostream>
#include <string>
using namespace std;

inline timespec diff(timespec _start, timespec _end)
{
	timespec temp;
	if ((_end.tv_nsec-_start.tv_nsec)<0) {
		temp.tv_sec = _end.tv_sec-_start.tv_sec-1;
		temp.tv_nsec = 1000000000+_end.tv_nsec-_start.tv_nsec;
	} else {
		temp.tv_sec = _end.tv_sec-_start.tv_sec;
		temp.tv_nsec = _end.tv_nsec-_start.tv_nsec;
	}
	return temp;
}

class timer {
protected:
	timespec _start,_end;
public:
	timer() { this->tick(); }
	double tick(bool update_time_stamp = true) {
		static bool first_call = true;
		if (first_call) {
			first_call = false;
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&_start);
			return 0.0;
		}
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&_end);
		timespec t = diff(_start,_end);
		if(update_time_stamp) _start = _end;
		double dt = double(t.tv_sec) + double(t.tv_nsec)/1000000000.0;
		return dt;
	}
	string tick_str() {
		double dt = tick();
		string s = to_string(dt * 1000) + "ms";
		return s;
	}
};



