#include "FPSLimiter.h"
#include <iostream>

int main()
{
	FPSLimiter fpslimiter;
  while (true)
	{
		std::cout << "Test" << std::endl;
		fpslimiter.Pulse(5);
	}
	return 0;
}