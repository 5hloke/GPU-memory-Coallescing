#include <ocelot/api/interface/ocelot.h>
#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/trace/interface/TraceEvent.h>
#include <iostream>

class TraceGenerator : public trace::TraceGenerator
{
	public:
		void event(const trace::TraceEvent & event)
		{
			std::cout << "Got event " << event.instruction->toString() << "\n";
		}
};

extern void reductionKernel();

int main()
{
	TraceGenerator generator;
	ocelot::addTraceGenerator( generator );
	
	reductionKernel();
}