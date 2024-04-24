#include <ocelot/api/interface/ocelot.h>
#include <boost/dynamic_bitset.hpp>
#include "ocelot/trace/interface/TraceGenerator.h"
#include "ocelot/trace/interface/TraceEvent.h"
#include <iostream>

namespace ID
{
    int static_ID = 0;
}
class TraceGenerator : public trace::TraceGenerator
{
public:
    	void event(const trace::TraceEvent &event)
    	{
    	// Here are the block dimensions
    	int bdimx = event.blockDim.x;
    	int bdimy = event.blockDim.y;
    	int bdimz = event.blockDim.z;
    	// Following is the format of the output
    	// blockidx blockidy blockidz threadidx threadyidx threadidz instruction memory_addresses
    	if (event.instruction->isMemoryInstruction()){
        	// so now get the block id of the event first
        	int blockidx = event.blockId.x;
        	int blockidy = event.blockId.y;
        	int blockidz = event.blockId.z;
        	// now get the thread id of the event
        	int threadidx = 0; // rows
        	int threadidy = 0; // column
        	int threadidz = 0;
        	auto vec = event.memory_addresses;
        	int memory_counter = 0;
        	for (size_t i = 0; i < event.active.size(); ++i) {
            	if (event.active[i]) {
                	// Now compute the x index based on the thread geometry
                	threadidz = i / (bdimx * bdimy); // Computing z ID

                	threadidx = i % bdimx; // Computing x ID
                	threadidy = (i / bdimx) % bdimy; // Computing y ID
                    /*
                    The format of the print is:
                    blockidx blockidy blockidz threadidx threadyidx threadidz ID::static_id instruction memory_addresses
                    */
                	std::cout << blockidx << " " << blockidy << " " << blockidz << " " << threadidx << " " << threadidy << " " << threadidz << " " << ID::static_ID << " "<< event.instruction->toString() << " ";
                	// Printing active threads information (Assuming that there will be as many memory addresses as active threads)
                    std::cout << vec[memory_counter] << '\n';
                    
                    memory_counter++;

            	}
        	}
            ID::static_ID++;
    	}
    }
};

extern void sampleKernel(int size);
extern void gaussianKernel();

int main()
{
    	TraceGenerator generator;
    	ocelot::addTraceGenerator( generator );

    	sampleKernel(5);
}

