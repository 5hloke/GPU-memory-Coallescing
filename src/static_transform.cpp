#include <ocelot/api/interface/ocelot.h>
#include <ocelot/transforms/interface/Pass.h>
#include <ocelot/transforms/interface/PassManager.h>
#include <ocelot/ir/interface/Module.h>
// #include <hydrazine/interface/Test.h>
// #include "ocelot/analysis/interface/Analysis.h"
#include <ocelot/analysis/interface/LoopAnalysis.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
// #include "ocelot/analysis/interface/DivergenceAnalysis.h"
// #include <hydrazine/interface/ArgumentParser.h>
// #include <hydrazine/interface/Exception.h>
// #include <hydrazine/hydrazine/interface/debug.h>

// // Boost Includes
// #include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>

namespace transforms {
class StaticTransform : public KernelPass
{
    private:

    public: 
    StaticTransform():KernelPass({"LoopAnalysis"}, "StaticTransform"){};
    void initialize(const ir::Module& m){};

    void runOnKernel(ir::IRKernel& k){
    std::cout << k.cfg()->size() << std::endl;

    auto beg = k.cfg()->begin();
    auto end = k.cfg()->end();
    for(; beg != end; beg++){
        auto block = *beg;
        auto instructions = block.instructions.begin();
        auto iend = block.instructions.end();
        for(; instructions != iend; instructions++){
            auto instruction = *instructions;
            std::cout << instruction->toString() << "\n";
        }
    }
    std::cout << "Hola Sir" << std::endl;
    Analysis* dfg_structure = getAnalysis("DataflowGraphAnalysis");
    assert(dfg_structure != 0);

     analysis::DataflowGraph& dfg =* static_cast<analysis::DataflowGraph*>(dfg_structure);

     analysis::DataflowGraph::iterator block = ++dfg.begin();
     analysis::DataflowGraph::iterator blockEnd = --dfg.end();
     for (; block != blockEnd; block++) {
       std::cout << "New Basic Block:\n";
       auto i = block->block()->instructions.begin();
       auto e = block->block()->instructions.end();
       while (i != e) {
         ir::PTXInstruction* inst = static_cast<ir::PTXInstruction*>(*i);
         std::cout << inst->toString() << std::endl;
         i++;
       }
     }
        // std::cout << "Hola Sir" << std::endl;
        // auto analysis = getAnalysis("LoopAnalysis");
        // // // std::cout << "Blocks: "<< analysis->blocks.size() << "\n";
        // analysis::LoopAnalysis* loopAnalysis = static_cast<analysis::LoopAnalysis*>(analysis);
        // // loopAnalysis->analyze(k);
        // auto loop = ++loopAnalysis->begin();
        // auto end = --loopAnalysis->end();
        // std::cout << "Begin: " << (loop == end) << " End: " << end->size() << "\n";
        // auto blocks = loop->getHeader();
        // std::cout << "Blocks: " << *blocks << "\n";
        // // for(; loop != end; loop++){
        //     std::cout << "Loop\n";
        //     auto blocks = loop->blocks;
        //     std::cout << "Blocks: " << "\n";
        //     auto bb = blocks.begin();
        //     auto bend = blocks.end();
        //     std::cout << "Here " <<std::endl;
        //     for(; bb != bend; bb++){
        //         std::cout << "Loop 2\n";
        //         auto block = *bb;
        //         auto instructions = block->instructions.begin();
        //         auto iend = block->instructions.end();
        //         for(; instructions != iend; instructions++){
        //             std::cout << "Loop 3\n";
        //             auto instruction = *instructions;
        //             std::cout << instruction->toString() << "\n";
        //         }
        //     }
        // // }
    }
    void finalize(){};
    // StaticTransform() : KernelPass(analysis::LoopAnalysis, "StaticTransform"){
    // }
    // void initialize( const ir::Module& m ){};
    // void runOnKernel( ir::IRKernel& k )
    // {
    //     auto analysis = getAnalysis("LoopAnalysis");
    //     analysis::LoopAnalysis* loopAnalysis = static_cast<analysis::LoopAnalysis*>(analysis);
    //     auto loop = loopAnalysis.begin();
    //     auto end = loopAnalysis.end();

    //     for(; loop != end; loop++){
    //         auto blocks = loop->blocks;
    //         auto bb = blocks.begin();
    //         auto bend = blocks.end();
    //         for(; bb != bend; bb++){
    //             auto block = *bb;
    //             auto instructions = block->instructions.begin();
    //             auto iend = block->instructions.end();
    //             for(; instructions != iend; instructions++){
    //                 auto instruction = *instructions;
    //                 std::cout << instruction->toString() << "\n";
    //             }
    //         }


    //     }
    // }
    // void finalize(){};

};

}
extern void reductionKernel();

int main()
{
	// Initialize Ocelot
    // ocelot::initialize();

    // // Create an instance of your custom pass with the desired instruction ID
    transforms::StaticTransform ST; // Replace 123 with the actual instruction ID you are interested in
    // std::ifstream file("test_reduction.ptx");
    // assert(file.is_open());

    // // print the first line of the file 
    // std::string line;
    // for (int i = 0; i < 5; ++i) {
    //     std::getline(file, line);
    //     std::cout << line << std::endl;
    // }
    // std::cout << "1" << std::endl;
    // ir::Module* copyModule = new ir::Module(file, "test_reduction.ptx");
    // std::cout << "2" << std::endl;
	// transforms::PassManager manager(copyModule);
    // std::cout << "3" << std::endl;
    // manager.addPass(&ST);
    // std::cout << "4" << std::endl;
    // manager.runOnKernel("kernel");
    // std::cout << "5" << std::endl;
    // manager.releasePasses();
    ocelot::addPTXPass( ST );
    reductionKernel();
    // Register the pass with Ocelot's pass manager
    // transforms::PassManager::get().addPass(ST);

    // Load the PTX or CUDA source code
    // std::string kernelSource = "path/to/kernel.ptx";
    // ocelot::registerPTXModule(kernelSource);

    // // Launch the kernel (you need to set up the correct execution configuration and arguments)
    // ocelot::launch("kernelFunctionName");

    // transforms::PassManager::get().runOnKernel("kernel");


    // Finalize Ocelot
    // ocelot::finalize();
    
    return 0;
}

