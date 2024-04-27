#include "ocelot/api/interface/ocelot.h"
#include "ocelot/transforms/interface/Pass.h"
// #include "ocelot/analysis/interface/Analysis.h"
#include "ocelot/analysis/interface/LoopAnalysis.h"

#include <iostream>

using namespace transforms;
class StaticTransform : public KernelPass
{
    private:

    public: 
    StaticTransform(){
        KernelPass(Analysis::LoopAnalysis, "StaticTransform");
    }
    virtual void initialize( const ir::Module& m ){};
    virtual void runOnKernel( ir::IRKernel& k )
    {
        auto analysis = getAnalysis("LoopAnalysis");
        analysis::LoopAnalysis* loopAnalysis = static_cast<analysis::LoopAnalysis*>(analysis);
        auto loop = loopAnalysis.begin();
        auto end = loopAnalysis.end();

        for(; loop != end; loop++){
            auto blocks = loop->blocks;
            auto bb = blocks.begin();
            auto bend = blocks.end();
            for(; bb != bend; bb++){
                auto block = *bb;
                auto instructions = block->instructions.begin();
                auto iend = block->instructions.end();
                for(; instructions != iend; instructions++){
                    auto instruction = *instructions;
                    std::cout << instruction->toString() << "\n";
                }
            }


        }
    }
    virtual void finalize(){};

};
extern void reductionKernel();

int main()
{
	// Initialize Ocelot
    ocelot::initialize();

    // Create an instance of your custom pass with the desired instruction ID
    StaticTransform ST; // Replace 123 with the actual instruction ID you are interested in

    // Register the pass with Ocelot's pass manager
    transforms::PassManager::get().addPass(ST);

    // Load the PTX or CUDA source code
    // std::string kernelSource = "path/to/kernel.ptx";
    // ocelot::registerPTXModule(kernelSource);

    // // Launch the kernel (you need to set up the correct execution configuration and arguments)
    // ocelot::launch("kernelFunctionName");

    transforms::PassManager::get().runOnKernel("kernel");


    // Finalize Ocelot
    ocelot::finalize();
    
    return 0;
}


// #include "ocelot/analysis/interface/DivergenceAnalysis.h"

// #include "iostream"

//  namespace transforms {

//    PrettyPrinterPass::PrettyPrinterPass()
//      : KernelPass(Analysis::DivergenceAnalysis, "PrettyPrinterPass") {
//    }

//    void PrettyPrinter::runOnKernel( ir::IRKernel& k ) {
//      Analysis* dfg_structure = getAnalysis(Analysis::DataflowGraphAnalysis);
//      assert(dfg_structure != 0);

//      analysis::DataflowGraph& dfg =
//        *static_cast(dfg_structure);

//      analysis::DataflowGraph::iterator block = ++dfg.begin();
//      analysis::DataflowGraph::iterator blockEnd = --dfg.end();
//      for (; block != blockEnd; block++) {
//        std::cout << "New Basic Block:\n";
//        std::_List_iterator i = block->block()->instructions.begin();
//        std::_List_iterator e = block->block()->instructions.end();
//        while (i != e) {
//          ir::PTXInstruction* inst = static_cast(*i);
//          std::cout <toString() << std::endl;
//          i++;
//        }
//      }
//   }
// }

// #ifndef PRETTY_PRINTER_PASS_H_
// #define PRETTY_PRINTER_PASS_H_

// #include "ocelot/transforms/interface/Pass.h"
// namespace transforms {
// /*! \brief This pass prints the instructions in the Dataflow Graph
//  */
// class PrettyPrinterPass: public KernelPass
// {
// private:

// public:
//   PrettyPrinterPass();
//   virtual ~PrettyPrinterPass() {};
//   virtual void initialize( const ir::Module& m ){};
//   virtual void runOnKernel( ir::IRKernel& k );
//   virtual void finalize(){};
// };
// }
// #endif

