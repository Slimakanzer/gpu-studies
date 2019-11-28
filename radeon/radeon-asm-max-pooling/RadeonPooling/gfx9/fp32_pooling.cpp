#include "dispatch.hpp"
#include <string.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>

using namespace amd::dispatch;
using namespace boost::program_options;

class HalfVectorPooling : public Dispatch {
private:
  Buffer* in;
  Buffer* out;
  unsigned w;
  unsigned s;
  unsigned p;
  unsigned t;
  unsigned w_out;
  std::string clang;
  std::string asm_source;
  std::string include_dir;
  std::string output_path;
  std::string debug_path;

public:
  HalfVectorPooling(int argc, const char **argv, 
    std::string &clang, 
    std::string &asm_source, 
    std::string &include_dir,
    std::string &output_path,
    std::string &debug_path,
    unsigned &debug_size,
    unsigned &benchmark_times,
    unsigned &w,
    unsigned &s,
    unsigned &p,
    unsigned &t)
    : Dispatch(argc, argv, debug_size, benchmark_times), 
      w{w},
      s{s},
      p{p},
      t{t},
      w_out{(w - s + 2*p)/t + 1 },
      clang{std::move(clang) }, 
      asm_source{std::move(asm_source) }, 
      include_dir{std::move(include_dir) },
      output_path{std::move(output_path) },
      debug_path{std::move(debug_path) } {

       }

  bool SetupCodeObject() override {
    std::stringstream stream;

    if (debug) {
      stream << debug->LocalPtr();
      std::string ptr_string = stream.str();
      setenv("ASM_DBG_BUF_ADDR", ptr_string.c_str(), 1);
    }

    stream.str("");
    stream << "cat  " 
      << asm_source << " | " 
      << clang << " -x assembler -target amdgcn--amdhsa -mcpu=gfx900 -mno-code-object-v3" 
      << " -I" << include_dir << " -o " << output_path << " -";

    std::string clang_call = stream.str();
    output << "Execute: " << clang_call << std::endl;
    
    if (system(clang_call.c_str())) { output << "Error: build source kernel failed - " << asm_source << std::endl; return false; }

    return LoadCodeObjectFromFile(output_path);
  }

  bool Setup() override {
    if (!AllocateKernarg(2 * sizeof(Buffer*) + sizeof(unsigned*))) { return false; }
    in = AllocateBuffer(w * sizeof(float));
    for (unsigned i = 0; i < w; ++i) {
      in->Data<float>(i) = i + 0.25f;
    }
    if (!CopyTo(in)) { output << "Error: failed to copy to local" << std::endl; return false; }
    out = AllocateBuffer(w_out * sizeof(float));

    Kernarg(in);
    Kernarg(out);
    Kernarg(&w);
    Kernarg(&s);
    Kernarg(&p);
    Kernarg(&t);
    SetGridSize(w_out);
    SetWorkgroupSize(64);
    return true;
  }

  bool Verify() override {
    if (!CopyFrom(out)) { output << "Error: failed to copy from local" << std::endl; return false; }
    bool ok = true;

    if (debug) {
      if (!CopyFrom(debug)) { output << "Error: failed to copy flom debug buffer" << std::endl; return false; }

      std::ofstream fs(debug_path, std::ios::out | std::ios::binary);
      if (!fs.is_open()) { output << "Error: failed to write debug buffer" << std:: endl; return false; }

      fs.write(debug->Ptr<char>(), debug->Size());
      fs.close();
    }

    for (unsigned i = 0; i < w_out; i++) {
      float out_value = out->Data<float>(i);
      float expected_value = 0.0f; // flt_min

      for (unsigned j = 0; j < s; j++) {
        unsigned index = i * t + j - p;
        float in_value;
        
        if (index < 0 || index > w - 1) {
          in_value = 0.0f; // pad_value
        } 
        else {
          in_value = in->Data<float>(index);
        }
        expected_value = std::max(expected_value, in_value);
      }

      output << "Expected: " << expected_value << " Out value:" << out_value << std::endl;
      if (out_value != expected_value) {
        output << "Error: not equivalent!" << std::endl;
        ok = false;
      }
    }
    output << "Execution time: " << exec_time << " microsec" << std::endl;
    return ok;
  }
};

int main(int argc, const char** argv) {
  try {
    options_description desc("General options");
    desc.add_options()
    ("help,h", "usage message")
    ("clang", value<std::string>()->default_value("clang"), "clang compiler path")
    ("asm", value<std::string>()->default_value("fp32_pooling_x1.s"), "kernel source")
    ("include", value<std::string>()->default_value("."), "inclide directories")
    ("output_path", value<std::string>()->default_value("./output.o"), "kernel binary output path")
    ("debug_path", value<std::string>()->default_value("debug_result"), "debug binary path")
    ("debug_size", value<unsigned>()->default_value(0), "debug size buffer")
    ("benchmark_times", value<unsigned>()->default_value(0), "benchmark loop times")
    ("w", value<unsigned>()->default_value(64), "input tensor width")
    ("s", value<unsigned>()->default_value(4), "kernel tensor width")
    ("p", value<unsigned>()->default_value(0), "padding")
    ("t", value<unsigned>()->default_value(1), "stride")
    ;

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    std::string clang = vm["clang"].as<std::string>();
    std::string asm_source = vm["asm"].as<std::string>();;
    std::string include_dir = vm["include"].as<std::string>();
    std::string output_path = vm["output_path"].as<std::string>();
    std::string debug_path = vm["debug_path"].as<std::string>();
    unsigned debug_size = vm["debug_size"].as<unsigned>();
    unsigned benchmark_times = vm["benchmark_times"].as<unsigned>();
    unsigned w = vm["w"].as<unsigned>();
    unsigned s = vm["s"].as<unsigned>();
    unsigned p = vm["p"].as<unsigned>();
    unsigned t = vm["t"].as<unsigned>();

    HalfVectorPooling(argc, argv,
      clang,
      asm_source,
      include_dir,
      output_path,
      debug_path,
      debug_size,
      benchmark_times,
      w, s, p, t).RunMain();
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}
