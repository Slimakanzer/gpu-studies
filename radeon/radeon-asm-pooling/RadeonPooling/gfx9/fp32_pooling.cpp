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
  unsigned n;
  unsigned c;
  unsigned h;
  unsigned w;
  unsigned r;
  unsigned s;
  unsigned pad_h;
  unsigned pad_w;
  unsigned stride_h;
  unsigned stride_w;
  unsigned out_h;
  unsigned out_w;
  unsigned group_size;
  std::string clang;
  std::string asm_source;
  std::string include_dir;
  std::string output_path;

public:
  HalfVectorPooling(int argc, const char **argv, 
    std::string &clang, 
    std::string &asm_source, 
    std::string &include_dir,
    std::string &output_path,
    unsigned &benchmark_times,
    unsigned &group_size,
    unsigned &n,
    unsigned &c,
    unsigned &h,
    unsigned &w,
    unsigned &r,
    unsigned &s,
    unsigned &pad_h,
    unsigned &pad_w,
    unsigned &stride_h,
    unsigned &stride_w)
    : Dispatch(argc, argv, benchmark_times), 
      n{n},
      c{c},
      h{h},
      w{w},
      r{r},
      s{s},
      pad_h{pad_h },
      pad_w{pad_w },
      stride_h{stride_h },
      stride_w{stride_w },
      out_h{(h - r + 2*pad_h)/stride_h + 1 },
      out_w{(w - s + 2*pad_w)/stride_w + 1 },
      group_size{group_size },
      clang{std::move(clang) }, 
      asm_source{std::move(asm_source) }, 
      include_dir{std::move(include_dir) },
      output_path{std::move(output_path) } { }

  bool SetupCodeObject() override {
    std::string clang_call = clang + " " + asm_source + " -x assembler -target amdgcn--amdhsa -mcpu=gfx900 -mno-code-object-v3 -I"
      + include_dir + " -o " + output_path;
    std::cout << "Execute: " << clang_call << std::endl;
    
    if (system(clang_call.c_str())) { output << "Error: build source kernel failed - " << asm_source << std::endl; return false; }

    return LoadCodeObjectFromFile(output_path);
  }

  bool Setup() override {
    if (!AllocateKernarg(2 * sizeof(Buffer*) + 12 * sizeof(unsigned*))) { return false; }
    in = AllocateBuffer(n*c*h*w* sizeof(float));
    for (unsigned i = 0; i < n*c*h*w; ++i) {
      in->Data<float>(i) = (i + 0.25f);
    }
    if (!CopyTo(in)) { output << "Error: failed to copy to local" << std::endl; return false; }
    out = AllocateBuffer(n*c*out_h* out_w* sizeof(float));

    Kernarg(in);
    Kernarg(out);
    Kernarg(&n);
    Kernarg(&c);
    Kernarg(&h);
    Kernarg(&w);
    Kernarg(&r);
    Kernarg(&s);
    Kernarg(&pad_h);
    Kernarg(&pad_w);
    Kernarg(&stride_h);
    Kernarg(&stride_w);
    Kernarg(&out_h);
    Kernarg(&out_w);
    SetGridSize(n * c * group_size);
    SetWorkgroupSize(group_size);
    return true;
  }

  bool Verify() override {
    if (!CopyFrom(out)) { output << "Error: failed to copy from local" << std::endl; return false; }
    bool ok = true;

    for (unsigned n_loop{0}; n_loop < n; n_loop++){
      output << "Batch: " << n_loop << std::endl;

      for (unsigned c_loop{0}; c_loop < c; c_loop++){
        output << "Chanel:" << c_loop << std::endl;
        
        for (unsigned i{0}; i < out_h; i++){
          for (unsigned j{0}; j < out_w; j++){
            float out_value = out->Data<float>(n_loop * c * out_h * out_w + c_loop * out_h * out_w + i * out_w + j);
            float expected_value = -std::numeric_limits<float>::infinity();

            unsigned base_id = n_loop * c * h * w + c_loop * h * w;

            for (unsigned r_loop{0}; r_loop < r; r_loop++){
              for (unsigned s_loop{0}; s_loop < s; s_loop++){
                unsigned id_y = i * stride_h + r_loop - pad_h;
                unsigned id_x = j * stride_w + s_loop - pad_w;

                float value{0};
                if (!(id_x < 0|| id_y < 0 || id_x >= w || id_y >= h))
                  value = in->Data<float>(base_id + id_y * w + id_x);

                expected_value = std::max(expected_value, value); 
              }
            }

            if (expected_value != out_value){
              ok = false;
              output << "ERROR: out_id_x: " << j << " out_id_y: " << i << " ";
              output << "Expected: " << expected_value << " Out: " << out_value << std::endl;
            }
          }
        }

        output << std::endl;
      }

      output << std::endl << std::endl;
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
    ("benchmark_times", value<unsigned>()->default_value(0), "benchmark loop times")
    ("group_size", value<unsigned>()->default_value(64), "group size")
    ("n", value<unsigned>()->default_value(1), "batch size")
    ("c", value<unsigned>()->default_value(1), "input tensor depth")
    ("h", value<unsigned>()->default_value(64), "input tensor height")
    ("w", value<unsigned>()->default_value(64), "input tensor width")
    ("r", value<unsigned>()->default_value(4), "kernel tensor height")
    ("s", value<unsigned>()->default_value(4), "kernel tensor width")
    ("pad_h", value<unsigned>()->default_value(0), "padding height")
    ("pad_w", value<unsigned>()->default_value(0), "padding width")
    ("stride_h", value<unsigned>()->default_value(1), "stride height")
    ("stride_w", value<unsigned>()->default_value(1), "stride width")
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
    unsigned benchmark_times = vm["benchmark_times"].as<unsigned>();
    unsigned group_size = vm["group_size"].as<unsigned>();
    unsigned n = vm["n"].as<unsigned>();
    unsigned c = vm["c"].as<unsigned>();
    unsigned h = vm["h"].as<unsigned>();
    unsigned w = vm["w"].as<unsigned>();
    unsigned r = vm["r"].as<unsigned>();
    unsigned s = vm["s"].as<unsigned>();
    unsigned pad_h = vm["pad_h"].as<unsigned>();
    unsigned pad_w = vm["pad_w"].as<unsigned>();
    unsigned stride_h = vm["stride_h"].as<unsigned>();
    unsigned stride_w = vm["stride_w"].as<unsigned>();

    HalfVectorPooling(argc, argv,
      clang,
      asm_source,
      include_dir,
      output_path,
      benchmark_times,
      group_size,
      n, c, h, w,
      r, s,
      pad_h, pad_w,
      stride_h, stride_w).RunMain();
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}
