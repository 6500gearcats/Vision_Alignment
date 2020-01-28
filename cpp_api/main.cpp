#include <iostream>
#include <vector>

#include "optimize_trained_net.hpp"
#include "model.hpp"

void run_test(void)
{
    std::vector<float> d;
    float **inp = (float **)calloc(4, sizeof(float  *));
    for(int i = 0; i < 4; ++i){inp[i] = (float*)calloc(2, sizeof(float));}
    inp[0][0] = 0; inp[0][1] = 0; inp[1][0] = 0; inp[1][1] = 1;
    inp[2][0] = 1; inp[2][1] = 1; inp[3][0] = 1; inp[3][1] = 0;

    float **out = (float **)calloc(4, sizeof(float *));
    for(int i = 0; i < 4; ++i){out[i] = (float*)malloc(sizeof(float));}
    out[0][0] = 0; out[1][0] = 1; out[2][0] = 0; out[3][0] = 1;
    out[0][1] = 1; out[1][1] = 0; out[2][1] = 1; out[3][1] = 0;

    Sequencial *model = new Sequencial(400, 4, false, true, "HalfMeanSquaredErr", 0.001);

    model->add(2, "sigmoid");
    //model->bias();
    model->add(9, "asinh");
    //model->bias();
    model->add(2, "hardmax");

    model->format_input_data(inp, 4);
    model->format_output_data(out);

    model->initialize_global_variables();
    model->run();
}

int main(int argc, const char **argv)
{
    std::cout << argv[0] << std::endl;
    
    system("python3 ~/desktop/github/vision_alignment/TF/train.py");
    std::vector<uint32_t> s, a;
    s.push_back(2);
    s.push_back(5);
    s.push_back(8);
    s.push_back(5);
    s.push_back(2);
    Network *net = new Network(s, a);
    net->run();
    //net->get_data("/Users/idler/Desktop/GitHub/Vision_Alignment/TF/out.txt");

    return 0;
}
