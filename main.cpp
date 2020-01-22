#include <iostream>
#include <vector>

#include "model.hpp"

int main(int argc, const char **argv)
{
    std::cout << argv[0] << std::endl;
    std::vector<float> d;
    std::vector<std::vector<float> > inp;
    std::vector<std::vector<float> > out;

    d.push_back(0);d.push_back(0);inp.push_back(d);d.clear();
    d.push_back(0);d.push_back(1);inp.push_back(d);d.clear();
    d.push_back(1);d.push_back(0);inp.push_back(d);d.clear();
    d.push_back(1);d.push_back(1);inp.push_back(d);d.clear();

    d.push_back(0);out.push_back(d);d.clear();
    d.push_back(1);out.push_back(d);d.clear();
    d.push_back(1);out.push_back(d);d.clear();
    d.push_back(0);out.push_back(d);d.clear();


    Sequencial *model = new Sequencial(3, true, true);

/*
    model->format_input_data(inp);
    model->format_output_data(out);
    model->add(2, "sigmoid");
    model->add(4, "sigmoid");
    model->add(1, "sigmoid");

    model->initialize_global_variables();
    model->run();
*/
    return 0;
}