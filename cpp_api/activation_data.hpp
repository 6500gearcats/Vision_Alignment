#include <string>
#include <cmath>
#include "layer.hpp"

#ifndef activation_data_hpp
#define activation_data_hpp

inline float __activate_value__(float x, std::string act_func)
{
    if(act_func == "sigmoid"){
        return (float)(1 / (1 + (float)exp(-1 * x)));
    }else if(act_func == "relu"){
        if(x >= 0) return x; else return 0;
    }else if(act_func == "leaky_relu"){
        if(x >= 0) return x; else return (x * (float)0.01);
    }else if(act_func == "tanh"){
        return (float)tanh(x);
    }else if(act_func == "asinh"){
        return (float)asinh(x);
    }else if(act_func == "log"){
        return (float)log(x);
    }else if(act_func == "cos"){
        return cosf(x);
    }else if(act_func == "linear"){
        return x;
    }else{
        std::cout << "NO ACT_FUNC NAME: " << act_func << std::endl;
        return -1.;
    }
}

inline void __activate_value__(Matrix *&x, uint32_t row_size, std::string act_func)
{
    uint32_t i = 0;
    float temp = 0;
    if(act_func == "softmax"){

        while(i < row_size)
        {
            temp += expf(x->get_value(i));
            i++;
        }i=0;

        while(i < row_size)
        {
            x->point_edit(i, expf(x->get_value(i)) / temp);
            i++;
        }

    }else if(act_func == "hardmax"){
        uint32_t highest_idx = 0;

        while(i < row_size - 1)
        {
            if(x->get_value(i) < x->get_value(i+1)){
                highest_idx = i+1;
            }
            i++;
        }i=0;
        
        while(i < row_size)
        {
            if(i == highest_idx){
                x->point_edit(i, 1.0);
            }else{
                x->point_edit(i, 0.0);
            }
            i++;
        }
    }else{
        std::cout << "NO ACT_FUNC NAME: " << act_func << std::endl;
    }
}

inline float __activation_dirivative__(float x, std::string act_func)
{
    if(act_func == "sigmoid"){
        return (float)(x * (1 - x));
    }else if(act_func == "relu"){
        return 1;
    }else if(act_func == "leaky_relu"){
        if(x >= 0) return 1; else return (float)0.01;
    }else if(act_func == "tanh"){
        return (1 - (float)pow((float)tanh(x), 2));
    }else if(act_func == "asinh"){
        return (1 / (float)sqrt(1 + (float)pow((float)sinh(x), 2)));
    }else if(act_func == "log"){
        return (1 / (float)exp(x));
    }else{
        std::cout << "NO ACT_FUNC NAME: " << act_func << std::endl;
        return -1.;
    }
}

inline float __activation_dirivative__(Matrix *x, std::string act_func)
{
    if(act_func == "softmax"){
        return 1;
    }else if(act_func == "hardmax"){
        return 1;
    }else{
        std::cout << "NO ACT_FUNC NAME: " << act_func << std::endl;
        return 1;
    }
}

inline float __cost_value_derivative__(float x, float y, std::string cost_name)
{
    if(cost_name == "HalfMeanSquaredErr"){
        return y - x;
    }else if(cost_name == "MeanSquaredErr"){
        return 2 * (y - x);
    }else if(cost_name == "HalfMeanAbsErr"){
        return (float)abs(y - x);
    }else if(cost_name == "MeanAbsErr"){
        return (float)abs(y - x) * 2;
    }else{
        std::cout << "NO COST FUNC NAME: " << cost_name << std::endl;
        return -1;
    }
}

inline float __cost_value_derivative__(Matrix *x, Matrix *y, uint32_t num_terms, std::string cost_name)
{
    uint32_t i = 0;
    float tot_cost = 0;
    if(cost_name == "HalfMeanSquaredErr"){
        while(i < num_terms)
        {
            tot_cost += (y->get_value(i) - x->get_value(i));
            i++;
        }
        return (2 * tot_cost) / (float)num_terms;
    }else if(cost_name == "MeanSquaredErr"){
        while(i < num_terms)
        {
            tot_cost += 2 * (y->get_value(i) - x->get_value(i));
            i++;
        }
        return (2 * tot_cost) / (float)num_terms;
    }else if(cost_name == "HalfMeanAbsErr"){
        while(i < num_terms)
        {
            if(x->get_value(i) > y->get_value(i))
                tot_cost += 1;
            else
                tot_cost += -1;
            i++;
        }
        return (2 * tot_cost) / (float)num_terms;
    }else if(cost_name == "MeanAbsErr"){
        while(i < num_terms)
        {
            if(x->get_value(i) > y->get_value(i))
                tot_cost += 2;
            else
                tot_cost += -2;
            i++;
        }
        return (2 * tot_cost) / (float)num_terms;
    }else{
        std::cout << "NO COST FUNC NAME: " << cost_name << std::endl;
        return -1;
    }
}

#endif