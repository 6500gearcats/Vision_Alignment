#ifndef activation_data_hpp
#define activation_data_hpp

#include <string>
#include <cmath>

template <class T>
inline T __activate_value__(T x, std::string act_func)
{
    if(act_func == "sigmoid"){
        return (T)(1 / (1 + (T)exp(-1 * x)));
    }else if(act_func == "relu"){
        if(x >= 0) return x; else return 0;
    }else if(act_func == "leaky_relu"){
        if(x >= 0) return x; else return (x * (T)0.01);
    }else if(act_func == "tanh"){
        return (T)tanh(x);
    }else if(act_func == "asinh"){
        return (T)asinh(x);
    }else if(act_func == "log"){
        return (T)log(x);
    }else{
        std::cout << "NO ACT_FUNC NAME: " << act_func << std::endl;
        return -1.;
    }
}

template <class T>
inline void __activate_value__(T *&x, uint32_t row_size, std::string act_func)
{
    uint32_t i = 0;
    T temp = 0;
    if(act_func == "softmax"){
        while(i < row_size)
        {
            temp += (T)exp(x[i]);
            i++;
        }i=0;
        while(i < row_size)
        {
            x[i] = x[i] / temp;
            i++;
        }
    }else if(act_func == "hardmax"){
        uint32_t highest_idx = 0;
        while(i < row_size - 1)
        {
            if(x[i] < x[i+1])
                highest_idx = i+1;
            i++;
        }i=0;
        while(i < row_size)
        {
            if(i == highest_idx)
                x[i] = 1;
            else
                x[i] = 0;
        }
    }else{
        std::cout << "NO ACT_FUNC NAME: " << act_func << std::endl;
    }
}

template <class T>
inline T __activation_dirivative__(T x, std::string act_func)
{
    if(act_func == "sigmoid"){
        return (T)(x * (1 - x));
    }else if(act_func == "relu"){
        return 1;
    }else if(act_func == "leaky_relu"){
        if(x >= 0) return 1; else return (T)0.01;
    }else if(act_func == "tanh"){
        return (1 - (T)pow((T)tanh(x), 2));
    }else if(act_func == "asinh"){
        return (1 / (T)sqrt(1 + (T)pow((T)sinh(x), 2)));
    }else if(act_func == "log"){
        return (1 / (T)exp(x));
    }else{
        std::cout << "NO ACT_FUNC NAME: " << act_func << std::endl;
        return -1.;
    }
}

template <class T>
inline T __cost_value_derivative__(T x, T y, std::string cost_name)
{
    if(cost_name == "HalfMeanSquaredErr"){
        return y - x;
    }else if(cost_name == "MeanSquaredErr"){
        return 2 * (y - x);
    }else if(cost_name == "HalfMeanAbsErr"){
        return (T)abs(y - x);
    }else if(cost_name == "MeanAbsErr"){
        return (T)abs(y - x) * 2;
    }else{
        std::cout << "NO COST FUNC NAME: " << cost_name << std::endl;
        return -1;
    }
}

template <class T>
inline T __cost_value_derivative__(T *x, T *y, uint32_t num_terms, std::string cost_name)
{
    uint32_t i = 0;
    T tot_cost = 0;
    if(cost_name == "HalfMeanSquaredErr"){
        while(i < num_terms)
        {
            tot_cost += (y[i] - x[i]);
            i++;
        }
        return (2 * tot_cost) / (T)num_terms;
    }else if(cost_name == "MeanSquaredErr"){
        while(i < num_terms)
        {
            tot_cost += 2 * (y[i] - x[i]);
            i++;
        }
        return (2 * tot_cost) / (T)num_terms;
    }else if(cost_name == "HalfMeanAbsErr"){
        while(i < num_terms)
        {
            if(x[i] > y[i])
                tot_cost += 1;
            else
                tot_cost += -1;
            i++;
        }
        return (2 * tot_cost) / (T)num_terms;
    }else if(cost_name == "MeanAbsErr"){
        while(i < num_terms)
        {
            if(x[i] > y[i])
                tot_cost += 2;
            else
                tot_cost += -2;
            i++;
        }
        return (2 * tot_cost) / (T)num_terms;
    }else{
        std::cout << "NO COST FUNC NAME: " << cost_name << std::endl;
        return -1;
    }
}

#endif