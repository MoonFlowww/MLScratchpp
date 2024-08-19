#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>

double train[][2] = { // *2
    {0,0},
    {1,2},
    {2,4},
    {3,3},
    {4,8},
    {5,10},
    {6,12},
    {7,14},
    {8,16},
    {9,18},
    {10,20}
};

#define train_count (sizeof(train)/sizeof(train[0]))

double predict(double input, double w, double b) {
    return w * input + b;
}

double cost(double w, double b) {
    double MSE = 0.0;

    for (size_t i = 0; i < train_count; ++i) {
        double x = train[i][0];
        double y = predict(x, w, b);
        double r = y - train[i][1]; //prediction error
        MSE += r * r;
    }
    return MSE / train_count;
}

void update() {
    double eps = 1e-2;
    double rate = eps;
    int EPOCH = 500;
    double prev_cost = 0;

    double w = static_cast<double>(std::rand()) / RAND_MAX;
    double b = static_cast<double>(std::rand()) / RAND_MAX;

    for (unsigned e = 0; e < EPOCH; ++e) { // Epoch loop

        double c = cost(w, b);
        double dw = (cost(w + eps, b) - cost(w, b)) / eps;
        double db = (cost(w, b + eps) - cost(w, b)) / eps;

        w -= rate * dw; // gradient update
        b -= rate * db; // gradient update

        std::cout << "Epoch : " << e
            << " | MSE : " << c
            << " | parms: " << w+b
            << " | impr: " << ((c - prev_cost)/prev_cost)*100 << std::endl;
        prev_cost = c;
    }

    std::cout << "\n\n\nYour turn, choose a number: ";
    double input;
    std::cin >> input;
    std::cout << "Prediction: " << predict(input, w, b) << std::endl;
}

int main() {
    std::cout << "Training the single neuron model...\n";
    update();
    return 0;
}
