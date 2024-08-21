#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <time.h>
#include <sstream>
std::stringstream oss;
double train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 3},
    {4, 8},
    {5, 10},
    {6, 12},
    {7, 14},
    {8, 16},
    {9, 18},
    {10, 20}
};

#define train_count (sizeof(train) / sizeof(train[0]))

double predict(int x, double w, double b) {
    return x * w + b;
}

double cost(double w, double b) {
    double MSE = 0.0;

    for (size_t i = 0; i < train_count; ++i) {
        double x = train[i][0];
        double y = predict(x, w, b);
        double r = y - train[i][1]; 
        MSE += r * r;
    }
    return MSE / train_count;
}

void update(int epoch) {
    srand(time(0));
    double w = (rand() / (double)RAND_MAX) * 10.0;
    double b = (rand() / (double)RAND_MAX) * 5.0;

    double eps = 1e-3;
    double rate = 1e-3;

    double init_cost = cost(w, b);
    double prev_c = init_cost;

    double backup_w = w;
    double backup_b = b;

    for (size_t e = 0; e < epoch; ++e) {
        double c = cost(w, b);
        double dw = (cost(w + eps, b) - cost(w, b)) / eps;
        double db = (cost(w, b + eps) - cost(w, b)) / eps;
        w -= rate * dw; // gradient
        b -= rate * db;

        double current_cost = cost(w, b);


        oss << "\n\n\nEpoch : " << e
            << "\n  w : " << w
            << "\n  b : " << b
            << "\n  cost :" << current_cost;

        std::cout << oss.str() << std::endl;
        oss.str("");

        if (current_cost < prev_c) {
            std::cout << "-> Better parameters found!!!" << std::endl;
            backup_w = w;
            backup_b = b;
            prev_c = current_cost;
        }
    }
    double impr = ((init_cost - prev_c) / prev_c);
    std::cout << "\n\n\nTotal improvement : " << impr*100  << " %   ||   MPE : " << (impr/epoch)*100 << " %\n";
    // After the loop, if the backup parameters are better, restore them
    if (cost(backup_w, backup_b) < cost(w, b)) {
        std::cout << "->Previous model is better than the last version!!! Restoring..." << std::endl;
        w = backup_w;
        b = backup_b;
    }

    std::cout << "\n\n\nYour turn, choose a number: ";
    double input;
    std::cin >> input;
    std::cout << "Prediction: " << predict(input, w, b) << std::endl;
}


int main() {
    int epoch = 4000;
    std::cout << "Training has started " << std::endl;
    update(epoch);

    return 0;
}
