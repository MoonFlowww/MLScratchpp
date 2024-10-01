//neuron 1 : f(x)=xw+b
//neuron 2 : f(x)=xw/b



#include <iostream>
#include <cstdlib>
#include <time.h>
#include <sstream>
#include <cmath>

std::stringstream oss;
double train[][2] = {
    {42, 6.4807},
    {81, 9},
    {36, 6},
    {100, 10},

};

#define train_count (sizeof(train) / sizeof(train[0]))

// Prediction using two layers
double predict(double x, double w1, double b1, double w2, double b2) {
    // Layer 1: f1(x) = xw1 + b1
    double layer1_output = x * w1 + b1;
    
    // Layer 2: f2(x) = (f1(x) * w2) / b2
    return (layer1_output * w2) / b2;
}

// Cost function
double cost(double w1, double b1, double w2, double b2) {
    double MSE = 0.0;

    for (size_t i = 0; i < train_count; ++i) {
        double x = train[i][0];
        double y = predict(x, w1, b1, w2, b2);
        double r = y - train[i][1];
        MSE += r * r;
    }
    return MSE / train_count;
}

void update(int epoch) {
    srand(time(0));
    double w1 = (rand() / (double)RAND_MAX) * 10.0;
    double b1 = (rand() / (double)RAND_MAX) * 5.0;
    double w2 = (rand() / (double)RAND_MAX) * 10.0;
    double b2 = (rand() / (double)RAND_MAX) * 5.0;

    double rate = 1e-2; // Learning rate

    // Adam optimizer parameters
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-5;

    // Adam optimizer moment estimates for all parameters
    double m_w1 = 0.0, v_w1 = 0.0;
    double m_b1 = 0.0, v_b1 = 0.0;
    double m_w2 = 0.0, v_w2 = 0.0;
    double m_b2 = 0.0, v_b2 = 0.0;

    double init_cost = cost(w1, b1, w2, b2);
    double prev_c = init_cost;

    // Backups for best parameters
    double backup_w1 = w1, backup_b1 = b1;
    double backup_w2 = w2, backup_b2 = b2;

    for (size_t e = 1; e <= epoch; ++e) {
        // Compute analytical gradients
        double grad_w1 = 0.0, grad_b1 = 0.0;
        double grad_w2 = 0.0, grad_b2 = 0.0;
        size_t n = train_count;

        for (size_t i = 0; i < n; ++i) {
            double x = train[i][0];
            double y_true = train[i][1];
            double layer1_output = x * w1 + b1;
            double y_pred = (layer1_output * w2) / b2; // Two-layer prediction
            double error = y_pred - y_true;

            // Gradient calculations
            grad_w2 += (2.0 / n) * error * layer1_output / b2;
            grad_b2 -= (2.0 / n) * error * (layer1_output * w2) / (b2 * b2);

            grad_w1 += (2.0 / n) * error * (x * w2) / b2;
            grad_b1 += (2.0 / n) * error * (w2) / b2;
        }

        // Update biased first moment estimates
        m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1;
        m_b1 = beta1 * m_b1 + (1 - beta1) * grad_b1;
        m_w2 = beta1 * m_w2 + (1 - beta1) * grad_w2;
        m_b2 = beta1 * m_b2 + (1 - beta1) * grad_b2;

        // Update biased second raw moment estimates
        v_w1 = beta2 * v_w1 + (1 - beta2) * grad_w1 * grad_w1;
        v_b1 = beta2 * v_b1 + (1 - beta2) * grad_b1 * grad_b1;
        v_w2 = beta2 * v_w2 + (1 - beta2) * grad_w2 * grad_w2;
        v_b2 = beta2 * v_b2 + (1 - beta2) * grad_b2 * grad_b2;

        // Compute bias-corrected first moment estimates
        double m_hat_w1 = m_w1 / (1 - pow(beta1, e));
        double m_hat_b1 = m_b1 / (1 - pow(beta1, e));
        double m_hat_w2 = m_w2 / (1 - pow(beta1, e));
        double m_hat_b2 = m_b2 / (1 - pow(beta1, e));

        // Compute bias-corrected second raw moment estimates
        double v_hat_w1 = v_w1 / (1 - pow(beta2, e));
        double v_hat_b1 = v_b1 / (1 - pow(beta2, e));
        double v_hat_w2 = v_w2 / (1 - pow(beta2, e));
        double v_hat_b2 = v_b2 / (1 - pow(beta2, e));

        // Update parameters
        w1 -= rate * m_hat_w1 / (sqrt(v_hat_w1) + eps);
        b1 -= rate * m_hat_b1 / (sqrt(v_hat_b1) + eps);
        w2 -= rate * m_hat_w2 / (sqrt(v_hat_w2) + eps);
        b2 -= rate * m_hat_b2 / (sqrt(v_hat_b2) + eps);

        double current_cost = cost(w1, b1, w2, b2);

        if (current_cost < prev_c) {
            backup_w1 = w1;
            backup_b1 = b1;
            backup_w2 = w2;
            backup_b2 = b2;
            prev_c = current_cost;
        }
    }

    double impr = ((init_cost - prev_c) / prev_c);
    std::cout << "\n\n\nTotal improvement : " << impr * 100  << " %   ||   MPE : " << (impr / epoch) * 100 << " %\n";

    // Restore the best parameters if necessary
    if (cost(backup_w1, backup_b1, backup_w2, backup_b2) < cost(w1, b1, w2, b2)) {
        std::cout << "->Previous model is better than the last version!!! Restoring..." << std::endl;
        w1 = backup_w1;
        b1 = backup_b1;
        w2 = backup_w2;
        b2 = backup_b2;
    }
    std::cout << "Final Parameters : w1->" << w1 << ", b1->" << b1 << ", w2->" << w2 << ", b2->" << b2 << std::endl;

    // Test prediction
    int input = 42;
    double y_pred_bonus = predict(input, w1, b1, w2, b2);
    double target_bonus = 6.4807;
    std::cout << "Prediction: " << y_pred_bonus << std::endl;
    std::cout << "Should be around " << target_bonus << std::endl;
    std::cout << "Error : " << (target_bonus - y_pred_bonus) / y_pred_bonus << std::endl;
}

int main() {
    int epoch = 400000;
    std::cout << "Training has started " << std::endl;
    update(epoch);

    return 0;
}
