// f(x) = (log2(x+d)^c) * w1 + b1

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

// Prediction using a single neuron with (log2(x+d)^c) * w1 + b1
double predict(double x, double w1, double b1, double c, double d) {
    // f(x) = (log2(x+d)^c) * w1 + b1
    return pow(log2(x + d), c) * w1 + b1;
}


// Cost function
double cost(double w1, double b1, double c, double d) {
    double MSE = 0.0;

    for (size_t i = 0; i < train_count; ++i) {
        double x = train[i][0];
        double y = predict(x, w1, b1, c, d);
        double r = y - train[i][1];
        MSE += r * r;
    }
    return MSE / train_count;
}


void update(int epoch) {
    srand(time(0));
    double w1 = (rand() / (double)RAND_MAX) * 10.0;
    double b1 = (rand() / (double)RAND_MAX) * 5.0;
    double c = (rand() / (double)RAND_MAX) * 5.0;
    double d = (rand() / (double)RAND_MAX) * 5.0; // Initialize d

    double rate = 1e-2; // Learning rate

    // Adam optimizer parameters
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;

    // Adam optimizer moment estimates for all parameters
    double m_w1 = 0.0, v_w1 = 0.0;
    double m_b1 = 0.0, v_b1 = 0.0;
    double m_c = 0.0, v_c = 0.0;
    double m_d = 0.0, v_d = 0.0; // Add for d

    double init_cost = cost(w1, b1, c, d);
    double prev_c = init_cost;

    // Backups for best parameters
    double backup_w1 = w1, backup_b1 = b1;
    double backup_c = c;
    double backup_d = d;

    for (size_t e = 1; e <= epoch; ++e) {
        // Compute analytical gradients
        double grad_w1 = 0.0, grad_b1 = 0.0, grad_c = 0.0, grad_d = 0.0;
        size_t n = train_count;

        for (size_t i = 0; i < n; ++i) {
            double x = train[i][0];
            double y_true = train[i][1];
            double log2_x = log2(x + d);
            double log2_x_c = pow(log2_x, c);
            double y_pred = log2_x_c * w1 + b1; // Single neuron prediction
            double error = y_pred - y_true;

            // Gradients with respect to w1, b1, c, and d
            grad_w1 += (2.0 / n) * error * log2_x_c;
            grad_b1 += (2.0 / n) * error;
            grad_c += (2.0 / n) * error * w1 * log2_x_c * log2_x * log(2.0); // Derivative w.r.t c
            grad_d += (2.0 / n) * error * w1 * c * log2_x_c / (x + d); // Derivative w.r.t d
        }

        // Update biased first moment estimates
        m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1;
        m_b1 = beta1 * m_b1 + (1 - beta1) * grad_b1;
        m_c = beta1 * m_c + (1 - beta1) * grad_c;
        m_d = beta1 * m_d + (1 - beta1) * grad_d;

        // Update biased second raw moment estimates
        v_w1 = beta2 * v_w1 + (1 - beta2) * grad_w1 * grad_w1;
        v_b1 = beta2 * v_b1 + (1 - beta2) * grad_b1 * grad_b1;
        v_c = beta2 * v_c + (1 - beta2) * grad_c * grad_c;
        v_d = beta2 * v_d + (1 - beta2) * grad_d * grad_d;

        // Compute bias-corrected first moment estimates
        double m_hat_w1 = m_w1 / (1 - pow(beta1, e));
        double m_hat_b1 = m_b1 / (1 - pow(beta1, e));
        double m_hat_c = m_c / (1 - pow(beta1, e));
        double m_hat_d = m_d / (1 - pow(beta1, e));

        // Compute bias-corrected second raw moment estimates
        double v_hat_w1 = v_w1 / (1 - pow(beta2, e));
        double v_hat_b1 = v_b1 / (1 - pow(beta2, e));
        double v_hat_c = v_c / (1 - pow(beta2, e));
        double v_hat_d = v_d / (1 - pow(beta2, e));

        // Update parameters
        w1 -= rate * m_hat_w1 / (sqrt(v_hat_w1) + eps);
        b1 -= rate * m_hat_b1 / (sqrt(v_hat_b1) + eps);
        c -= rate * m_hat_c / (sqrt(v_hat_c) + eps);
        d -= rate * m_hat_d / (sqrt(v_hat_d) + eps);

        double current_cost = cost(w1, b1, c, d);

        if (current_cost < prev_c) {
            backup_w1 = w1;
            backup_b1 = b1;
            backup_c = c;
            backup_d = d;
            prev_c = current_cost;
        }
    }

    double impr = ((init_cost - prev_c) / prev_c);
    std::cout << "\n\n\nTotal improvement : " << impr * 100  << " %   ||   MPE : " << (impr / epoch) * 100 << " %\n";

    // Restore the best parameters if necessary
    if (cost(backup_w1, backup_b1, backup_c, backup_d) < cost(w1, b1, c, d)) {
        std::cout << "->Previous model is better than the last version!!! Restoring..." << std::endl;
        w1 = backup_w1;
        b1 = backup_b1;
        c = backup_c;
        d = backup_d;
    }
    std::cout << "Final Parameters : w1->" << w1 << ", b1->" << b1 << ", c->" << c << ", d->" << d << std::endl;

    // Test prediction
    int input = 49;
    double y_pred_bonus = predict(input, w1, b1, c, d);
    double target_bonus = 7;
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
