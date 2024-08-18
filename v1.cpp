#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>
#include <iomanip>
#include <sstream>
#include <Eigen/Dense>

std::stringstream oss;


double train[][2] = { // square
    {0,0},
    {1,1},
    {2,4},
    {3,9},
    {4,16},
    {5,25},
    {6,36},
    {7,49},
    {8,64},
    {9,81},
    {10,100}
};
#define train_count (sizeof(train)/sizeof(train[0]))

class NN {
private:
    int n_input;
    int n_hidden;
    int nL_hidden;
    int n_output;

    double eps = 1e-2;
    double rate = 1e-6;

    Eigen::MatrixXd w_input;
    Eigen::MatrixXd w_hidden;
    Eigen::MatrixXd w_output;

    Eigen::VectorXd hidden_layer;
    Eigen::VectorXd output_layer;
    Eigen::VectorXd final_output;

public:
    NN(int n_input, int n_hidden, int nL_hidden, int n_output)
        : n_input(n_input), n_hidden(n_hidden), nL_hidden(nL_hidden), n_output(n_output),
        w_input(Eigen::MatrixXd::Random(n_hidden, n_input)),
        w_hidden(Eigen::MatrixXd::Random(n_output, n_hidden)),
        w_output(Eigen::MatrixXd::Random(1, n_output)) {}

    double cost() {
        double total_cost = 0.0;
        for (int i = 0; i < train_count; ++i) {
            double input = train[i][0];
            double target = train[i][1];
            double prediction = predict(input);
            double error = prediction - target;
            total_cost += error * error;
        }
        return total_cost / train_count;
    }

    double predict(double input) {
        Eigen::VectorXd input_vector(1);
        input_vector << input;

        // Forward pass
        // Calculate hidden layer activations
        hidden_layer = (w_input * input_vector).array().tanh();

        // Calculate output layer activations
        output_layer = (w_hidden * hidden_layer).array().tanh();

        // Calculate final output
        final_output = (w_output * output_layer).array().tanh();

        return final_output[0];
    }

    void update() {
        for (unsigned e = 0; e < 500; ++e) { // Epoch Loop
            Eigen::MatrixXd w_input_grad = Eigen::MatrixXd::Zero(w_input.rows(), w_input.cols());
            Eigen::MatrixXd w_hidden_grad = Eigen::MatrixXd::Zero(w_hidden.rows(), w_hidden.cols());
            Eigen::MatrixXd w_output_grad = Eigen::MatrixXd::Zero(w_output.rows(), w_output.cols());

            double prev_cost = cost();

            for (int i = 0; i < train_count; ++i) {
                double input = train[i][0];
                double target = train[i][1];

                // Forward pass
                predict(input);

                // Compute loss
                Eigen::VectorXd target_vector(1);
                target_vector << target;
                Eigen::VectorXd error = final_output - target_vector;

                // Backward pass
                Eigen::VectorXd d_output = error.array() * (1 - final_output.array().square());
                Eigen::VectorXd d_hidden = (w_hidden.transpose() * d_output).array() * (1 - hidden_layer.array().square());

                w_output_grad += d_output * output_layer.transpose();
                w_hidden_grad += d_output * hidden_layer.transpose();
                w_input_grad += d_hidden * Eigen::VectorXd::Constant(1, input);
            }

            // Average gradients
            w_input_grad /= train_count;
            w_hidden_grad /= train_count;
            w_output_grad /= train_count;

            // Update weights
            w_input -= rate * w_input_grad;
            w_hidden -= rate * w_hidden_grad;
            w_output -= rate * w_output_grad;

            double current_cost = cost();
            double improvement = ((prev_cost - current_cost) / prev_cost) * 100;

            oss << "Epoch : " << e
                << " | MSE : " << current_cost
                << " | Improvement of : " << improvement << "%\n";


        }
        std::cout << oss.str() << "\n\n\n\n\n" << std::endl;
    }
};

int main() {
    NN nn(1, 100, 30, 1);

    std::cout << "Training the neural network...\n";
    nn.update();

    std::cout << "Your turn, choose a number: ";
    double input;
    std::cin >> input;
    std::cout << "Prediction: " << nn.predict(input) << std::endl;

    return 0;
}