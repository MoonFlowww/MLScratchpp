#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

// Squared data
int train[][2] = {
    {0, 0}, 
    {1, 1}, 
    {2, 4}, 
    {3, 9}, 
    {4, 16},
    {5, 25}, 
    {6, 36}, 
    {7, 49}, 
    {8, 64}, 
    {9, 81}, 
    {10, 100}
};

#define train_count (sizeof(train) / sizeof(train[0]))

class NN {
private:
    int n_input, n_output;
    std::vector<int> hidden_layers;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    double learning_rate = 1; // off
    int current_epoch = 0;

public:
    NN(int n_input, const std::vector<int>& hidden_layers, int n_output)
        : n_input(n_input), n_output(n_output), hidden_layers(hidden_layers) {

        std::vector<int> all_layers = { n_input };
        all_layers.insert(all_layers.end(), hidden_layers.begin(), hidden_layers.end());
        all_layers.push_back(n_output);

        for (size_t i = 1; i < all_layers.size(); ++i) {
            weights.push_back(Eigen::MatrixXd::Random(all_layers[i], all_layers[i - 1]) * std::sqrt(2.0 / all_layers[i - 1])); //  interesting approach
            biases.push_back(Eigen::VectorXd::Zero(all_layers[i]));
        }
    }

    void PrintParms() {
        std::cout << "\033[0mWeights: \n";
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << "Layer " << i + 1 << " weights:\n" << weights[i] << "\n";
        }

        std::cout << "\nBiases: \n";
        for (size_t i = 0; i < biases.size(); ++i) {
            std::cout << "Layer " << i + 1 << " biases:\n" << biases[i] << "\n";
        }
    }

    
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    }

    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& z) {
        Eigen::VectorXd sigmoid_z = sigmoid(z);
        return sigmoid_z.array() * (1 - sigmoid_z.array());
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& x) {
        Eigen::VectorXd activation = x;
        for (size_t i = 0; i < weights.size(); ++i) {
            activation = (i == weights.size() - 1) ? sigmoid(weights[i] * activation + biases[i]) : sigmoid(weights[i] * activation + biases[i]);
        }
        return activation;
    }

    void backpropagate(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
        std::vector<Eigen::VectorXd> activations;
        std::vector<Eigen::VectorXd> zs;
        Eigen::VectorXd activation = input;
        activations.push_back(activation);

        // Forward
        for (size_t i = 0; i < weights.size(); ++i) {
            Eigen::VectorXd z = weights[i] * activation + biases[i];
            zs.push_back(z);
            activation = sigmoid(z);
            activations.push_back(activation);
        }

        // Backward
        Eigen::VectorXd delta = (activations.back() - target).array() * sigmoid_derivative(zs.back()).array();
        for (int i = weights.size() - 1; i >= 0; --i) {
            weights[i] -= learning_rate * (delta * activations[i].transpose());
            biases[i] -= learning_rate * delta;

            if (i > 0) {
                delta = (weights[i].transpose() * delta).array() * sigmoid_derivative(zs[i - 1]).array(); // (target - y)y(1-y) ->y(1-y) : sig
            }
        }
    }

    double cost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
        double mse = 0.0;
        for (int i = 0; i < X.cols(); ++i) {
            Eigen::VectorXd y_pred = forward(X.col(i));
            mse += (y_pred - Y.col(i)).squaredNorm();
        }
        return mse / X.cols();
    }

    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, bool verbose = false, int fold = -1) {
        int initial_epoch = current_epoch;
        for (int e = initial_epoch; e < initial_epoch + epochs; ++e) {
            for (int i = 0; i < X.cols(); ++i) {
                backpropagate(X.col(i), Y.col(i));
            }
            if (verbose && ((e + 1) % 500 == 0 || e == initial_epoch + epochs - 1)) {
                if (fold >= 0) {
                    std::cout << "Fold " << fold + 1 << ", ";
                }
                std::cout << "Epoch " << e + 1 << ", Cost: " << cost(X, Y) << std::endl;
            }
        }
        current_epoch += epochs;
    }


};

//k fold, completly useless in our case, overfitting is interesting in that type of cases which is not a dynamical env
void rollingWindowCrossValidation(NN& nn, int k, int total_epochs) {
    Eigen::MatrixXd X(1, train_count);
    Eigen::MatrixXd Y(1, train_count);
    for (int i = 0; i < train_count; ++i) {
        X(0, i) = train[i][0] / 10.0;  // Normalize input
        Y(0, i) = train[i][1] / 100.0; // the model don't really improve through epochs if i use the same divider. I don't understand why 
    }

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(train_count);
    perm.setIdentity();
    std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
    X = X * perm;
    Y = Y * perm;

    int fold_size = train_count / k;
    int epochs_per_fold = std::ceil(static_cast<double>(total_epochs) / k); // no idea why that was strugguling here ^^
    double total_cost = 0.0;

    for (int fold = 0; fold < k; ++fold) {
        int test_start = fold * fold_size;
        int test_end = (fold == k - 1) ? train_count : test_start + fold_size;
        int test_size = test_end - test_start;

        Eigen::MatrixXd X_test = X.block(0, test_start, 1, test_size);
        Eigen::MatrixXd Y_test = Y.block(0, test_start, 1, test_size);

        Eigen::MatrixXd X_train(1, train_count - test_size);
        Eigen::MatrixXd Y_train(1, train_count - test_size);

        int train_idx = 0;
        for (int i = 0; i < train_count; ++i) {
            if (i < test_start || i >= test_end) {
                X_train(0, train_idx) = X(0, i);
                Y_train(0, train_idx) = Y(0, i);
                train_idx++;
            }
        }

        std::cout << "\n\n\033[35mFold " << fold + 1 << " training:\033[0m" << std::endl;
        nn.train(X_train, Y_train, epochs_per_fold, true, fold);

        double test_cost = nn.cost(X_test, Y_test);
        std::cout << "\033[35mFold " << fold + 1 << ", Test Cost: " << test_cost << "\033[0m" << std::endl;
        total_cost += test_cost;
    }

    std::cout << "Average Test Cost: " << total_cost / k << std::endl;
}

int main() {
    int k = 5;
    int total_epochs = 100000;

    std::vector<int> hidden_layers = { 10, 10 }; // Lh1 : 10 neurons, Lh2 : 7 neurons ...

    NN nn(1, hidden_layers, 1);

    rollingWindowCrossValidation(nn, k, total_epochs);

    Eigen::MatrixXd X_train(1, train_count);
    Eigen::MatrixXd Y_train(1, train_count);
    for (int i = 0; i < train_count; ++i) {
        X_train(0, i) = train[i][0] / 10.0;
        Y_train(0, i) = train[i][1] / 100.0;
    }

    int final_epochs = total_epochs * 0.5 / k;
    std::cout << "\n\n\n \033[36m-> Final training with the whole data, based on " << final_epochs << " epochs." << std::endl;

    
    nn.train(X_train, Y_train, final_epochs, true);
    nn.PrintParms();
    std::cout << "\n\n\n";
    std::cout << "\033[0m \n\n\nPredictions after final training:" << std::endl;
    Eigen::VectorXd input(1);
    Eigen::VectorXd y;
    for (int i = 0; i < train_count; ++i) {
        
        input << train[i][0] / 10.0;

        y = nn.forward(input);
        std::cout << "\033[33mx: " << train[i][0] << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (train[i][1] - y(0)* 100.0) <<"\033[0m" << std::endl;
    }
    input << 13 / 10.0;
    y = nn.forward(input);
    std::cout << "\033[33mx: " << 13 << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (169 - y(0) * 100.0) << "\033[0m" << std::endl;


    input << 100.0 / 10.0;
    y = nn.forward(input);
    std::cout << "\033[33mx: " << 100 << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (10000.0 - y(0) * 100.0) << "\033[0m" << std::endl;
    return 0;
}




