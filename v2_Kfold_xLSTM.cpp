#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

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
    int nInput, nOutput;
    std::vector<int> layerSizes;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;

    Eigen::VectorXd CurrLintput;

    std::vector<Eigen::VectorXd> PreAct;
    std::vector<Eigen::VectorXd> Acted;

    double lr = 0.0001;  // Reduced learning rate
    const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-9;
    std::vector<Eigen::MatrixXd> m_weights, v_weights;
    std::vector<Eigen::VectorXd> m_biases, v_biases;
    int t;

    int curr_e;

    // Removed xLSTM parameters as they're not used in the current implementation

public:
    NN(int n_input, const std::vector<int>& hidden_layers, int n_output)
        : nInput(n_input), nOutput(n_output), curr_e(0), t(0)
    {
        if (hidden_layers.empty()) {
            throw std::invalid_argument("Hidden layers cannot be empty");
        }

        layerSizes = { nInput };
        layerSizes.insert(layerSizes.end(), hidden_layers.begin(), hidden_layers.end());
        layerSizes.push_back(nOutput);

        int num_layers = layerSizes.size();
        weights.reserve(num_layers - 1);
        biases.reserve(num_layers - 1);

        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 1; i < layerSizes.size(); ++i) {
            int prevSize = layerSizes[i - 1];
            int currentSize = layerSizes[i];
            double stddev = std::sqrt(2.0 / (prevSize + currentSize));
            std::normal_distribution<> d(0, stddev);

            weights.emplace_back(currentSize, prevSize);
            for (int r = 0; r < currentSize; ++r) {
                for (int c = 0; c < prevSize; ++c) {
                    weights.back()(r, c) = d(gen);
                }
            }
            biases.emplace_back(Eigen::VectorXd::Zero(currentSize));

            m_weights.emplace_back(Eigen::MatrixXd::Zero(currentSize, prevSize));
            v_weights.emplace_back(Eigen::MatrixXd::Zero(currentSize, prevSize));
            m_biases.emplace_back(Eigen::VectorXd::Zero(currentSize));
            v_biases.emplace_back(Eigen::VectorXd::Zero(currentSize));
        }

        PreAct.resize(num_layers - 1);
        Acted.resize(num_layers - 1);
        for (size_t i = 0; i < num_layers - 1; ++i) {
            PreAct[i] = Eigen::VectorXd::Zero(layerSizes[i + 1]);
            Acted[i] = Eigen::VectorXd::Zero(layerSizes[i + 1]);
        }
    }

    void forward(const Eigen::VectorXd& x) {
        if (x.size() != nInput) {
            throw std::invalid_argument("Input size does not match network input size");
        }

        Eigen::VectorXd current = x;
        for (size_t i = 0; i < weights.size(); ++i) {
            PreAct[i] = weights[i] * current + biases[i];
            Acted[i] = relu(PreAct[i]);
            current = Acted[i];
        }

        CurrLintput = current;
    }

    void backpropagate(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
        forward(input);
        std::vector<Eigen::VectorXd> deltas(weights.size());

        deltas.back() = (Acted.back() - target).array() * relu_derivative(PreAct.back()).array();

        for (int i = weights.size() - 1; i >= 0; --i) {
            Eigen::VectorXd prev_layer = (i == 0) ? input : Acted[i - 1];

            m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * deltas[i] * prev_layer.transpose();
            v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * (deltas[i] * prev_layer.transpose()).array().square().matrix();
            Eigen::MatrixXd m_hat = m_weights[i] / (1 - std::pow(beta1, t + 1));
            Eigen::MatrixXd v_hat = v_weights[i] / (1 - std::pow(beta2, t + 1));

            // Correction ici
            weights[i] -= (lr * m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();

            m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * deltas[i];
            v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * deltas[i].array().square().matrix();
            Eigen::VectorXd m_hat_b = m_biases[i] / (1 - std::pow(beta1, t + 1));
            Eigen::VectorXd v_hat_b = v_biases[i] / (1 - std::pow(beta2, t + 1));

            // Correction ici
            biases[i] -= (lr * m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix();

            if (i > 0) {
                deltas[i - 1] = (weights[i].transpose() * deltas[i]).array() * relu_derivative(PreAct[i - 1]).array();
            }
        }
        t++;
    }

    Eigen::VectorXd relu(const Eigen::VectorXd& x) {
        return x.array().max(0);
    }

    Eigen::VectorXd relu_derivative(const Eigen::VectorXd& z) {
        return (z.array() > 0).cast<double>();
    }

    double cost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y_target) {
        double MSE = 0;
        for (int i = 0; i < X.cols(); ++i) {
            forward(X.col(i));
            Eigen::VectorXd y_pred = CurrLintput;
            MSE += (y_pred - Y_target.col(i)).squaredNorm();
        }
        return MSE / X.cols();
    }

    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, bool verbose, int fold) {
        int initial_e = curr_e;
        for (int e = initial_e; e < initial_e + epochs; ++e) {
            for (int i = 0; i < X.cols(); ++i) {
                backpropagate(X.col(i), Y.col(i));
            }

            if (verbose && ((e + 1) % 500 == 0 || e == initial_e + epochs - 1)) {
                if (fold >= 0) {
                    std::cout << "Fold " << fold + 1 << ", ";
                }
                std::cout << "Epoch " << e + 1 << ", Cost: " << cost(X, Y) << std::endl;
            }
        }
        curr_e += epochs;
    }

    Eigen::VectorXd rForward(const Eigen::VectorXd& x) {
        forward(x);
        return CurrLintput;
    }

    void PrintParms() {
        std::cout << "\n\n\n\033[0mWeights:";
        for (unsigned i = 0; i < weights.size(); ++i) {
            std::cout << "\nLayer " << i + 1 << " weights:\n" << weights[i] << "\n";
        }

        std::cout << "\n\nBiases: \n";
        for (unsigned i = 0; i < biases.size(); ++i) {
            std::cout << "Layer " << i + 1 << " biases:\n" << biases[i] << "\n";
        }
    }
};

void Kfold(NN& nn, int k, int Tepochs) {
    Eigen::MatrixXd X(1, train_count);
    Eigen::MatrixXd Y(1, train_count);

    for (size_t i = 0; i < train_count; ++i) {
        X(0, i) = train[i][0] / 10.0;
        Y(0, i) = train[i][1] / 100.0;
    }

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(train_count);
    perm.setIdentity();
    std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
    X = (X * perm).eval();
    Y = (Y * perm).eval();

    int f_size = train_count / k;
    int f_epochs = std::ceil(static_cast<double>(Tepochs) / k);
    double Tcost = 0.0;

    for (int fold = 0; fold < k; ++fold) {
        int f_start = fold * f_size;
        int f_end = std::min(f_start + f_size, static_cast<int>(train_count));

        Eigen::MatrixXd X_test = X.block(0, f_start, 1, f_end - f_start);
        Eigen::MatrixXd Y_test = Y.block(0, f_start, 1, f_end - f_start);

        Eigen::MatrixXd X_train(1, train_count - (f_end - f_start));
        Eigen::MatrixXd Y_train(1, train_count - (f_end - f_start));

        int train_idx = 0;
        for (int i = 0; i < train_count; ++i) {
            if (i < f_start || i >= f_end) {
                X_train(0, train_idx) = X(0, i);
                Y_train(0, train_idx) = Y(0, i);
                ++train_idx;
            }
        }

        std::cout << "\n\n\033[35mFold " << fold + 1 << " training:\033[0m" << std::endl;
        nn.train(X_train, Y_train, f_epochs, true, fold);

        double test_cost = nn.cost(X_test, Y_test);
        std::cout << "\033[35mFold " << fold + 1 << ", Test Cost: " << test_cost << "\033[0m" << std::endl;
        Tcost += test_cost;
    }
    std::cout << "Average Test Cost: " << Tcost / k << std::endl;
}

int main() {
    int k = train_count;
    int total_epochs = 100000;

    std::vector<int> hidden_layers = { 10, 10 };

    NN nn(1, hidden_layers, 1);

    Kfold(nn, k, total_epochs);

    Eigen::MatrixXd X_train(1, train_count);
    Eigen::MatrixXd Y_train(1, train_count);
    for (int i = 0; i < train_count; ++i) {
        X_train(0, i) = train[i][0] / 10.0;
        Y_train(0, i) = train[i][1] / 100.0;
    }

    int final_epochs = total_epochs * 0.5 / k;
    std::cout << "\n\n\n \033[36m-> Final training with the whole data, based on " << final_epochs << " epochs." << std::endl;

    nn.train(X_train, Y_train, final_epochs, true, -1);
    nn.PrintParms();
    std::cout << "\n\n\n";
    std::cout << "\033[0m \n\n\nPredictions after final training:" << std::endl;
    Eigen::VectorXd input(1);
    Eigen::VectorXd y;
    for (int i = 0; i < train_count; ++i) {
        input << train[i][0] / 10.0;
        y = nn.rForward(input);
        std::cout << "\033[33mx: " << train[i][0] << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (train[i][1] - y(0) * 100.0) << "\033[0m" << std::endl;
    }
    input << 13.0 / 10.0;
    y = nn.rForward(input);
    std::cout << "\033[33mx: " << 13 << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (169 - y(0) * 100.0) << "\033[0m" << std::endl;

    input << 100.0 / 10.0;
    y = nn.rForward(input);
    std::cout << "\033[33mx: " << 100 << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (10000.0 - y(0) * 100.0) << "\033[0m" << std::endl;
    return 0;
}
