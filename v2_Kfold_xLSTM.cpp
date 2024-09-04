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

#define train_count (sizeof(train) / sizeof(train[0])) //nTuples

class NN {
private:
    int nInput, nOutput; // num input/output
    std::vector<int> initHlayer; // num neurons per H layers ( v.size() == num H Layers)
    std::vector<Eigen::MatrixXd> weights; //1 matrix per layer (bcs x weight for y neurons)
    std::vector<Eigen::VectorXd> biases; // 1 vector per layer (1 B for each neurons)

    Eigen::VectorXd CurrLintput;// output previous layer. Should find a new name  . . .

    std::vector<Eigen::VectorXd> PreAct; // before activation + hist all pred
    std::vector<Eigen::VectorXd> Acted; // same but after activation function

    double lr = 0.01;

    int curr_e;

    // xLSTM parameters
    Eigen::MatrixXd W_f, W_i, W_o, W_C, W_u, W_r;
    Eigen::VectorXd b_f, b_i, b_o, b_C, b_u, b_r;

public:
    NN(int n_input, const std::vector<int>& hidden_layers, int n_output)
        : nInput(n_input), initHlayer(hidden_layers), nOutput(n_output),
        W_i(Eigen::MatrixXd::Random()), W_o(Eigen::MatrixXd::Random()), // w : intput gate, output gate
        W_f(Eigen::MatrixXd::Random()), W_C(Eigen::MatrixXd::Random()), // w : forget gate, cell state
        W_u(Eigen::MatrixXd::Random()), W_r(Eigen::MatrixXd::Random()), // w : additionnal gate * 2
        b_i(Eigen::VectorXd::Zero()), b_o(Eigen::VectorXd::Zero()), // b : intput gate, output gate
        b_f(Eigen::VectorXd::Zero()), b_C(Eigen::VectorXd::Zero()),// b : forget gate, cell state
        b_u(Eigen::VectorXd::Zero()), b_r(Eigen::VectorXd::Zero()) // b : additionnal gate * 2
    {
        std::vector<int> Arch = { nInput };
        Arch.insert(Arch.end(), initHlayer.begin(), initHlayer.end());
        Arch.push_back(nOutput);

        weights.push_back(Eigen::MatrixXd::Random());

        for (int i = 0; i < Arch.size(); ++i) {
            weights.push_back(Eigen::MatrixXd::Random(Arch[i], Arch[-1])); // Init rand parms w
            biases.push_back(Eigen::VectorXd::Zero(Arch[i])); // not rand bcs B is a corr of the exp comp due to the mult 
        }

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

    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
        return (1 / (1 + (-x.array()).exp())).matrix();
    }

    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& z) {
        Eigen::VectorXd sigmoid_z = sigmoid(z);
        return sigmoid_z.array() * (1 - sigmoid_z.array());
    }

    Eigen::VectorXd xLSTM(const Eigen::VectorXd& x, const Eigen::VectorXd& h_prev, const Eigen::VectorXd& c_prev) {
        // Compute xLSTM cell operations
        Eigen::VectorXd concat = (x.transpose().replicate(h_prev.size(), 1).array() + h_prev.array()).matrix();

        Eigen::VectorXd f_t = sigmoid(W_f * concat + b_f);
        Eigen::VectorXd i_t = sigmoid(W_i * concat + b_i);
        Eigen::VectorXd C_tilde = (W_C * concat + b_C).array().tanh().matrix();
        Eigen::VectorXd C_t = f_t.array() * c_prev.array() + i_t.array() * C_tilde.array();
        Eigen::VectorXd o_t = sigmoid(W_o * concat + b_o);
        Eigen::VectorXd h_t = o_t.array() * C_t.array().tanh();

        return h_t;
    }

    void forward(const Eigen::VectorXd& x) {
        Eigen::VectorXd h_prev = Eigen::VectorXd::Zero(initHlayer.back()); // Initialize with zeros
        Eigen::VectorXd c_prev = Eigen::VectorXd::Zero(initHlayer.back()); // Initialize with zeros

        CurrLintput = x;
        for (unsigned i = 0; i < weights.size(); ++i) {
            if (i == weights.size() - 1) {
                // Output layer
                CurrLintput = sigmoid(CurrLintput * weights[i] + biases[i]);
            }
            else {
                // xLSTM layers
                CurrLintput = xLSTM(CurrLintput, h_prev, c_prev);
                h_prev = CurrLintput; // Update hidden state
            }
        }
    }

    Eigen::VectorXd rForward(const Eigen::VectorXd& x) {
        forward(x);
        return CurrLintput;
    }

    void backpropagate(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
        forward(input);

        Eigen::VectorXd di = (Acted.back() - target) * sigmoid_derivative(PreAct.back());

        for (unsigned i = weights.size() - 1; i >= 0; --i) {
            weights[i] -= lr * (di * Acted[i].transpose());
            biases[i] -= lr * di;

            if (i > 0) {
                di = (weights[i].transpose() * di).array() * sigmoid_derivative(PreAct[i - 1]).array();
            }
        }
    }

    double cost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y_target) {
        double MSE = 0;
        for (unsigned i = 0; i < X.cols(); ++i) {
            forward(X.col(i));
            Eigen::VectorXd y_pred = CurrLintput;
            MSE += (y_pred - Y_target.col(i)).squaredNorm();
        }
        return MSE / X.cols();
    }

    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, bool verbose = false, int fold = -1) {
        int initial_e = curr_e;
        for (int e = initial_e; e < initial_e + epochs; ++e) {
            for (int i = 0; X.cols(); ++i) {
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
};

//k fold
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
    X *= perm;
    Y *= perm;

    int f_size = train_count / k;
    int f_epochs = std::ceil(Tepochs / k);
    double Tcost = 0.0;

    for (int fold = 0; fold < k; ++fold) {
        int f_start = fold * f_size;
        int f_end = f_start + f_size;

        Eigen::MatrixXd X_train = X.block(0, f_start, 1, f_size);
        Eigen::MatrixXd Y_train = Y.block(0, f_start, 1, f_size);

        Eigen::MatrixXd X_test(1, train_count - f_size);
        Eigen::MatrixXd Y_test(1, train_count - f_size);

        int train_idx = 0;
        for (int i = 0; i < train_count; ++i) {
            if (i < f_start || i >= f_end) {
                X_train(0, train_idx) = X(0, i);
                Y_train(0, train_idx) = Y(0, i);
                train_idx;
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
    int k = 5;
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

    nn.train(X_train, Y_train, final_epochs, true);
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
    input << 13 / 10.0;
    y = nn.rForward(input);
    std::cout << "\033[33mx: " << 13 << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (169 - y(0) * 100.0) << "\033[0m" << std::endl;

    input << 100.0 / 10.0;
    y = nn.rForward(input);
    std::cout << "\033[33mx: " << 100 << "\033[0m, \033[96my: " << y(0) * 100.0 << "\033[0m || \033[32mround(y): " << round(y(0) * 100.0) << "\033[0m, \033[91mError: " << (10000.0 - y(0) * 100.0) << "\033[0m" << std::endl;
    return 0;
}
